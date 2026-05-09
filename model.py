"""
CineSync — Model Training Pipeline
====================================

"""

import ast
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
MOVIES_CSV = DATA_DIR / "tmdb_5000_movies.csv"
CREDITS_CSV = DATA_DIR / "tmdb_5000_credits.csv"
OUT_MOVIES = Path("movie_dict.pkl")
OUT_SIMILARITY = Path("similarity.pkl")

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------

def load_raw_data() -> pd.DataFrame:
    log.info("Loading raw data …")
    movies = pd.read_csv(MOVIES_CSV)
    credits = pd.read_csv(CREDITS_CSV)
    df = movies.merge(credits, on="title")
    log.info("  Merged shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# 2. PARSING HELPERS
# ---------------------------------------------------------------------------

def parse_json_field(obj: str, key: str = "name", limit: Optional[int] = None) -> list[str]:
    
    try:
        data = ast.literal_eval(obj)
        items = [i[key] for i in data if key in i]
        return items[:limit] if limit else items
    except (ValueError, SyntaxError):
        return []


def get_director(obj: str) -> list[str]:
    try:
        data = ast.literal_eval(obj)
        return [i["name"] for i in data if i.get("job") == "Director"]
    except (ValueError, SyntaxError):
        return []


def collapse_spaces(tokens: list[str]) -> list[str]:
    return [t.replace(" ", "") for t in tokens]


# ---------------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    
    log.info("Engineering features …")
    df = df[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]].dropna().copy()

    df["genres"]   = df["genres"].apply(parse_json_field).apply(collapse_spaces)
    df["keywords"] = df["keywords"].apply(parse_json_field).apply(collapse_spaces)
    df["cast"]     = df["cast"].apply(lambda x: parse_json_field(x, limit=3)).apply(collapse_spaces)
    df["crew"]     = df["crew"].apply(get_director).apply(collapse_spaces)
    df["overview"] = df["overview"].apply(lambda x: x.split() if isinstance(x, str) else [])

    # Director weighted 3× — domain-aware feature boosting
    df["tags"] = (
        df["overview"]
        + df["genres"]
        + df["keywords"]
        + df["cast"]
        + df["crew"] * 3
    )

    result = df[["movie_id", "title", "tags", "genres"]].copy()
    result["tags"] = result["tags"].apply(lambda tokens: " ".join(tokens).lower())
    log.info("  Feature frame shape: %s", result.shape)
    return result


# ---------------------------------------------------------------------------
# 4. TEXT PREPROCESSING
# ---------------------------------------------------------------------------

def stem_tags(df: pd.DataFrame) -> pd.DataFrame:
    
    log.info("Stemming tags …")
    ps = PorterStemmer()

    def _stem(text: str) -> str:
        return " ".join(ps.stem(w) for w in text.split())

    df = df.copy()
    df["tags"] = df["tags"].apply(_stem)
    return df


# ---------------------------------------------------------------------------
# 5. VECTORISATION & SIMILARITY
# ---------------------------------------------------------------------------

def build_similarity(df: pd.DataFrame) -> np.ndarray:
    
    log.info("Vectorising with TF-IDF …")
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    vectors = tfidf.fit_transform(df["tags"]).toarray()
    log.info("  Vector shape: %s", vectors.shape)

    log.info("Computing cosine similarity matrix …")
    sim = cosine_similarity(vectors)
    log.info("  Similarity matrix shape: %s", sim.shape)
    return sim


# ---------------------------------------------------------------------------
# 6. OFFLINE EVALUATION  (Precision@K)
# ---------------------------------------------------------------------------

def evaluate_precision_at_k(
    df: pd.DataFrame,
    sim: np.ndarray,
    k: int = 10,
    sample_size: int = 200,
) -> float:
    
    log.info("Evaluating Precision@%d over %d samples …", k, sample_size)
    np.random.seed(42)
    indices = np.random.choice(len(df), size=min(sample_size, len(df)), replace=False)

    precisions: list[float] = []
    for idx in indices:
        query_genres = set(df.iloc[idx]["genres"])
        top_k_indices = np.argsort(sim[idx])[::-1][1 : k + 1]
        hits = sum(
            1
            for j in top_k_indices
            if set(df.iloc[j]["genres"]) & query_genres  # non-empty intersection
        )
        precisions.append(hits / k)

    mean_p = float(np.mean(precisions))
    log.info("  Mean Precision@%d = %.4f", k, mean_p)
    return mean_p


# ---------------------------------------------------------------------------
# 7. SERIALISE ARTEFACTS
# ---------------------------------------------------------------------------

def save_artefacts(df: pd.DataFrame, sim: np.ndarray) -> None:
    log.info("Saving artefacts …")
    with OUT_MOVIES.open("wb") as f:
        pickle.dump(df.to_dict(), f)
    with OUT_SIMILARITY.open("wb") as f:
        pickle.dump(sim, f)
    log.info("  Saved: %s, %s", OUT_MOVIES, OUT_SIMILARITY)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    raw = load_raw_data()
    feature_df = build_features(raw)
    stemmed_df = stem_tags(feature_df)
    sim_matrix = build_similarity(stemmed_df)

    precision = evaluate_precision_at_k(stemmed_df, sim_matrix, k=10, sample_size=200)
    print(f"\n✅  Precision@10 (genre-overlap proxy): {precision:.2%}")

    save_artefacts(stemmed_df, sim_matrix)
    print("🚀  Model training complete! Artefacts ready for app.py.")


if __name__ == "__main__":
    main()
