"""
CineSync — Streamlit App
=========================

"""

import logging
import os
import pickle
from typing import Optional

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
load_dotenv()  # reads .env file if present (safe no-op if absent)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

TMDB_API_KEY: str = os.environ.get("TMDB_API_KEY", "")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER_POSTER = "https://via.placeholder.com/500x750?text=No+Poster"

# ---------------------------------------------------------------------------
# 1. DATA LOADING  (cached — loaded once per session)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data() -> tuple[Optional[pd.DataFrame], Optional[object]]:
    
    try:
        with open("movie_dict.pkl", "rb") as f:
            movies_df = pd.DataFrame(pickle.load(f))
        with open("similarity.pkl", "rb") as f:
            similarity_matrix = pickle.load(f)
        return movies_df, similarity_matrix
    except FileNotFoundError as exc:
        log.error("Artefact not found: %s", exc)
        return None, None
    except Exception as exc:  # noqa: BLE001
        log.error("Unexpected load error: %s", exc)
        return None, None


# ---------------------------------------------------------------------------
# 2. TMDB API
# ---------------------------------------------------------------------------

def fetch_poster(movie_id: int) -> str:
    
    if not TMDB_API_KEY:
        log.warning("TMDB_API_KEY not set — skipping poster fetch.")
        return PLACEHOLDER_POSTER

    url = f"{TMDB_BASE_URL}/movie/{movie_id}"
    params = {"api_key": TMDB_API_KEY, "language": "en-US"}

    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        poster_path = data.get("poster_path")
        return f"{POSTER_BASE_URL}{poster_path}" if poster_path else PLACEHOLDER_POSTER
    except requests.exceptions.Timeout:
        log.warning("TMDB request timed out for movie_id=%s", movie_id)
        return PLACEHOLDER_POSTER
    except requests.exceptions.HTTPError as exc:
        log.warning("TMDB HTTP error for movie_id=%s: %s", movie_id, exc)
        return PLACEHOLDER_POSTER
    except Exception as exc:  # noqa: BLE001
        log.warning("Unexpected error fetching poster for movie_id=%s: %s", movie_id, exc)
        return PLACEHOLDER_POSTER


# ---------------------------------------------------------------------------
# 3. RECOMMENDATION ENGINE
# ---------------------------------------------------------------------------

def recommend(
    movie: str,
    genre_filter: str,
    movies_df: pd.DataFrame,
    similarity_matrix,
    top_n: int = 5,
    candidate_pool: int = 50,
) -> tuple[list[str], list[str]]:
    
    matches = movies_df[movies_df["title"] == movie]
    if matches.empty:
        log.warning("Movie not found in dataset: %s", movie)
        return [], []

    movie_index: int = matches.index[0]
    distances = similarity_matrix[movie_index]

    ranked = sorted(enumerate(distances), key=lambda x: x[1], reverse=True)
    candidates = ranked[1 : candidate_pool + 1]

    titles: list[str] = []
    posters: list[str] = []

    for idx, _score in candidates:
        row = movies_df.iloc[idx]
        if genre_filter != "All" and genre_filter not in row["genres"]:
            continue
        titles.append(row["title"])
        posters.append(fetch_poster(int(row["movie_id"])))
        if len(titles) == top_n:
            break

    return titles, posters


# ---------------------------------------------------------------------------
# 4. PAGE CONFIG
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CineSync",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 5. LOAD DATA — fail fast with a clear message
# ---------------------------------------------------------------------------

movies, similarity = load_data()

if movies is None:
    st.error(
        "⚠️ Model artefacts not found. "
        "Run `python model.py` to generate `movie_dict.pkl` and `similarity.pkl`."
    )
    st.stop()

# ---------------------------------------------------------------------------
# 6. SIDEBAR
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    theme = st.toggle("Light mode", value=False)
    st.divider()

    st.markdown("## 📊 Dataset stats")
    st.metric("Total movies", f"{len(movies):,}")

    all_genres: list[str] = sorted(
        {g for sublist in movies["genres"] for g in sublist}
    )
    st.metric("Genres available", len(all_genres))

    if not TMDB_API_KEY:
        st.warning("TMDB_API_KEY not set.\nPosters won't load.\nAdd it to a `.env` file.")

    st.divider()
    st.caption("Built with TF-IDF + Cosine Similarity · TMDB API")

# ---------------------------------------------------------------------------
# 7. THEME TOKENS
# ---------------------------------------------------------------------------

if theme:
    bg          = "linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%)"
    text        = "#1e293b"
    card_bg     = "rgba(0,0,0,0.04)"
    title_grad  = "linear-gradient(90deg, #1e293b, #475569)"
    input_bg    = "rgba(0,0,0,0.04)"
    border      = "rgba(0,0,0,0.10)"
    hover_card  = "rgba(0,0,0,0.08)"
    score_color = "#E50914"
else:
    bg          = "radial-gradient(ellipse at top left, #1e293b 0%, #0f172a 50%, #000 100%)"
    text        = "#f1f5f9"
    card_bg     = "rgba(255,255,255,0.04)"
    title_grad  = "linear-gradient(90deg, #f8fafc, #94a3b8)"
    input_bg    = "rgba(255,255,255,0.06)"
    border      = "rgba(255,255,255,0.10)"
    hover_card  = "rgba(255,255,255,0.09)"
    score_color = "#ff4d57"

# ---------------------------------------------------------------------------
# 8. GLOBAL CSS
# ---------------------------------------------------------------------------

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&display=swap');

/* ---------- Layout ---------- */
.stApp {{
    background: {bg} !important;
    color: {text} !important;
    font-family: 'DM Sans', sans-serif;
}}

/* ---------- Headings ---------- */
h1 {{
    background: {title_grad};
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 52px !important;
    font-weight: 700 !important;
    letter-spacing: -1px;
    text-align: center;
    margin-bottom: 4px !important;
}}
.subtitle {{
    text-align: center;
    color: {text};
    opacity: 0.55;
    font-size: 15px;
    margin-top: 0;
    margin-bottom: 2rem;
}}

/* ---------- Selectbox ---------- */
.stSelectbox div[data-baseweb="select"] {{
    background-color: {input_bg} !important;
    border: 1px solid {border} !important;
    border-radius: 12px !important;
    color: {text} !important;
}}

/* ---------- Button ---------- */
div.stButton > button {{
    background: linear-gradient(90deg, #E50914, #9b060d);
    color: white !important;
    border-radius: 12px;
    height: 52px;
    width: 100%;
    font-size: 17px;
    font-weight: 600;
    border: none;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    letter-spacing: 0.2px;
}}
div.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(229,9,20,0.35);
}}
div.stButton > button:active {{
    transform: translateY(0);
}}

/* ---------- Poster cards ---------- */
.poster-card {{
    background: {card_bg};
    border: 1px solid {border};
    border-radius: 18px;
    padding: 14px;
    text-align: center;
    transition: transform 0.3s ease, background 0.3s ease;
    backdrop-filter: blur(6px);
    height: 100%;
}}
.poster-card:hover {{
    transform: translateY(-8px);
    background: {hover_card};
}}
.poster-card img {{
    width: 100%;
    border-radius: 12px;
    margin-bottom: 12px;
    display: block;
}}
.movie-title {{
    font-size: 14px;
    font-weight: 600;
    color: {text};
    line-height: 1.4;
    margin: 0;
}}
.similarity-badge {{
    display: inline-block;
    margin-top: 6px;
    padding: 2px 10px;
    border-radius: 99px;
    font-size: 11px;
    font-weight: 600;
    background: rgba(229,9,20,0.15);
    color: {score_color};
    border: 1px solid rgba(229,9,20,0.25);
}}

/* ---------- Divider ---------- */
hr {{ border-color: {border} !important; }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 9. HEADER
# ---------------------------------------------------------------------------

st.title("CineSync")
st.markdown('<p class="subtitle">Content-based recommendations powered by TF-IDF &amp; Cosine Similarity</p>', unsafe_allow_html=True)
st.divider()

# ---------------------------------------------------------------------------
# 10. CONTROLS
# ---------------------------------------------------------------------------

col_a, col_b, col_c = st.columns([3, 1, 1])

with col_a:
    selected_movie: str = st.selectbox(
        "Search for a movie",
        options=movies["title"].values,
        help="Type to search across all 5 000 movies",
    )

with col_b:
    selected_genre: str = st.selectbox(
        "Filter by genre",
        options=["All"] + all_genres,
        help="Narrow results to a specific genre",
    )

with col_c:
    top_n: int = st.selectbox(
        "Results",
        options=[3, 5, 8, 10],
        index=1,
        help="How many recommendations to show",
    )

st.markdown("<br>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([2, 2, 2])
with btn_col:
    show_recs = st.button("Show Recommendations 🎬", use_container_width=True)

# ---------------------------------------------------------------------------
# 11. RESULTS
# ---------------------------------------------------------------------------

if show_recs:
    with st.spinner("Finding your next favourite film …"):
        names, posters = recommend(
            movie=selected_movie,
            genre_filter=selected_genre,
            movies_df=movies,
            similarity_matrix=similarity,
            top_n=int(top_n),
        )

    if not names:
        st.info(
            f"No **{selected_genre}** movies found similar to **{selected_movie}**. "
            "Try changing the genre filter or pick a different movie."
        )
    else:
        st.markdown(f"#### Top {len(names)} picks similar to *{selected_movie}*")
        cols = st.columns(len(names))
        for i, col in enumerate(cols):
            with col:
                st.markdown(
                    f"""
                    <div class="poster-card">
                        <img src="{posters[i]}" alt="Poster for {names[i]}">
                        <p class="movie-title">{names[i]}</p>
                        <span class="similarity-badge">#{i + 1} match</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
