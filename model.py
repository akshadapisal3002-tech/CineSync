
import pandas as pd
import ast
import pickle
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. LOAD DATA
movies = pd.read_csv("data/tmdb_5000_movies.csv")
credits = pd.read_csv("data/tmdb_5000_credits.csv")
df = movies.merge(credits, on="title")

# 2. CLEANING TOOLS
# Instead of 3 different functions, one flexible helper handles the JSON strings
def parse_json(obj, key='name', limit=None):
    data = ast.literal_eval(obj)
    items = [i[key] for i in data]
    return items[:limit] if limit else items

def get_director(obj):
    data = ast.literal_eval(obj)
    return [i['name'] for i in data if i['job'] == 'Director']

# 3. APPLYING LOGIC
# Keep only what we need
df = df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']].dropna()

df['genres']   = df['genres'].apply(parse_json)
df['keywords'] = df['keywords'].apply(parse_json)
df['cast']     = df['cast'].apply(lambda x: parse_json(x, limit=3))
df['crew']     = df['crew'].apply(get_director)
df['overview'] = df['overview'].apply(lambda x: x.split())

# Remove spaces to make "Johnny Depp" -> "JohnnyDepp" (crucial for tags)
for col in ['genres', 'keywords', 'cast', 'crew']:
    df[col] = df[col].apply(lambda x: [i.replace(" ", "") for i in x])

# 4. CREATING THE "TAGS"
# We give the director (crew) extra weight by repeating them
df['tags'] = df['overview'] + df['genres'] + df['keywords'] + df['cast'] + (df['crew'] * 3)

# Final skinny dataframe for the UI
new_df = df[['movie_id', 'title', 'tags', 'genres']].copy()
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

# 5. ML PROCESSING (Stemming & Vectorization)
ps = PorterStemmer()
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join([ps.stem(word) for word in x.split()]))

# TF-IDF gives better importance to unique words than simple counting
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
vectors = tfidf.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vectors)

# 6. EXPORTING
pickle.dump(new_df.to_dict(), open('movie_dict.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

print("🚀 Model training complete! Files saved for Streamlit.")