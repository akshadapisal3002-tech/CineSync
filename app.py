import streamlit as st
import pickle
import pandas as pd
import requests

# ==============================
# 1. API & RECOMMENDATION LOGIC
# ==============================

def fetch_poster(movie_id):
    """Fetches movie poster from TMDB API with error handling."""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
        return "https://via.placeholder.com/500x750?text=No+Poster"
    except Exception:
        return "https://via.placeholder.com/500x750?text=Error+Loading"

def recommend(movie, genre_filter, movies_df, similarity_matrix):
    """Calculates top 5 recommendations based on TF-IDF similarity and genre."""
    try:
        movie_index = movies_df[movies_df['title'] == movie].index[0]
        distances = similarity_matrix[movie_index]

        # Get top 50 potential matches to allow more room for genre filtering
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:51]

        recommended_movies = []
        recommended_posters = []

        for i in movies_list:
            movie_row = movies_df.iloc[i[0]]
            
            # Apply Genre Filter (Handles list of genres in data)
            if genre_filter != "All":
                if genre_filter not in movie_row['genres']:
                    continue

            recommended_movies.append(movie_row.title)
            recommended_posters.append(fetch_poster(movie_row.movie_id))

            if len(recommended_movies) == 5:
                break
        
        return recommended_movies, recommended_posters
    except Exception:
        return [], []

# ==============================
# 2. DATA LOADING (WITH CACHING)
# ==============================

@st.cache_data
def load_data():
    """Loads and caches data to prevent reloading on every interaction."""
    try:
        movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
        movies_df = pd.DataFrame(movies_dict)
        similarity_matrix = pickle.load(open('similarity.pkl', 'rb'))
        return movies_df, similarity_matrix
    except FileNotFoundError:
        return None, None

movies, similarity = load_data()

if movies is None:
    st.error("Model files not found. Please ensure 'movie_dict.pkl' and 'similarity.pkl' are in the directory.")
    st.stop()

# ==============================
# 3. UI CONFIGURATION & THEME
# ==============================

st.set_page_config(page_title="CineSync", layout="wide")

# Sidebar Theme Toggle
with st.sidebar:
    st.header("Toggle")
    theme = st.toggle("Switch to Light Mode", value=False)

# Dynamic Variable Assignment
if theme:
    bg_gradient = "linear-gradient(to right, #f8fafc, #e2e8f0)"
    text_color = "#1e293b"
    card_bg = "rgba(0, 0, 0, 0.05)"
    title_gradient = "linear-gradient(90deg, #1e293b, #64748b)"
    input_bg = "rgba(0, 0, 0, 0.05)"
    border_color = "rgba(0, 0, 0, 0.1)"
else:
    bg_gradient = "radial-gradient(circle at top left, #1e293b, #0f172a, #000000)"
    text_color = "#f8fafc"
    card_bg = "rgba(255, 255, 255, 0.03)"
    title_gradient = "linear-gradient(90deg, #ffffff, #94a3b8)"
    input_bg = "rgba(255, 255, 255, 0.05)"
    border_color = "rgba(255, 255, 255, 0.1)"

st.markdown(f"""
<style>
    .stApp {{ background: {bg_gradient} !important; color: {text_color} !important; font-family: 'Inter', sans-serif; }}
    h1 {{ background: {title_gradient}; -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 56px !important; font-weight: 800 !important; text-align: center; }}
    .stSelectbox div[data-baseweb="select"] {{ background-color: {input_bg} !important; border: 1px solid {border_color} !important; border-radius: 12px !important; color: {text_color} !important; }}
    div.stButton > button {{ background: linear-gradient(90deg, #E50914, #9b060d); color: white !important; border-radius: 12px; height: 55px; width: 100%; font-size: 20px; font-weight: 700; border: none; transition: all 0.4s ease; }}
    div.stButton > button:hover {{ transform: translateY(-3px); background: linear-gradient(90deg, #ff0f1a, #E50914); }}
    .poster-card {{ background: {card_bg}; border: 1px solid {border_color}; border-radius: 20px; padding: 15px; text-align: center; transition: all 0.4s ease; backdrop-filter: blur(5px); height: 100%; }}
    .poster-card:hover {{ transform: translateY(-10px); background: rgba(128, 128, 128, 0.1); }}
    .poster-card img {{ border-radius: 15px; margin-bottom: 15px; }}
    .movie-title {{ font-size: 16px; font-weight: 700; color: {text_color}; }}
</style>
""", unsafe_allow_html=True)

st.title('Movie Recommendation System;')

# User Inputs
col_a, col_b = st.columns([2, 1])
with col_a:
    selected_movie = st.selectbox('Search for a movie:', movies['title'].values)
with col_b:
    # Extract unique genres for filter
    all_genres = sorted(list(set([g for sublist in movies['genres'] for g in sublist])))
    selected_genre = st.selectbox("Filter by Genre:", ["All"] + all_genres)

# ==============================
# ==============================
# 4. SHOW RECOMMENDATIONS
# ==============================

if st.button('Show Me Recommendations 🎬'):
    # Fetch data from our helper function
    names, posters = recommend(selected_movie, selected_genre, movies, similarity)

    if not names:
        st.info(f"Hmm, we couldn't find any {selected_genre} movies similar to that. Try changing the filter!")
    else:
        # Instead of hard-coding 5, we create as many columns as we have results
        cols = st.columns(len(names))
        
        for i, col in enumerate(cols):
            with col:
                # Using a cleaner f-string approach for the HTML
                st.markdown(
                    f"""
                    <div class="poster-container">
                        <img src="{posters[i]}" class="movie-poster">
                        <p class="movie-label">{names[i]}</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )