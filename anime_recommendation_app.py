import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
import os
import gdown
from config import USERS_SCORE_FILE_URL, USERS_SCORE_FILE_PATH

# Debugging: Check if the file exists
if not os.path.exists(USERS_SCORE_FILE_PATH):
    st.write("Downloading file...")
    try:
        gdown.download(USERS_SCORE_FILE_URL, USERS_SCORE_FILE_PATH, quiet=False)
        st.success("File downloaded successfully!")
    except Exception as e:
        st.error(f"Failed to download the file: {e}")
        st.stop()
else:
    st.write("File already exists. Skipping download.")

st.title("Anime Recommendation System")

# Cache the anime dataset loading
@st.cache_data
def load_anime_data():
    st.write("Loading anime data...")
    file_path = "anime-dataset-2023.csv"
    return pd.read_csv(
        file_path, 
        low_memory=False, 
        usecols=[
            'anime_id', 'Name', 'Score', 'Genres', 'Type', 'Episodes', 'Aired', 
            'Producers', 'Licensors', 'Studios', 'Source', 'Synopsis', 'Rating', 'Popularity'
        ]
    )

# Cache the user ratings dataset loading and preprocessing
@st.cache_data
def load_user_data():
    st.write("Loading user data...")
    user_rec = pd.read_csv(USERS_SCORE_FILE_PATH, usecols=['anime_id', 'user_id', 'rating'])
    
    # Convert IDs to numeric and drop invalid rows
    user_rec['user_id'] = pd.to_numeric(user_rec['user_id'], errors='coerce')
    user_rec['anime_id'] = pd.to_numeric(user_rec['anime_id'], errors='coerce')
    user_rec.dropna(subset=['user_id', 'anime_id'], inplace=True)
    
    # Keep only active users (rated at least 5 anime)
    active_users = user_rec.groupby('user_id')['anime_id'].count()
    user_rec = user_rec[user_rec['user_id'].isin(active_users[active_users >= 5].index)]
    
    # Keep only popular anime (rated by at least 20 users)
    popular_anime = user_rec.groupby('anime_id')['user_id'].count()
    user_rec = user_rec[user_rec['anime_id'].isin(popular_anime[popular_anime >= 20].index)]
    
    st.write("User data loaded successfully!")
    return user_rec

# Cache the similarity matrix computation
@st.cache_resource
def build_similarity_matrix(user_rec):
    st.write("Building similarity matrix...")
    # Convert user and anime IDs to category codes for efficient memory usage
    user_rec['user_code'] = user_rec['user_id'].astype('category').cat.codes
    user_rec['anime_code'] = user_rec['anime_id'].astype('category').cat.codes
    
    # Create the sparse matrix
    sparse_matrix = csr_matrix(
        (user_rec['rating'], (user_rec['user_code'], user_rec['anime_code']))
    )
    
    # Compute cosine similarity between anime (transpose the matrix)
    anime_similarity_cosine = cosine_similarity(sparse_matrix.T, dense_output=False)
    
    # Build a DataFrame with anime IDs as both the index and columns
    similarity_df = pd.DataFrame(
        anime_similarity_cosine.toarray(),
        index=user_rec['anime_id'].astype('category').cat.categories,
        columns=user_rec['anime_id'].astype('category').cat.categories
    )
    st.write("Similarity matrix built successfully!")
    return similarity_df

# Load the datasets and build the similarity matrix
anime_list = load_anime_data()
user_rec = load_user_data()
anime_similarity_df = build_similarity_matrix(user_rec)

def get_recommendations_by_name(anime_name, suggest_amount=10):
    # Use fuzzy matching to find the best match in the anime list
    best_match = process.extractOne(anime_name, anime_list['Name'])
    if best_match is None:
        return f"No anime found with name similar to '{anime_name}'"
    
    anime_title = best_match[0]
    anime_id = anime_list.loc[anime_list['Name'] == anime_title, 'anime_id'].values[0]
    
    if anime_id not in anime_similarity_df.columns:
        return f"Anime '{anime_title}' not found in similarity matrix."
    
    # Retrieve similarity scores and select the top suggestions (skip self-match)
    sim_scores = anime_similarity_df[anime_id].sort_values(ascending=False)[1:suggest_amount+1]
    
    # Get details for the recommended anime
    recommended_anime = anime_list[anime_list['anime_id'].isin(sim_scores.index)][['Name', 'Score', 'Genres']]
    # Sort recommendations by score (descending)
    recommended_anime = recommended_anime.sort_values(by='Score', ascending=False)
    
    return anime_title, recommended_anime

# Streamlit UI
anime_input = st.text_input("Enter an anime name to get recommendations:", "Naruto")
if st.button("Get Recommendations"):
    result = get_recommendations_by_name(anime_input)
    if isinstance(result, str):
        st.error(result)
    else:
        anime_title, recommendations = result
        st.subheader(f"Recommendations for {anime_title}:")
        st.dataframe(recommendations)
