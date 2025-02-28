import streamlit as st
import pandas as pd
import numpy as np
import gdown
import re
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
import os

# Download the user score dataset from Google Drive
gdrive_url = "https://drive.google.com/uc?id=1eAZUQLfzxBtWqLr9qx845NRJTw2kM0Pn"
file_path = "users-score-2023.csv"

if not os.path.exists(file_path):
    gdown.download(gdrive_url, file_path, quiet=False)

# Load the Anime Dataset
anime_file_path = "anime-dataset-2023.csv"
anime_list = pd.read_csv(anime_file_path, 
                        low_memory=False, 
                        usecols=['anime_id', 'Name', 'Score', 'Genres', 'Type', 'Episodes', 'Aired', 
                                 'Producers', 'Licensors', 'Studios', 'Source', 'Synopsis', 'Rating', 'Popularity']
                        )

# Load user rating dataset
user_rec = pd.read_csv(file_path, usecols=['anime_id', 'user_id', 'rating'])

# Convert IDs to integers
user_rec['user_id'] = pd.to_numeric(user_rec['user_id'], errors='coerce')
user_rec['anime_id'] = pd.to_numeric(user_rec['anime_id'], errors='coerce')
user_rec.dropna(subset=['user_id', 'anime_id'], inplace=True)

# Keep only active users (rated at least 5 anime)
active_users = user_rec.groupby('user_id')['anime_id'].count()
user_rec = user_rec[user_rec['user_id'].isin(active_users[active_users >= 5].index)]

# Keep only popular anime (rated by at least 20 users)
popular_anime = user_rec.groupby('anime_id')['user_id'].count()
user_rec = user_rec[user_rec['anime_id'].isin(popular_anime[popular_anime >= 20].index)]

# Convert user and anime IDs to category codes (for efficient memory usage)
user_rec['user_code'] = user_rec['user_id'].astype('category').cat.codes
user_rec['anime_code'] = user_rec['anime_id'].astype('category').cat.codes

# Create sparse matrix
sparse_matrix = csr_matrix(
    (user_rec['rating'], (user_rec['user_code'], user_rec['anime_code']))
)

# Compute cosine similarity using sparse matrix multiplication
anime_similarity_cosine = cosine_similarity(sparse_matrix.T, dense_output=False)

# Convert to DataFrame with anime IDs as index and columns
anime_similarity_df = pd.DataFrame(
    anime_similarity_cosine.toarray(),
    index=user_rec['anime_id'].astype('category').cat.categories,
    columns=user_rec['anime_id'].astype('category').cat.categories
)

# Function to get recommendations
def get_recommendations_by_name(anime_name, suggest_amount=10):
    best_match = process.extractOne(anime_name, anime_list['Name'])
    
    if best_match is None:
        return f"No anime found with name similar to '{anime_name}'"
    
    anime_title = best_match[0]
    anime_id = anime_list.loc[anime_list['Name'] == anime_title, 'anime_id'].values[0]
    
    if anime_id not in anime_similarity_df.columns:
        return f"Anime '{anime_title}' not found in similarity matrix."
    
    # Get similar anime
    sim_scores = anime_similarity_df[anime_id].sort_values(ascending=False)[1:suggest_amount+1]
    
    # Get recommended anime details
    recommended_anime = anime_list[anime_list['anime_id'].isin(sim_scores.index)][['Name', 'Score', 'Genres']]
    
    # Sort by score (descending)
    recommended_anime = recommended_anime.sort_values(by='Score', ascending=False)
    
    return anime_title, recommended_anime

# Streamlit App
st.title("Anime Recommendation System")

anime_name_input = st.text_input("Enter an anime name:")

if st.button("Get Recommendations"):
    if anime_name_input:
        anime_title, recommendations = get_recommendations_by_name(anime_name_input)
        st.subheader(f"Recommendations for: {anime_title}")
        st.dataframe(recommendations)
    else:
        st.warning("Please enter an anime name.")