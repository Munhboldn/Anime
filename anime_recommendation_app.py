import streamlit as st
import pandas as pd
import numpy as np
import re
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
import os

# Load Anime Dataset
anime_file_path = "anime-dataset-2023.csv"
anime_list = pd.read_csv(anime_file_path, low_memory=False, usecols=['anime_id', 'Name', 'Score', 'Genres'])

# Load User Rating Dataset
file_url1 = "https://raw.githubusercontent.com/Munhboldn/Anime/main/users-score-part1.parquet"
file_url2 = "https://raw.githubusercontent.com/Munhboldn/Anime/main/users-score-part2.parquet"
user_rec = pd.concat([pd.read_parquet(file_url1), pd.read_parquet(file_url2)], ignore_index=True)

# Filter Active Users & Popular Anime
active_users = user_rec.groupby('user_id')['anime_id'].count()
user_rec = user_rec[user_rec['user_id'].isin(active_users[active_users >= 10].index)]

popular_anime = user_rec['anime_id'].value_counts().head(10000).index
user_rec = user_rec[user_rec['anime_id'].isin(popular_anime)]

# Encode User and Anime IDs
user_rec['user_code'] = user_rec['user_id'].astype('category').cat.codes
user_rec['anime_code'] = user_rec['anime_id'].astype('category').cat.codes

# Create Sparse Matrix & Compute Similarity
sparse_matrix = csr_matrix((user_rec['rating'], (user_rec['user_code'], user_rec['anime_code'])))
anime_similarity_df = pd.DataFrame(cosine_similarity(sparse_matrix.T, dense_output=False).toarray(),
                                   index=user_rec['anime_id'].astype('category').cat.categories,
                                   columns=user_rec['anime_id'].astype('category').cat.categories)

def get_recommendations_by_name(anime_name, suggest_amount=10):
    best_match = process.extractOne(anime_name, anime_list['Name'])
    if not best_match or best_match[1] < 60:
        return "No similar anime found."
    
    anime_title = best_match[0]
    anime_id = anime_list.loc[anime_list['Name'] == anime_title, 'anime_id'].values[0]
    sim_scores = anime_similarity_df.loc[anime_id].sort_values(ascending=False)[1:suggest_amount+1]
    recommended_anime = anime_list[anime_list['anime_id'].isin(sim_scores.index)][['Name', 'Score', 'Genres']]
    return anime_title, recommended_anime.sort_values(by='Score', ascending=False)

def main():
    st.set_page_config(page_title="Anime Recommender", layout="wide")
    st.title("🎬 Anime Recommendation System")
    st.markdown("Find anime similar to your favorite one!")
    anime_name_input = st.text_input("Enter an anime name:", placeholder="e.g., Naruto")
    
    if st.button("Get Recommendations", use_container_width=True):
        if anime_name_input:
            result = get_recommendations_by_name(anime_name_input)  # FIXED FUNCTION CALL
            if isinstance(result, tuple):
                anime_title, recommendations = result
                st.subheader(f"Recommended Anime for: {anime_title}")
                st.dataframe(recommendations)
            else:
                st.warning(result)
        else:
            st.warning("Please enter an anime name.")

if __name__ == "__main__":
    main()
