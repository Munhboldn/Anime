import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

# Cache data loading and preprocessing to improve performance
@st.cache_data
def load_data():
    # Load the Anime Dataset
    anime_file_path = "anime-dataset-2023.csv"
    anime_list = pd.read_csv(
        anime_file_path,
        low_memory=False,
        usecols=['anime_id', 'Name', 'Score', 'Genres', 'Type', 'Episodes', 'Aired', 
                 'Producers', 'Licensors', 'Studios', 'Source', 'Synopsis', 'Rating', 'Popularity']
    )

    # Load user rating dataset (split files from GitHub)
    file_url1 = "https://raw.githubusercontent.com/Munhboldn/Anime/main/users-score-part1.parquet"
    file_url2 = "https://raw.githubusercontent.com/Munhboldn/Anime/main/users-score-part2.parquet"

    # Read both files
    user_rec1 = pd.read_parquet(file_url1)
    user_rec2 = pd.read_parquet(file_url2)

    # Merge them into a single dataset
    user_rec = pd.concat([user_rec1, user_rec2], ignore_index=True)

    # Keep only active users (rated at least 10 anime)
    active_users = user_rec.groupby('user_id')['anime_id'].count()
    user_rec = user_rec[user_rec['user_id'].isin(active_users[active_users >= 10].index)]

    # Keep only the top 10,000 most-rated anime
    popular_anime = user_rec['anime_id'].value_counts().head(10000).index
    user_rec = user_rec[user_rec['anime_id'].isin(popular_anime)]

    # Convert user and anime IDs to category codes (for efficient memory usage)
    user_rec['user_code'] = user_rec['user_id'].astype('category').cat.codes
    user_rec['anime_code'] = user_rec['anime_id'].astype('category').cat.codes

    # Create sparse matrix
    sparse_matrix = csr_matrix(
        (user_rec['rating'], (user_rec['user_code'], user_rec['anime_code']))
    )

    # Compute cosine similarity using sparse matrix multiplication
    anime_similarity_cosine = cosine_similarity(sparse_matrix.T, dense_output=False)

    return anime_list, user_rec, anime_similarity_cosine

# Function to get recommendations
def get_recommendations_by_name(anime_name, suggest_amount=10):
    try:
        st.write(f"🔍 Searching for: {anime_name}")

        if anime_list.empty:
            st.error("⚠️ Error: Anime dataset is empty!")
            return "Anime dataset is empty."

        best_match = process.extractOne(anime_name, anime_list['Name'])

        if best_match is None or best_match[1] < 60:
            st.error(f"⚠️ No anime found similar to '{anime_name}'")
            return f"No anime found with a name similar to '{anime_name}'"

        anime_title = best_match[0]
        st.write(f"✅ Best match found: {anime_title}")

        anime_id = anime_list.loc[anime_list['Name'] == anime_title, 'anime_id'].values
        if len(anime_id) == 0:
            st.error(f"⚠️ Anime '{anime_title}' not found in dataset.")
            return f"Anime '{anime_title}' not found in dataset."

        anime_id = anime_id[0]
        st.write(f"🆔 Anime ID: {anime_id}")

        if anime_similarity_df.empty:
            st.error("⚠️ Error: Anime similarity matrix is empty!")
            return "Anime similarity matrix is empty."

        if anime_id not in anime_similarity_df.index:
            st.error(f"⚠️ Anime '{anime_title}' does not exist in the similarity matrix.")
            return f"Anime '{anime_title}' does not exist in the similarity matrix."

        # Debugging: Check data types before computing similarity
        st.write(f"📊 Checking data types: anime_id={type(anime_id)}, similarity matrix index={anime_similarity_df.index.dtype}")

        # Convert anime_id to match index type
        anime_id = str(anime_id) if anime_similarity_df.index.dtype == 'object' else int(anime_id)

        # Get similar anime
        sim_scores = anime_similarity_df.loc[anime_id].sort_values(ascending=False)[1:suggest_amount+1]
        st.write("📊 Similarity scores calculated!")

        recommended_anime = anime_list[anime_list['anime_id'].isin(sim_scores.index)][['Name', 'Score', 'Genres']]
        recommended_anime = recommended_anime.sort_values(by='Score', ascending=False)

        return anime_title, recommended_anime

    except Exception as e:
        st.error(f"⚠️ Full Error: {repr(e)}")
        return f"Error: {repr(e)}"


# Streamlit App
def main():
    st.title("Anime Recommendation System")
    
    # Load data
    anime_list, user_rec, anime_similarity_cosine = load_data()
    
    # User input
    anime_name_input = st.text_input("Enter an anime name:")
    suggest_amount = st.slider("Number of recommendations", min_value=1, max_value=20, value=10)
    
    if st.button("Get Recommendations"):
        if anime_name_input:
            with st.spinner("Fetching recommendations..."):
                result = get_recommendations_by_name(
                    anime_name_input, anime_list, user_rec, anime_similarity_cosine, suggest_amount
                )
            
            if result[1] is not None:
                anime_title, recommendations = result
                st.subheader(f"Recommendations for: {anime_title}")
                st.dataframe(recommendations)
            else:
                st.warning(result[0])
        else:
            st.warning("Please enter an anime name.")

if __name__ == "__main__":
    main()
