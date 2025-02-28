import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

# Load Anime Dataset
anime_file_path = "anime-dataset-2023.csv"

try:
    anime_list = pd.read_csv(anime_file_path, low_memory=False, 
                            usecols=['anime_id', 'Name', 'Score', 'Genres', 'Type', 'Episodes', 'Aired', 
                                     'Producers', 'Licensors', 'Studios', 'Source', 'Synopsis', 'Rating', 'Popularity'])
    st.write("✅ Anime dataset loaded successfully!")
except Exception as e:
    st.error(f"⚠️ Failed to load the anime dataset: {e}")
    st.stop()

# Load User Rating Dataset
file_url1 = "https://raw.githubusercontent.com/Munhboldn/Anime/main/users-score-part1.parquet"
file_url2 = "https://raw.githubusercontent.com/Munhboldn/Anime/main/users-score-part2.parquet"

try:
    user_rec1 = pd.read_parquet(file_url1)
    user_rec2 = pd.read_parquet(file_url2)
    user_rec = pd.concat([user_rec1, user_rec2], ignore_index=True)
    st.write("✅ User rating dataset loaded successfully!")
except Exception as e:
    st.error(f"⚠️ Failed to load the user rating dataset: {e}")
    st.stop()

# Filter Active Users & Popular Anime
try:
    active_users = user_rec.groupby('user_id')['anime_id'].count()
    user_rec = user_rec[user_rec['user_id'].isin(active_users[active_users >= 10].index)]

    popular_anime = user_rec['anime_id'].value_counts().head(10000).index
    user_rec = user_rec[user_rec['anime_id'].isin(popular_anime)]

    # Convert user and anime IDs to category codes
    user_rec['user_code'] = user_rec['user_id'].astype('category').cat.codes
    user_rec['anime_code'] = user_rec['anime_id'].astype('category').cat.codes

    # Create Sparse Matrix & Compute Similarity
    sparse_matrix = csr_matrix((user_rec['rating'], (user_rec['user_code'], user_rec['anime_code'])))
    anime_similarity_df = pd.DataFrame(
        cosine_similarity(sparse_matrix.T, dense_output=False).toarray(),
        index=user_rec['anime_id'].astype('category').cat.categories,
        columns=user_rec['anime_id'].astype('category').cat.categories
    )
    st.write("✅ Anime similarity matrix computed successfully!")
except Exception as e:
    st.error(f"⚠️ Error processing user ratings: {e}")
    st.stop()

# Function to get recommendations
def get_recommendations_by_name(anime_name, suggest_amount=10):
    try:
        best_match = process.extractOne(anime_name, anime_list['Name'])
        
        if not best_match or best_match[1] < 60:
            return f"⚠️ No anime found similar to '{anime_name}'"
        
        anime_title = best_match[0]
        anime_id = anime_list.loc[anime_list['Name'] == anime_title, 'anime_id'].values[0]

        if anime_id not in anime_similarity_df.index:
            return f"⚠️ Anime '{anime_title}' is not available for recommendations."
        
        # Get similar anime, ensuring we have enough results
        sim_scores = anime_similarity_df.loc[anime_id].sort_values(ascending=False)
        sim_scores = sim_scores.iloc[1:min(len(sim_scores), suggest_amount+1)]

        recommended_anime = anime_list[anime_list['anime_id'].isin(sim_scores.index)][['Name', 'Score', 'Genres']]
        recommended_anime = recommended_anime.sort_values(by='Score', ascending=False)

        return anime_title, recommended_anime
    except Exception as e:
        return f"⚠️ Error generating recommendations: {e}"

# Streamlit App
def main():
    st.set_page_config(page_title="Anime Recommender", layout="wide")
    
    st.markdown("<h1 style='text-align: center;'>🎬 Anime Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Find similar anime based on your favorite one!</p>", unsafe_allow_html=True)
    
    anime_name_input = st.text_input("Enter an anime name:", placeholder="e.g., Naruto")

    if st.button("Get Recommendations", use_container_width=True):
        if not anime_name_input:
            st.warning("⚠️ Please enter an anime name.")
            return
        
        with st.spinner("🔍 Searching for recommendations..."):
            result = get_recommendations_by_name(anime_name_input)

        if isinstance(result, tuple):
            anime_title, recommendations = result
            st.subheader(f"🎯 Recommendations for: **{anime_title}**")
            st.dataframe(recommendations)
        else:
            st.error(result)

# Run the app
if __name__ == "__main__":
    main()
