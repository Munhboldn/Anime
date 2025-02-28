import streamlit as st
import pandas as pd
import numpy as np
import re
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
import os
import time

# Set page configuration
st.set_page_config(
    page_title="Anime Recommender",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS to improve appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #4B4BFF;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .stProgress > div > div > div {
        background-color: #FF4B4B;
    }
    .recommendation-box {
        background-color: #f0f0f5;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Display app title
st.markdown("<h1 class='main-header'>📺 Anime Recommendation System</h1>", unsafe_allow_html=True)

# Add a sidebar for additional features
with st.sidebar:
    st.header("About")
    st.write("This application recommends anime based on collaborative filtering and content similarity.")
    
    st.divider()
    
    st.header("Settings")
    suggest_amount = st.slider("Number of recommendations", min_value=5, max_value=20, value=10, step=5)
    min_score = st.slider("Minimum rating score", min_value=1.0, max_value=9.0, value=7.0, step=0.5)

# Function to load data with caching
@st.cache_data(ttl=3600, show_spinner=False)
def load_anime_data():
    try:
        anime_file_path = "anime-dataset-2023.csv"
        return pd.read_csv(anime_file_path, 
                          low_memory=False, 
                          usecols=['anime_id', 'Name', 'Score', 'Genres', 'Type', 'Episodes', 'Aired', 
                                   'Producers', 'Licensors', 'Studios', 'Source', 'Synopsis', 'Rating', 'Popularity']
                          )
    except Exception as e:
        st.error(f"⚠️ Failed to load anime dataset: {e}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_user_data():
    try:
        file_url1 = "https://raw.githubusercontent.com/Munhboldn/Anime/main/users-score-part1.parquet"
        file_url2 = "https://raw.githubusercontent.com/Munhboldn/Anime/main/users-score-part2.parquet"
        
        user_rec1 = pd.read_parquet(file_url1)
        user_rec2 = pd.read_parquet(file_url2)
        return pd.concat([user_rec1, user_rec2], ignore_index=True)
    except Exception as e:
        st.error(f"⚠️ Failed to load user dataset: {e}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def prepare_similarity_matrix(user_rec):
    if user_rec is None:
        return None
        
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

    # Convert to DataFrame with anime IDs as index and columns
    return pd.DataFrame(
        anime_similarity_cosine.toarray(),
        index=user_rec['anime_id'].astype('category').cat.categories,
        columns=user_rec['anime_id'].astype('category').cat.categories
    )

# Load data with progress indicator
with st.spinner("Loading anime database..."):
    anime_list = load_anime_data()
    if anime_list is not None:
        st.success("✅ Anime dataset loaded successfully!")

with st.spinner("Loading user ratings..."):
    user_rec = load_user_data()
    if user_rec is not None:
        st.success("✅ User rating dataset loaded successfully!")

with st.spinner("Building recommendation engine..."):
    anime_similarity_df = prepare_similarity_matrix(user_rec)
    if anime_similarity_df is not None:
        st.success("✅ Recommendation engine ready!")

# Function to get recommendations
def get_recommendations_by_name(anime_name, suggest_amount=10, min_score=7.0):
    try:
        if anime_list is None or anime_list.empty:
            return "⚠️ Anime dataset is missing or empty!"

        # Find the best match for the input anime name
        best_match = process.extractOne(anime_name, anime_list['Name'])
        if best_match is None or best_match[1] < 60:
            return f"No anime found with a name similar to '{anime_name}'"

        anime_title = best_match[0]
        match_confidence = best_match[1]
        anime_id = anime_list.loc[anime_list['Name'] == anime_title, 'anime_id'].values

        if len(anime_id) == 0:
            return f"Anime '{anime_title}' not found in dataset."

        anime_id = anime_id[0]
        
        # Get details of the selected anime
        selected_anime_details = anime_list[anime_list['anime_id'] == anime_id].iloc[0]

        if anime_similarity_df is None or anime_similarity_df.empty:
            return "⚠️ Anime similarity matrix is empty!"

        if anime_id not in anime_similarity_df.index:
            return f"Anime '{anime_title}' does not exist in the similarity matrix."

        # Get similar anime
        sim_scores = anime_similarity_df.loc[anime_id].sort_values(ascending=False)[1:suggest_amount*2+1]
        recommended_anime = anime_list[anime_list['anime_id'].isin(sim_scores.index)]
        
        # Filter by minimum score
        recommended_anime = recommended_anime[recommended_anime['Score'] >= min_score]
        
        # Sort by similarity score and then by rating
        recommended_anime = recommended_anime.sort_values(by=['Score'], ascending=False).head(suggest_amount)
        
        # Get final result with selected columns
        final_recommendations = recommended_anime[['Name', 'Score', 'Genres', 'Type', 'Episodes']]

        return anime_title, match_confidence, selected_anime_details, final_recommendations

    except Exception as e:
        st.error(f"⚠️ Recommendation Error: {repr(e)}")
        return f"Error: {repr(e)}"

# Main app interface
col1, col2 = st.columns([2, 3])

with col1:
    anime_name_input = st.text_input("Enter an anime name:", placeholder="e.g. Naruto, Attack on Titan, Death Note")
    
    # Add some example buttons for popular anime
    st.write("Or try one of these popular anime:")
    example_col1, example_col2, example_col3 = st.columns(3)
    
    with example_col1:
        if st.button("Death Note"):
            anime_name_input = "Death Note"
    
    with example_col2:
        if st.button("One Piece"):
            anime_name_input = "One Piece"
            
    with example_col3:
        if st.button("Attack on Titan"):
            anime_name_input = "Attack on Titan"

with col2:
    search_button = st.button("🔍 Get Recommendations", type="primary", use_container_width=True)

# Process search request
if search_button or (anime_name_input and 'last_search' not in st.session_state):
    try:
        if not anime_name_input:
            st.warning("⚠️ Please enter an anime name.")
        else:
            st.session_state.last_search = anime_name_input
            
            with st.spinner(f"🔍 Finding recommendations for: {anime_name_input}"):
                # Add slight delay to show the spinner
                time.sleep(0.5)
                result = get_recommendations_by_name(anime_name_input, suggest_amount, min_score)

            if isinstance(result, tuple):
                anime_title, match_confidence, selected_anime, recommendations = result
                
                # Display information about the selected anime
                st.markdown(f"<h2 class='subheader'>Selected Anime: {anime_title}</h2>", unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.metric("Match Confidence", f"{match_confidence}%")
                    st.metric("Rating", f"{selected_anime['Score']}/10")
                    st.metric("Episodes", selected_anime['Episodes'])
                
                with col2:
                    st.write(f"**Type:** {selected_anime['Type']}")
                    st.write(f"**Genres:** {selected_anime['Genres']}")
                    if not pd.isna(selected_anime['Synopsis']):
                        synopsis = selected_anime['Synopsis']
                        # Truncate long synopsis
                        if len(synopsis) > 300:
                            synopsis = synopsis[:300] + "..."
                        st.write(f"**Synopsis:** {synopsis}")
                
                # Display recommendations
                st.markdown(f"<h2 class='subheader'>Top {len(recommendations)} Recommendations</h2>", unsafe_allow_html=True)
                
                # Convert genres to lists for better display
                recommendations['Genres'] = recommendations['Genres'].apply(lambda x: x.split(', ') if isinstance(x, str) else x)
                
                # Display each recommendation in a card-like format
                for i, (_, anime) in enumerate(recommendations.iterrows()):
                    with st.container():
                        st.markdown(f"""
                        <div class='recommendation-box'>
                            <h3>{i+1}. {anime['Name']}</h3>
                            <p><strong>Score:</strong> {anime['Score']}/10 | <strong>Type:</strong> {anime['Type']} | <strong>Episodes:</strong> {anime['Episodes']}</p>
                            <p><strong>Genres:</strong> {', '.join(anime['Genres']) if isinstance(anime['Genres'], list) else anime['Genres']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.error(result)
    
    except Exception as e:
        st.error(f"⚠️ Application Error: {repr(e)}")

# Add footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8rem;'>
    Anime Recommendation System | Built with Streamlit | Data source: MyAnimeList
</div>
""", unsafe_allow_html=True)
