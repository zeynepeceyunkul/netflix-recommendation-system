"""
Streamlit App for Netflix-like Recommendation System
Redesigned with Netflix-style UI and explainable recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import load_all_data
from preprocessing import preprocess_movies, preprocess_ratings, get_user_features
from content_based import ContentBasedRecommender
from collaborative import CollaborativeFilteringRecommender
from kmeans_model import UserSegmentation
from popularity import PopularityRecommender

# Page configuration
st.set_page_config(
    page_title="Netflix Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Netflix-like styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #E50914;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #808080;
        margin-bottom: 2rem;
    }
    .movie-card {
        background-color: #141414;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem;
        border: 1px solid #333;
        transition: transform 0.2s;
    }
    .movie-card:hover {
        transform: scale(1.02);
        border-color: #E50914;
    }
    .movie-title {
        font-size: 1.1rem;
        font-weight: bold;
        color: #FFFFFF;
        margin-bottom: 0.5rem;
    }
    .movie-genres {
        color: #B3B3B3;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .explanation-box {
        background-color: #1F1F1F;
        border-left: 3px solid #E50914;
        padding: 0.75rem;
        margin-top: 0.5rem;
        border-radius: 4px;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #FFFFFF;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E50914;
    }
    .stMetric {
        background-color: #1F1F1F;
        padding: 1rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🎬 Netflix-like Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by Machine Learning • MovieLens Dataset</div>', unsafe_allow_html=True)

# Data disclaimer
with st.expander("ℹ️ About Data Source", expanded=False):
    st.info("""
    **Important**: This system uses the **MovieLens open-source dataset**, not Netflix's proprietary data.
    Netflix data is not publicly available. This project simulates Netflix-style recommendation logic
    using publicly available research data.
    """)

# Load data
@st.cache_data
def load_data():
    """Load and preprocess all data."""
    movies_df, ratings_df, users_df = load_all_data()
    ratings_df = preprocess_ratings(ratings_df, min_ratings_per_user=5)
    movies_df = preprocess_movies(movies_df, ratings_df)
    return movies_df, ratings_df, users_df

@st.cache_resource
def load_models(movies_df, ratings_df):
    """Load and cache ML models."""
    content_model = ContentBasedRecommender(movies_df, similarity_threshold=0.1)
    collaborative_model = CollaborativeFilteringRecommender(ratings_df)
    popularity_model = PopularityRecommender(ratings_df, movies_df)
    user_features = get_user_features(ratings_df, movies_df)
    segmentation_model = UserSegmentation(user_features, n_clusters=5)
    return content_model, collaborative_model, popularity_model, segmentation_model

# Load data
with st.spinner("Loading data and training models... This may take a moment."):
    movies_df, ratings_df, users_df = load_data()
    content_model, collaborative_model, popularity_model, segmentation_model = load_models(movies_df, ratings_df)

# Sidebar
st.sidebar.header("⚙️ Settings")

# Educational mode toggle
educational_mode = st.sidebar.checkbox(
    "📚 Educational Mode",
    value=False,
    help="Show recommendation explanations, similarity scores, and algorithm details"
)

# Select recommendation method
method = st.sidebar.selectbox(
    "🎯 Recommendation Method",
    ["Content-Based Filtering", "Collaborative Filtering", "User Segmentation", "Popular Movies"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Stats")
st.sidebar.metric("Movies", len(movies_df))
st.sidebar.metric("Users", len(users_df))
st.sidebar.metric("Ratings", len(ratings_df))
st.sidebar.metric("Avg Rating", f"{ratings_df['rating'].mean():.2f}")

# Helper function to display movie cards
def display_movie_cards(recommendations_df, show_explanations=False, max_cards=10):
    """Display movies in Netflix-style card layout with improved information."""
    if len(recommendations_df) == 0:
        st.warning("No recommendations available.")
        return
    
    # Limit number of cards
    display_df = recommendations_df.head(max_cards)
    
    # Create columns (3 movies per row)
    n_cols = 3
    n_rows = (len(display_df) + n_cols - 1) // n_cols
    
    for row in range(n_rows):
        cols = st.columns(n_cols)
        for col_idx in range(n_cols):
            idx = row * n_cols + col_idx
            if idx < len(display_df):
                movie = display_df.iloc[idx]
                with cols[col_idx]:
                    # Movie card
                    genres_display = movie['genres'].replace('|', ' • ')
                    st.markdown(f"""
                    <div class="movie-card">
                        <div class="movie-title">{movie['title']}</div>
                        <div class="movie-genres">{genres_display}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show scores/explanations in educational mode
                    if show_explanations:
                        if 'similarity_score' in movie:
                            # Color code similarity score
                            sim_score = movie['similarity_score']
                            if sim_score >= 0.5:
                                color = "🟢"
                            elif sim_score >= 0.3:
                                color = "🟡"
                            else:
                                color = "🟠"
                            st.caption(f"{color} Similarity: {sim_score:.3f}")
                        
                        if 'predicted_rating' in movie:
                            st.caption(f"⭐ Predicted Rating: {movie['predicted_rating']:.2f}/5.0")
                        
                        if 'explanation' in movie and pd.notna(movie['explanation']):
                            st.markdown(f"""
                            <div class="explanation-box">
                                <small>💡 {movie['explanation']}</small>
                            </div>
                            """, unsafe_allow_html=True)

# Main content area
if method == "Content-Based Filtering":
    st.markdown('<div class="section-header">🎯 Content-Based Recommendations</div>', unsafe_allow_html=True)
    st.markdown("**Find movies similar in genre and theme** - Based on combined genre and title analysis using TF-IDF and cosine similarity")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Movie selection
        movie_titles = movies_df['title'].tolist()
        selected_movie = st.selectbox("Select a movie:", movie_titles, key="content_movie")
        selected_movie_id = movies_df[movies_df['title'] == selected_movie]['movieId'].iloc[0]
        n_recs = st.slider("Number of recommendations:", 5, 20, 10, key="content_n_recs")
    
    with col2:
        # Show selected movie info
        movie_info = movies_df[movies_df['movieId'] == selected_movie_id].iloc[0]
        st.markdown("### Selected Movie")
        st.write(f"**{movie_info['title']}**")
        genres_display = movie_info['genres'].replace('|', ' • ')
        st.write(f"Genres: {genres_display}")
        if 'n_ratings' in movie_info:
            st.caption(f"📊 {movie_info['n_ratings']} ratings, ⭐ {movie_info.get('avg_rating', 'N/A'):.2f}")
    
    # Get recommendations
    if st.button("Get Recommendations", key="content_btn", type="primary"):
        with st.spinner("Analyzing movie content and finding similar movies..."):
            recommendations = content_model.get_similar_movies(
                selected_movie_id, 
                n_recs,
                include_explanation=educational_mode
            )
        
        if len(recommendations) > 0:
            # More honest header
            st.markdown(
                f'<div class="section-header">🎬 Movies similar in genre and theme to "{selected_movie}"</div>', 
                unsafe_allow_html=True
            )
            
            # Show info about recommendation quality
            if educational_mode:
                avg_similarity = recommendations['similarity_score'].mean()
                st.info(f"Average similarity score: {avg_similarity:.3f} (higher = more similar)")
            
            display_movie_cards(recommendations, show_explanations=educational_mode)
            
            # Debug information if educational mode
            if educational_mode:
                with st.expander("🔍 Why these movies were recommended"):
                    st.markdown("**Recommendation Logic:**")
                    st.write("- Movies are compared using TF-IDF on combined genre + title features")
                    st.write("- Cosine similarity measures how similar the content is")
                    st.write("- Only movies with similarity ≥ 0.2 are shown (quality filter)")
                    st.write("- Results are sorted by similarity score (highest first)")
                    
                    st.markdown("**Selected Movie Features:**")
                    ref_genres = set(movie_info['genres'].split('|'))
                    st.write(f"- Genres: {', '.join(sorted(ref_genres))}")
                    st.write(f"- Title keywords: {selected_movie}")
        else:
            st.warning("No similar movies found. Try selecting a different movie or lowering the similarity threshold.")

elif method == "Collaborative Filtering":
    st.markdown('<div class="section-header">👥 Collaborative Filtering Recommendations</div>', unsafe_allow_html=True)
    st.markdown("**Personalized recommendations** - Based on users with similar tastes using Matrix Factorization (SVD)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_ids = sorted(ratings_df['userId'].unique())
        selected_user = st.selectbox("Select a user:", user_ids, key="collab_user")
        n_recs = st.slider("Number of recommendations:", 5, 20, 10, key="collab_n_recs")
    
    with col2:
        # Show user stats
        user_ratings = collaborative_model.get_user_ratings(selected_user)
        st.markdown("### User Profile")
        st.metric("Movies Rated", len(user_ratings))
        st.metric("Avg Rating", f"{user_ratings['rating'].mean():.2f}")
        if len(user_ratings) == 0:
            st.warning("This user has no ratings. Using popularity-based recommendations.")
    
    # Get recommendations
    if st.button("Get Recommendations", key="collab_btn", type="primary"):
        with st.spinner("Generating personalized recommendations..."):
            if len(user_ratings) > 0:
                recommendations = collaborative_model.recommend_for_user(
                    selected_user, 
                    movies_df, 
                    n_recs,
                    include_explanation=educational_mode
                )
            else:
                # Fallback to popularity
                recommendations = popularity_model.recommend(n_recs)
                st.info("Using popularity-based recommendations (user has no rating history)")
        
        st.markdown('<div class="section-header">🎬 Recommended for You</div>', unsafe_allow_html=True)
        display_movie_cards(recommendations, show_explanations=educational_mode)
        
        # Show user's current ratings
        if len(user_ratings) > 0 and educational_mode:
            with st.expander("View Your Rated Movies"):
                user_ratings_display = user_ratings.merge(
                    movies_df[['movieId', 'title', 'genres']],
                    on='movieId'
                )[['title', 'genres', 'rating']].sort_values('rating', ascending=False)
                st.dataframe(user_ratings_display, use_container_width=True, hide_index=True)
        
        # Model evaluation
        if educational_mode:
            with st.expander("Model Performance"):
                eval_metrics = collaborative_model.evaluate()
                col1, col2 = st.columns(2)
                col1.metric("RMSE", f"{eval_metrics['RMSE']:.3f}")
                col2.metric("MAE", f"{eval_metrics['MAE']:.3f}")

elif method == "User Segmentation":
    st.markdown('<div class="section-header">🔍 User Segmentation Recommendations</div>', unsafe_allow_html=True)
    st.markdown("**Cluster-based recommendations** - Based on users with similar behavior patterns (K-Means clustering)")
    
    user_ids = sorted(ratings_df['userId'].unique())
    selected_user = st.selectbox("Select a user:", user_ids, key="seg_user")
    n_recs = st.slider("Number of recommendations:", 5, 20, 10, key="seg_n_recs")
    
    try:
        user_cluster = segmentation_model.get_user_cluster(selected_user)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Your Cluster", f"Cluster {user_cluster}")
        with col2:
            cluster_stats = segmentation_model.get_cluster_stats()
            cluster_size = cluster_stats[cluster_stats['cluster'] == user_cluster]['n_users'].iloc[0]
            st.metric("Cluster Size", f"{int(cluster_size)} users")
        with col3:
            avg_rating = cluster_stats[cluster_stats['cluster'] == user_cluster]['avg_rating_mean'].iloc[0]
            st.metric("Avg Cluster Rating", f"{avg_rating:.2f}")
        
        if st.button("Get Cluster Recommendations", key="seg_btn", type="primary"):
            with st.spinner("Generating cluster-based recommendations..."):
                recommendations = segmentation_model.recommend_for_cluster(
                    user_cluster, ratings_df, movies_df, n_recs
                )
            
            st.markdown('<div class="section-header">🎬 Popular in Your Cluster</div>', unsafe_allow_html=True)
            display_movie_cards(recommendations, show_explanations=educational_mode)
        
        if educational_mode:
            with st.expander("Cluster Statistics"):
                st.dataframe(cluster_stats, use_container_width=True, hide_index=True)
            
            with st.expander("Cluster Visualization (PCA)"):
                import matplotlib.pyplot as plt
                
                X_pca, pca = segmentation_model.get_pca_components(n_components=2)
                cluster_labels = segmentation_model.user_features['cluster'].values
                
                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                                   cmap='viridis', alpha=0.6, s=50)
                
                # Highlight selected user
                user_idx = segmentation_model.user_features[
                    segmentation_model.user_features['userId'] == selected_user
                ].index[0]
                ax.scatter(X_pca[user_idx, 0], X_pca[user_idx, 1], 
                          c='red', s=200, marker='*', edgecolors='black', linewidths=2)
                
                ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
                ax.set_title('User Clusters (PCA Visualization)')
                ax.legend(*scatter.legend_elements(), title="Clusters")
                plt.colorbar(scatter, ax=ax)
                
                st.pyplot(fig)
    
    except ValueError as e:
        st.error(f"Error: {e}")

elif method == "Popular Movies":
    st.markdown('<div class="section-header">🔥 Popular Movies</div>', unsafe_allow_html=True)
    st.markdown("**Trending recommendations** - Most popular movies based on rating count and average rating")
    
    n_recs = st.slider("Number of recommendations:", 5, 20, 10, key="pop_n_recs")
    
    if st.button("Show Popular Movies", key="pop_btn", type="primary"):
        with st.spinner("Loading popular movies..."):
            recommendations = popularity_model.recommend(n_recs)
        
        st.markdown('<div class="section-header">🎬 Trending Now</div>', unsafe_allow_html=True)
        display_movie_cards(recommendations, show_explanations=educational_mode)
        
        if educational_mode:
            st.markdown("### Popularity Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg Popularity Score", f"{recommendations['popularity_score'].mean():.3f}")
            col2.metric("Total Ratings", f"{recommendations['n_ratings'].sum():.0f}")
            col3.metric("Avg Rating", f"{recommendations['avg_rating'].mean():.2f}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #808080; padding: 2rem;">
    <p>Built with ❤️ using MovieLens Dataset • Not affiliated with Netflix</p>
    <p><small>This is a simulation of Netflix-style recommendation logic using open-source data</small></p>
</div>
""", unsafe_allow_html=True)
