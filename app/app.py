"""
Streamlit App for Netflix-like Recommendation System
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
from evaluation import evaluate_recommendations

# Page configuration
st.set_page_config(
    page_title="Netflix Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# Title
st.title("🎬 Netflix-like Recommendation System")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    """Load and preprocess all data."""
    movies_df, ratings_df, users_df = load_all_data()
    movies_df = preprocess_movies(movies_df)
    ratings_df = preprocess_ratings(ratings_df)
    return movies_df, ratings_df, users_df

@st.cache_resource
def load_models(movies_df, ratings_df):
    """Load and cache ML models."""
    content_model = ContentBasedRecommender(movies_df)
    collaborative_model = CollaborativeFilteringRecommender(ratings_df)
    user_features = get_user_features(ratings_df, movies_df)
    segmentation_model = UserSegmentation(user_features, n_clusters=5)
    return content_model, collaborative_model, segmentation_model

# Load data
with st.spinner("Loading data and models..."):
    movies_df, ratings_df, users_df = load_data()
    content_model, collaborative_model, segmentation_model = load_models(movies_df, ratings_df)

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Select recommendation method
method = st.sidebar.selectbox(
    "Select Recommendation Method",
    ["Content-Based Filtering", "Collaborative Filtering", "User Segmentation"]
)

st.sidebar.markdown("---")

# Main content area
if method == "Content-Based Filtering":
    st.header("🎯 Content-Based Filtering")
    st.markdown("Recommends movies based on genre similarity using TF-IDF and cosine similarity.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Movie selection
        movie_titles = movies_df['title'].tolist()
        selected_movie = st.selectbox("Select a movie:", movie_titles)
        
        # Get movie ID
        selected_movie_id = movies_df[movies_df['title'] == selected_movie]['movieId'].iloc[0]
        
        # Number of recommendations
        n_recs = st.slider("Number of recommendations:", 5, 20, 10)
    
    with col2:
        # Show selected movie info
        movie_info = movies_df[movies_df['movieId'] == selected_movie_id].iloc[0]
        st.subheader("Selected Movie")
        st.write(f"**Title:** {movie_info['title']}")
        st.write(f"**Genres:** {movie_info['genres']}")
    
    # Get recommendations
    if st.button("Get Recommendations"):
        with st.spinner("Finding similar movies..."):
            recommendations = content_model.get_similar_movies(selected_movie_id, n_recs)
        
        st.subheader("🎬 Recommended Movies")
        st.dataframe(
            recommendations[['title', 'genres', 'similarity_score']].style.format({
                'similarity_score': '{:.3f}'
            }),
            use_container_width=True
        )
        
        # Display as cards
        st.markdown("### Movie Cards")
        cols = st.columns(3)
        for idx, (_, movie) in enumerate(recommendations.iterrows()):
            with cols[idx % 3]:
                st.markdown(f"""
                <div style="border: 1px solid #ddd; padding: 10px; margin: 5px; border-radius: 5px;">
                    <h4>{movie['title']}</h4>
                    <p><strong>Genres:</strong> {movie['genres']}</p>
                    <p><strong>Similarity:</strong> {movie['similarity_score']:.3f}</p>
                </div>
                """, unsafe_allow_html=True)

elif method == "Collaborative Filtering":
    st.header("👥 Collaborative Filtering")
    st.markdown("Recommends movies based on similar users' preferences using Matrix Factorization (SVD).")
    
    # User selection
    user_ids = sorted(ratings_df['userId'].unique())
    selected_user = st.selectbox("Select a user:", user_ids)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        n_recs = st.slider("Number of recommendations:", 5, 20, 10)
    
    with col2:
        # Show user stats
        user_ratings = collaborative_model.get_user_ratings(selected_user)
        st.subheader("User Stats")
        st.write(f"**Movies Rated:** {len(user_ratings)}")
        st.write(f"**Average Rating:** {user_ratings['rating'].mean():.2f}")
    
    # Get recommendations
    if st.button("Get Recommendations"):
        with st.spinner("Generating personalized recommendations..."):
            recommendations = collaborative_model.recommend_for_user(
                selected_user, movies_df, n_recs
            )
        
        st.subheader("🎬 Recommended Movies")
        st.dataframe(
            recommendations[['title', 'genres', 'predicted_rating']].style.format({
                'predicted_rating': '{:.2f}'
            }),
            use_container_width=True
        )
        
        # Show user's current ratings
        with st.expander("View User's Rated Movies"):
            user_ratings_display = user_ratings.merge(
                movies_df[['movieId', 'title', 'genres']],
                on='movieId'
            )[['title', 'genres', 'rating']].sort_values('rating', ascending=False)
            st.dataframe(user_ratings_display, use_container_width=True)
        
        # Model evaluation
        with st.expander("Model Performance"):
            eval_metrics = collaborative_model.evaluate()
            st.metric("RMSE", f"{eval_metrics['RMSE']:.3f}")
            st.metric("MAE", f"{eval_metrics['MAE']:.3f}")

elif method == "User Segmentation":
    st.header("🔍 User Segmentation (K-Means)")
    st.markdown("Groups users into clusters and recommends movies based on cluster preferences.")
    
    # User selection
    user_ids = sorted(ratings_df['userId'].unique())
    selected_user = st.selectbox("Select a user:", user_ids)
    
    # Get user cluster
    try:
        user_cluster = segmentation_model.get_user_cluster(selected_user)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("User Cluster", f"Cluster {user_cluster}")
        
        with col2:
            cluster_stats = segmentation_model.get_cluster_stats()
            cluster_size = cluster_stats[cluster_stats['cluster'] == user_cluster]['n_users'].iloc[0]
            st.metric("Cluster Size", f"{int(cluster_size)} users")
        
        with col3:
            avg_rating = cluster_stats[cluster_stats['cluster'] == user_cluster]['avg_rating_mean'].iloc[0]
            st.metric("Avg Cluster Rating", f"{avg_rating:.2f}")
        
        # Cluster statistics
        with st.expander("Cluster Statistics"):
            st.dataframe(cluster_stats, use_container_width=True)
        
        # Get recommendations for cluster
        n_recs = st.slider("Number of recommendations:", 5, 20, 10)
        
        if st.button("Get Cluster Recommendations"):
            with st.spinner("Generating cluster-based recommendations..."):
                recommendations = segmentation_model.recommend_for_cluster(
                    user_cluster, ratings_df, movies_df, n_recs
                )
            
            st.subheader("🎬 Recommended Movies for Your Cluster")
            st.dataframe(
                recommendations[['title', 'genres', 'avg_rating', 'n_ratings']].style.format({
                    'avg_rating': '{:.2f}',
                    'n_ratings': '{:.0f}'
                }),
                use_container_width=True
            )
        
        # Visualization
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

# Footer
st.markdown("---")
st.markdown("### 📊 Dataset Statistics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Movies", len(movies_df))
col2.metric("Total Users", len(users_df))
col3.metric("Total Ratings", len(ratings_df))
col4.metric("Avg Rating", f"{ratings_df['rating'].mean():.2f}")

