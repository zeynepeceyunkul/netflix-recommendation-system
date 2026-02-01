"""
User Segmentation using K-Means Clustering.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional


class UserSegmentation:
    """
    User segmentation using K-Means clustering.
    """
    
    def __init__(self, user_features: pd.DataFrame, n_clusters: int = 5,
                 random_state: int = 42):
        """
        Initialize user segmentation model.
        
        Args:
            user_features: DataFrame with user features (from preprocessing.get_user_features)
            n_clusters: Number of clusters
            random_state: Random seed
        """
        self.user_features = user_features.copy()
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = None
        self.feature_cols = ['avg_rating', 'n_ratings', 'rating_std']
        self.cluster_labels = None
        self._fit()
    
    def _fit(self):
        """Fit K-Means model on user features."""
        # Extract numeric features
        X = self.user_features[self.feature_cols].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit K-Means
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        
        self.cluster_labels = self.kmeans.fit_predict(X_scaled)
        self.user_features['cluster'] = self.cluster_labels
    
    def get_user_cluster(self, user_id: int) -> int:
        """
        Get cluster assignment for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Cluster label
        """
        user_row = self.user_features[self.user_features['userId'] == user_id]
        if len(user_row) == 0:
            raise ValueError(f"User {user_id} not found")
        return user_row.iloc[0]['cluster']
    
    def get_cluster_users(self, cluster_id: int) -> pd.DataFrame:
        """
        Get all users in a cluster.
        
        Args:
            cluster_id: Cluster ID
            
        Returns:
            DataFrame with users in the cluster
        """
        return self.user_features[self.user_features['cluster'] == cluster_id].copy()
    
    def get_cluster_stats(self) -> pd.DataFrame:
        """
        Get statistics for each cluster.
        
        Returns:
            DataFrame with cluster statistics
        """
        cluster_stats = self.user_features.groupby('cluster').agg({
            'avg_rating': ['mean', 'std'],
            'n_ratings': ['mean', 'std'],
            'rating_std': ['mean', 'std'],
            'userId': 'count'
        }).reset_index()
        
        cluster_stats.columns = [
            'cluster', 'avg_rating_mean', 'avg_rating_std',
            'n_ratings_mean', 'n_ratings_std',
            'rating_std_mean', 'rating_std_std', 'n_users'
        ]
        
        return cluster_stats
    
    def get_pca_components(self, n_components: int = 2) -> Tuple[np.ndarray, PCA]:
        """
        Get PCA components for visualization.
        
        Args:
            n_components: Number of PCA components
            
        Returns:
            Tuple of (transformed data, PCA object)
        """
        X = self.user_features[self.feature_cols].values
        X_scaled = self.scaler.transform(X)
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        return X_pca, pca
    
    def recommend_for_cluster(self, cluster_id: int, ratings_df: pd.DataFrame,
                             movies_df: pd.DataFrame, n_recommendations: int = 10) -> pd.DataFrame:
        """
        Recommend movies for a cluster based on cluster preferences.
        
        Args:
            cluster_id: Cluster ID
            ratings_df: Ratings dataframe
            movies_df: Movies dataframe
            n_recommendations: Number of recommendations
            
        Returns:
            DataFrame with recommended movies
        """
        # Get users in cluster
        cluster_users = self.get_cluster_users(cluster_id)['userId'].tolist()
        
        # Get ratings from cluster users
        cluster_ratings = ratings_df[ratings_df['userId'].isin(cluster_users)]
        
        # Get average rating per movie
        movie_ratings = cluster_ratings.groupby('movieId').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        
        movie_ratings.columns = ['movieId', 'avg_rating', 'n_ratings']
        
        # Filter movies with at least 5 ratings
        movie_ratings = movie_ratings[movie_ratings['n_ratings'] >= 5]
        
        # Sort by average rating
        movie_ratings = movie_ratings.sort_values('avg_rating', ascending=False)
        
        # Merge with movie info
        recommendations = movie_ratings.merge(
            movies_df[['movieId', 'title', 'genres']],
            on='movieId',
            how='left'
        )
        
        return recommendations.head(n_recommendations)

