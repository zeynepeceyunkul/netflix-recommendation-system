"""
Popularity-based recommendation system.
Used as fallback for cold-start users.
"""

import pandas as pd
import numpy as np
from typing import List


class PopularityRecommender:
    """
    Popularity-based recommendation system.
    Recommends most popular movies (by rating count and average rating).
    """
    
    def __init__(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        """
        Initialize popularity-based recommender.
        
        Args:
            ratings_df: Ratings dataframe
            movies_df: Movies dataframe
        """
        self.ratings_df = ratings_df.copy()
        self.movies_df = movies_df.copy()
        self.popularity_scores = None
        self._calculate_popularity()
    
    def _calculate_popularity(self):
        """Calculate popularity scores for all movies."""
        # Get movie statistics
        movie_stats = self.ratings_df.groupby('movieId').agg({
            'rating': ['count', 'mean']
        }).reset_index()
        
        movie_stats.columns = ['movieId', 'n_ratings', 'avg_rating']
        
        # Normalize both metrics (0-1 scale)
        movie_stats['n_ratings_norm'] = (
            (movie_stats['n_ratings'] - movie_stats['n_ratings'].min()) /
            (movie_stats['n_ratings'].max() - movie_stats['n_ratings'].min() + 1e-10)
        )
        
        movie_stats['avg_rating_norm'] = (
            (movie_stats['avg_rating'] - movie_stats['avg_rating'].min()) /
            (movie_stats['avg_rating'].max() - movie_stats['avg_rating'].min() + 1e-10)
        )
        
        # Combined popularity score (weighted: 60% count, 40% rating)
        movie_stats['popularity_score'] = (
            0.6 * movie_stats['n_ratings_norm'] +
            0.4 * movie_stats['avg_rating_norm']
        )
        
        self.popularity_scores = movie_stats
    
    def recommend(self, n_recommendations: int = 10, 
                  exclude_movie_ids: List[int] = None) -> pd.DataFrame:
        """
        Get most popular movies.
        
        Args:
            n_recommendations: Number of recommendations
            exclude_movie_ids: List of movie IDs to exclude
            
        Returns:
            DataFrame with popular movies
        """
        # Sort by popularity
        popular = self.popularity_scores.sort_values(
            'popularity_score', 
            ascending=False
        )
        
        # Exclude specified movies
        if exclude_movie_ids:
            popular = popular[~popular['movieId'].isin(exclude_movie_ids)]
        
        # Get top N
        top_popular = popular.head(n_recommendations)
        
        # Merge with movie info
        recommendations = top_popular.merge(
            self.movies_df[['movieId', 'title', 'genres']],
            on='movieId',
            how='left'
        )
        
        return recommendations[['movieId', 'title', 'genres', 'popularity_score', 
                                'n_ratings', 'avg_rating']]

