"""
Content-Based Filtering Recommendation System.
Uses TF-IDF on movie genres and cosine similarity.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple


class ContentBasedRecommender:
    """
    Content-based recommendation system using TF-IDF and cosine similarity.
    """
    
    def __init__(self, movies_df: pd.DataFrame):
        """
        Initialize the content-based recommender.
        
        Args:
            movies_df: DataFrame with columns: movieId, title, genres
        """
        self.movies_df = movies_df.copy()
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self._fit()
    
    def _fit(self):
        """Fit the TF-IDF vectorizer on movie genres."""
        # Prepare genres for TF-IDF (replace | with space)
        genres_text = self.movies_df['genres'].str.replace('|', ' ', regex=False)
        
        # Fit and transform
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(genres_text)
    
    def get_similar_movies(self, movie_id: int, n_recommendations: int = 10) -> pd.DataFrame:
        """
        Get similar movies based on content similarity.
        
        Args:
            movie_id: ID of the reference movie
            n_recommendations: Number of similar movies to return
            
        Returns:
            DataFrame with similar movies and similarity scores
        """
        # Find movie index
        movie_idx = self.movies_df[self.movies_df['movieId'] == movie_id].index
        
        if len(movie_idx) == 0:
            raise ValueError(f"Movie with ID {movie_id} not found")
        
        movie_idx = movie_idx[0]
        
        # Calculate cosine similarity
        cosine_sim = cosine_similarity(
            self.tfidf_matrix[movie_idx:movie_idx+1],
            self.tfidf_matrix
        ).flatten()
        
        # Get top similar movies (excluding the movie itself)
        similar_indices = cosine_sim.argsort()[::-1][1:n_recommendations+1]
        similar_scores = cosine_sim[similar_indices]
        
        # Create results dataframe
        results = self.movies_df.iloc[similar_indices].copy()
        results['similarity_score'] = similar_scores
        
        return results[['movieId', 'title', 'genres', 'similarity_score']]
    
    def recommend_for_user(self, user_ratings: pd.DataFrame, 
                          n_recommendations: int = 10) -> pd.DataFrame:
        """
        Recommend movies for a user based on their rated movies.
        
        Args:
            user_ratings: DataFrame with columns: movieId, rating
            n_recommendations: Number of recommendations to return
            
        Returns:
            DataFrame with recommended movies
        """
        # Get top-rated movies by user
        top_movies = user_ratings.nlargest(5, 'rating')['movieId'].tolist()
        
        # Get similar movies for each top-rated movie
        all_recommendations = []
        for movie_id in top_movies:
            try:
                similar = self.get_similar_movies(movie_id, n_recommendations=20)
                all_recommendations.append(similar)
            except ValueError:
                continue
        
        if not all_recommendations:
            return pd.DataFrame(columns=['movieId', 'title', 'genres', 'similarity_score'])
        
        # Combine and aggregate
        combined = pd.concat(all_recommendations, ignore_index=True)
        
        # Remove movies already rated by user
        rated_movie_ids = set(user_ratings['movieId'].tolist())
        combined = combined[~combined['movieId'].isin(rated_movie_ids)]
        
        # Aggregate by movie (average similarity score)
        aggregated = combined.groupby('movieId').agg({
            'title': 'first',
            'genres': 'first',
            'similarity_score': 'mean'
        }).reset_index()
        
        # Sort by similarity and return top N
        aggregated = aggregated.sort_values('similarity_score', ascending=False)
        
        return aggregated.head(n_recommendations)
    
    def get_movie_by_id(self, movie_id: int) -> pd.Series:
        """Get movie details by ID."""
        movie = self.movies_df[self.movies_df['movieId'] == movie_id]
        if len(movie) == 0:
            raise ValueError(f"Movie with ID {movie_id} not found")
        return movie.iloc[0]

