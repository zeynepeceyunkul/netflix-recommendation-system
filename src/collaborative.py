"""
Collaborative Filtering Recommendation System.
Uses Matrix Factorization (SVD) for rating prediction.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional


class CollaborativeFilteringRecommender:
    """
    Collaborative filtering recommendation system using SVD.
    """
    
    def __init__(self, ratings_df: pd.DataFrame, n_factors: int = 50, 
                 random_state: int = 42):
        """
        Initialize the collaborative filtering recommender.
        
        Args:
            ratings_df: DataFrame with columns: userId, movieId, rating
            n_factors: Number of latent factors for SVD
            random_state: Random seed
        """
        self.ratings_df = ratings_df.copy()
        self.n_factors = n_factors
        self.random_state = random_state
        self.model = None
        self.user_item_matrix = None
        self.user_ids = None
        self.movie_ids = None
        self.user_means = None
        self._fit()
    
    def _fit(self):
        """Fit the SVD model on ratings data."""
        # Create user-item matrix
        self.user_item_matrix = self.ratings_df.pivot_table(
            index='userId',
            columns='movieId',
            values='rating',
            fill_value=0
        )
        
        # Store user and movie IDs
        self.user_ids = self.user_item_matrix.index.values
        self.movie_ids = self.user_item_matrix.columns.values
        
        # Calculate user means for centering
        self.user_means = self.user_item_matrix.mean(axis=1)
        
        # Center the matrix (subtract user means)
        matrix_centered = self.user_item_matrix.sub(self.user_means, axis=0)
        
        # Apply SVD
        self.model = TruncatedSVD(
            n_components=self.n_factors,
            random_state=self.random_state
        )
        
        # Fit SVD on centered matrix
        self.user_factors = self.model.fit_transform(matrix_centered)
        self.movie_factors = self.model.components_.T
        
        # Store for predictions
        self.matrix_centered = matrix_centered
    
    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """
        Predict rating for a user-movie pair.
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            Predicted rating
        """
        # Find indices
        if user_id not in self.user_ids:
            # Return average rating if user not found
            return self.ratings_df['rating'].mean()
        
        if movie_id not in self.movie_ids:
            # Return user's average rating if movie not found
            user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]['rating']
            return user_ratings.mean() if len(user_ratings) > 0 else self.ratings_df['rating'].mean()
        
        user_idx = np.where(self.user_ids == user_id)[0][0]
        movie_idx = np.where(self.movie_ids == movie_id)[0][0]
        
        # Predict using matrix factorization: user_factor * movie_factor^T
        pred = np.dot(self.user_factors[user_idx], self.movie_factors[movie_idx])
        
        # Add user mean back
        user_mean = self.user_means.iloc[user_idx]
        pred_rating = pred + user_mean
        
        # Clip to valid rating range
        pred_rating = np.clip(pred_rating, 0.5, 5.0)
        
        return float(pred_rating)
    
    def recommend_for_user(self, user_id: int, movies_df: pd.DataFrame,
                          n_recommendations: int = 10,
                          exclude_rated: bool = True) -> pd.DataFrame:
        """
        Recommend movies for a user.
        
        Args:
            user_id: User ID
            movies_df: DataFrame with movie information
            n_recommendations: Number of recommendations to return
            exclude_rated: Whether to exclude movies already rated by user
            
        Returns:
            DataFrame with recommended movies and predicted ratings
        """
        # Get movies user hasn't rated
        if exclude_rated:
            rated_movies = set(
                self.ratings_df[self.ratings_df['userId'] == user_id]['movieId'].tolist()
            )
            candidate_movies = movies_df[~movies_df['movieId'].isin(rated_movies)]
        else:
            candidate_movies = movies_df
        
        # Predict ratings for all candidate movies
        predictions = []
        for _, movie in candidate_movies.iterrows():
            movie_id = movie['movieId']
            try:
                pred_rating = self.predict_rating(user_id, movie_id)
                predictions.append({
                    'movieId': movie_id,
                    'title': movie['title'],
                    'genres': movie['genres'],
                    'predicted_rating': pred_rating
                })
            except:
                continue
        
        if not predictions:
            return pd.DataFrame(columns=['movieId', 'title', 'genres', 'predicted_rating'])
        
        # Create results dataframe
        results = pd.DataFrame(predictions)
        results = results.sort_values('predicted_rating', ascending=False)
        
        return results.head(n_recommendations)
    
    def evaluate(self) -> dict:
        """
        Evaluate the model on test set.
        
        Returns:
            Dictionary with evaluation metrics (RMSE, MAE)
        """
        # Split data
        train_df, test_df = train_test_split(
            self.ratings_df,
            test_size=0.2,
            random_state=self.random_state
        )
        
        # Create a temporary model for evaluation
        temp_matrix = train_df.pivot_table(
            index='userId',
            columns='movieId',
            values='rating',
            fill_value=0
        )
        
        user_means_temp = temp_matrix.mean(axis=1)
        matrix_centered_temp = temp_matrix.sub(user_means_temp, axis=0)
        
        temp_model = TruncatedSVD(
            n_components=self.n_factors,
            random_state=self.random_state
        )
        user_factors_temp = temp_model.fit_transform(matrix_centered_temp)
        movie_factors_temp = temp_model.components_.T
        
        # Get movie and user IDs from temp matrix
        temp_user_ids = temp_matrix.index.values
        temp_movie_ids = temp_matrix.columns.values
        
        # Calculate predictions for test set
        errors = []
        for _, row in test_df.iterrows():
            user_id = row['userId']
            movie_id = row['movieId']
            actual_rating = row['rating']
            
            if user_id in temp_user_ids and movie_id in temp_movie_ids:
                user_idx = np.where(temp_user_ids == user_id)[0][0]
                movie_idx = np.where(temp_movie_ids == movie_id)[0][0]
                
                pred = np.dot(user_factors_temp[user_idx], movie_factors_temp[movie_idx])
                user_mean = user_means_temp.iloc[user_idx]
                pred_rating = np.clip(pred + user_mean, 0.5, 5.0)
                
                errors.append(abs(actual_rating - pred_rating))
        
        if len(errors) == 0:
            return {'RMSE': 0.0, 'MAE': 0.0}
        
        rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        mae = np.mean(errors)
        
        return {
            'RMSE': float(rmse),
            'MAE': float(mae)
        }
    
    def get_user_ratings(self, user_id: int) -> pd.DataFrame:
        """
        Get all ratings for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            DataFrame with user's ratings
        """
        return self.ratings_df[self.ratings_df['userId'] == user_id].copy()
