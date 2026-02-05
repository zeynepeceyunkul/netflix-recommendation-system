"""
Data preprocessing utilities for the recommendation system.
Includes feature engineering, data cleaning, and semantic enrichment.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple

try:
    from semantic_features import enrich_movies_semantic
except ImportError:
    enrich_movies_semantic = None


def preprocess_movies(movies_df: pd.DataFrame, ratings_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Preprocess movies data with derived features and semantic enrichment.
    Adds themes, emotional_tone, narrative_style, cinematography_style,
    pacing, target_audience, and semantic_text when semantic_features is available.
    
    Args:
        movies_df: Raw movies dataframe
        ratings_df: Optional ratings dataframe for popularity calculation
        
    Returns:
        Preprocessed movies dataframe with additional features
    """
    df = movies_df.copy()
    
    # Handle missing genres
    df['genres'] = df['genres'].fillna('Unknown')
    
    # Ensure movieId is integer
    df['movieId'] = df['movieId'].astype(int)
    
    # Semantic enrichment: themes, tone, narrative style, etc.
    if enrich_movies_semantic is not None:
        df = enrich_movies_semantic(df)
    
    # Add derived features if ratings are provided
    if ratings_df is not None:
        # Calculate movie popularity (number of ratings)
        movie_popularity = ratings_df.groupby('movieId').agg({
            'rating': ['count', 'mean']
        }).reset_index()
        movie_popularity.columns = ['movieId', 'n_ratings', 'avg_rating']
        
        # Merge with movies
        df = df.merge(movie_popularity, on='movieId', how='left')
        df['n_ratings'] = df['n_ratings'].fillna(0).astype(int)
        df['avg_rating'] = df['avg_rating'].fillna(0.0)
        
        # Normalize popularity (0-1 scale)
        if df['n_ratings'].max() > 0:
            df['popularity_score'] = (df['n_ratings'] - df['n_ratings'].min()) / (
                df['n_ratings'].max() - df['n_ratings'].min()
            )
        else:
            df['popularity_score'] = 0.0
    
    return df


def preprocess_ratings(ratings_df: pd.DataFrame, min_ratings_per_user: int = 5) -> pd.DataFrame:
    """
    Preprocess ratings data with cleaning and filtering.
    
    Args:
        ratings_df: Raw ratings dataframe
        min_ratings_per_user: Minimum number of ratings per user (filters sparse users)
        
    Returns:
        Preprocessed ratings dataframe
    """
    df = ratings_df.copy()
    
    # Remove duplicate ratings (keep first)
    df = df.drop_duplicates(subset=['userId', 'movieId'], keep='first')
    
    # Ensure correct data types
    df['userId'] = df['userId'].astype(int)
    df['movieId'] = df['movieId'].astype(int)
    df['rating'] = df['rating'].astype(float)
    
    # Remove invalid ratings (should be between 0.5 and 5.0)
    df = df[(df['rating'] >= 0.5) & (df['rating'] <= 5.0)]
    
    # Filter sparse users (users with too few ratings)
    user_rating_counts = df.groupby('userId').size()
    active_users = user_rating_counts[user_rating_counts >= min_ratings_per_user].index
    df = df[df['userId'].isin(active_users)]
    
    return df


def create_user_item_matrix(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create user-item rating matrix.
    
    Args:
        ratings_df: Ratings dataframe
        
    Returns:
        User-item matrix (users as rows, movies as columns)
    """
    matrix = ratings_df.pivot_table(
        index='userId',
        columns='movieId',
        values='rating',
        fill_value=0
    )
    return matrix


def get_user_features(ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract comprehensive user features for clustering:
    - Average rating
    - Number of movies rated (activity level)
    - Rating variance
    - Preferred genres (top 3)
    - User activity level (categorical)
    
    Args:
        ratings_df: Ratings dataframe
        movies_df: Movies dataframe
        
    Returns:
        DataFrame with user features
    """
    user_stats = ratings_df.groupby('userId').agg({
        'rating': ['mean', 'count', 'std']
    }).reset_index()
    
    user_stats.columns = ['userId', 'avg_rating', 'n_ratings', 'rating_std']
    user_stats['rating_std'] = user_stats['rating_std'].fillna(0)
    
    # Add user activity level (categorical feature)
    def categorize_activity(n_ratings):
        if n_ratings < 10:
            return 'Low'
        elif n_ratings < 50:
            return 'Medium'
        else:
            return 'High'
    
    user_stats['user_activity_level'] = user_stats['n_ratings'].apply(categorize_activity)
    
    # Merge with movies to get genre preferences
    ratings_with_genres = ratings_df.merge(
        movies_df[['movieId', 'genres']],
        on='movieId',
        how='left'
    )
    
    # Get top genres per user
    user_genres = []
    for user_id in user_stats['userId']:
        user_ratings = ratings_with_genres[ratings_with_genres['userId'] == user_id]
        user_ratings = user_ratings[user_ratings['rating'] >= 4]  # Only high ratings
        
        all_genres = []
        for genres_str in user_ratings['genres'].dropna():
            all_genres.extend(genres_str.split('|'))
        
        if all_genres:
            genre_counts = pd.Series(all_genres).value_counts()
            top_genres = genre_counts.head(3).index.tolist()
            user_genres.append('|'.join(top_genres))
        else:
            user_genres.append('')
    
    user_stats['top_genres'] = user_genres
    
    return user_stats


def normalize_features(features_df: pd.DataFrame, feature_cols: list) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Normalize features using StandardScaler.
    
    Args:
        features_df: DataFrame with features
        feature_cols: List of column names to normalize
        
    Returns:
        Tuple of (normalized dataframe, fitted scaler)
    """
    scaler = StandardScaler()
    df = features_df.copy()
    
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    return df, scaler
