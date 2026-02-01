"""
Evaluation metrics for recommendation systems.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple


def precision_at_k(recommended_items: List[int], relevant_items: List[int], k: int) -> float:
    """
    Calculate Precision@K.
    
    Args:
        recommended_items: List of recommended item IDs
        relevant_items: List of relevant (ground truth) item IDs
        k: Number of top recommendations to consider
        
    Returns:
        Precision@K score
    """
    if k == 0 or len(recommended_items) == 0:
        return 0.0
    
    # Get top k recommendations
    top_k = recommended_items[:k]
    
    # Count how many are relevant
    relevant_count = len(set(top_k) & set(relevant_items))
    
    return relevant_count / k


def recall_at_k(recommended_items: List[int], relevant_items: List[int], k: int) -> float:
    """
    Calculate Recall@K.
    
    Args:
        recommended_items: List of recommended item IDs
        relevant_items: List of relevant (ground truth) item IDs
        k: Number of top recommendations to consider
        
    Returns:
        Recall@K score
    """
    if len(relevant_items) == 0:
        return 0.0
    
    # Get top k recommendations
    top_k = recommended_items[:k]
    
    # Count how many relevant items were retrieved
    retrieved_relevant = len(set(top_k) & set(relevant_items))
    
    return retrieved_relevant / len(relevant_items)


def f1_at_k(recommended_items: List[int], relevant_items: List[int], k: int) -> float:
    """
    Calculate F1@K.
    
    Args:
        recommended_items: List of recommended item IDs
        relevant_items: List of relevant (ground truth) item IDs
        k: Number of top recommendations to consider
        
    Returns:
        F1@K score
    """
    prec = precision_at_k(recommended_items, relevant_items, k)
    rec = recall_at_k(recommended_items, relevant_items, k)
    
    if prec + rec == 0:
        return 0.0
    
    return 2 * (prec * rec) / (prec + rec)


def mean_reciprocal_rank(recommended_items: List[int], relevant_items: List[int]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    Args:
        recommended_items: List of recommended item IDs
        relevant_items: List of relevant (ground truth) item IDs
        
    Returns:
        MRR score
    """
    if len(relevant_items) == 0:
        return 0.0
    
    for idx, item in enumerate(recommended_items, start=1):
        if item in relevant_items:
            return 1.0 / idx
    
    return 0.0


def evaluate_recommendations(recommendations_df: pd.DataFrame,
                            test_ratings: pd.DataFrame,
                            k: int = 10,
                            rating_threshold: float = 4.0) -> dict:
    """
    Evaluate recommendations against test ratings.
    
    Args:
        recommendations_df: DataFrame with recommended movies (columns: movieId, ...)
        test_ratings: DataFrame with test ratings (columns: userId, movieId, rating)
        k: Number of top recommendations to consider
        rating_threshold: Minimum rating to consider as relevant
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Get relevant items (highly rated in test set)
    relevant_items = set(
        test_ratings[test_ratings['rating'] >= rating_threshold]['movieId'].tolist()
    )
    
    # Get recommended items
    recommended_items = recommendations_df['movieId'].tolist()
    
    # Calculate metrics
    metrics = {
        'Precision@K': precision_at_k(recommended_items, list(relevant_items), k),
        'Recall@K': recall_at_k(recommended_items, list(relevant_items), k),
        'F1@K': f1_at_k(recommended_items, list(relevant_items), k),
        'MRR': mean_reciprocal_rank(recommended_items, list(relevant_items))
    }
    
    return metrics


def calculate_coverage(recommendations_df: pd.DataFrame, 
                      all_movies: pd.DataFrame) -> float:
    """
    Calculate catalog coverage (percentage of movies recommended).
    
    Args:
        recommendations_df: DataFrame with recommendations
        all_movies: DataFrame with all available movies
        
    Returns:
        Coverage percentage
    """
    if len(all_movies) == 0:
        return 0.0
    
    recommended_movies = set(recommendations_df['movieId'].unique())
    all_movie_ids = set(all_movies['movieId'].unique())
    
    return len(recommended_movies) / len(all_movie_ids) * 100


def calculate_diversity(recommendations_df: pd.DataFrame,
                       movies_df: pd.DataFrame) -> float:
    """
    Calculate diversity of recommendations based on genre variety.
    
    Args:
        recommendations_df: DataFrame with recommendations
        movies_df: DataFrame with movie genres
        
    Returns:
        Diversity score (average number of unique genres per recommendation set)
    """
    if len(recommendations_df) == 0:
        return 0.0
    
    # Merge with movie genres
    rec_with_genres = recommendations_df.merge(
        movies_df[['movieId', 'genres']],
        on='movieId',
        how='left'
    )
    
    # Count unique genres
    all_genres = set()
    for genres_str in rec_with_genres['genres'].dropna():
        all_genres.update(genres_str.split('|'))
    
    return len(all_genres) / len(recommendations_df) if len(recommendations_df) > 0 else 0.0

