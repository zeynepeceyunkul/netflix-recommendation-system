"""
Content-Based Filtering Recommendation System.
Uses TF-IDF on combined genres + title keywords and cosine similarity.
Includes explainable recommendations with improved similarity logic.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict
import re


class ContentBasedRecommender:
    """
    Content-based recommendation system using TF-IDF and cosine similarity.
    Uses combined genre + title features for better semantic understanding.
    Provides explainable recommendations with reasoning.
    """
    
    def __init__(self, movies_df: pd.DataFrame, similarity_threshold: float = 0.2, 
                 max_features: int = 5000):
        """
        Initialize the content-based recommender.
        
        Args:
            movies_df: DataFrame with columns: movieId, title, genres
            similarity_threshold: Minimum similarity score to consider (default 0.2 for better quality)
            max_features: Maximum number of features for TF-IDF (prevents overfitting)
        """
        self.movies_df = movies_df.copy()
        self.similarity_threshold = similarity_threshold
        self.max_features = max_features
        
        # Initialize TF-IDF with stop words removal and better parameters
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',  # Remove common English stop words
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]{2,}\b',  # Only words with 2+ characters
            ngram_range=(1, 2)  # Use unigrams and bigrams for better matching
        )
        self.tfidf_matrix = None
        self._fit()
    
    def _prepare_text_features(self, movies_df: pd.DataFrame) -> pd.Series:
        """
        Prepare combined text features from genres and titles.
        
        Args:
            movies_df: Movies dataframe
            
        Returns:
            Series of combined text features
        """
        combined_texts = []
        
        for _, row in movies_df.iterrows():
            # Get genres (replace | with space)
            genres = row['genres'].replace('|', ' ').lower()
            
            # Get title and clean it
            title = str(row['title']).lower()
            # Remove year in parentheses if present
            title = re.sub(r'\s*\(\d{4}\)\s*', ' ', title)
            # Remove special characters but keep spaces
            title = re.sub(r'[^\w\s]', ' ', title)
            
            # Combine: genres first (more important), then title keywords
            # Repeat genres once to give them more weight
            combined = f"{genres} {genres} {title}"
            
            combined_texts.append(combined)
        
        return pd.Series(combined_texts)
    
    def _fit(self):
        """Fit the TF-IDF vectorizer on combined genre + title features."""
        # Prepare combined text features
        combined_text = self._prepare_text_features(self.movies_df)
        
        # Fit and transform
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(combined_text)
    
    def get_similar_movies(self, movie_id: int, n_recommendations: int = 10, 
                          include_explanation: bool = False) -> pd.DataFrame:
        """
        Get similar movies based on content similarity.
        Uses improved similarity logic with proper thresholding and sorting.
        
        Args:
            movie_id: ID of the reference movie
            n_recommendations: Number of similar movies to return
            include_explanation: Whether to include explanation text
            
        Returns:
            DataFrame with similar movies, similarity scores, and optionally explanations
        """
        # Find movie index
        movie_idx = self.movies_df[self.movies_df['movieId'] == movie_id].index
        
        if len(movie_idx) == 0:
            raise ValueError(f"Movie with ID {movie_id} not found")
        
        movie_idx = movie_idx[0]
        reference_movie = self.movies_df.iloc[movie_idx]
        
        # Calculate cosine similarity
        cosine_sim = cosine_similarity(
            self.tfidf_matrix[movie_idx:movie_idx+1],
            self.tfidf_matrix
        ).flatten()
        
        # Get valid indices (exclude self and filter by threshold)
        valid_indices = np.where(cosine_sim >= self.similarity_threshold)[0]
        valid_indices = valid_indices[valid_indices != movie_idx]  # Exclude self
        
        if len(valid_indices) == 0:
            # If no movies meet threshold, lower it slightly and try again
            fallback_threshold = max(0.1, self.similarity_threshold * 0.5)
            valid_indices = np.where(cosine_sim >= fallback_threshold)[0]
            valid_indices = valid_indices[valid_indices != movie_idx]
            
            if len(valid_indices) == 0:
                return pd.DataFrame(columns=['movieId', 'title', 'genres', 'similarity_score', 'explanation'])
        
        # Sort strictly by similarity score (descending)
        sorted_indices = valid_indices[np.argsort(cosine_sim[valid_indices])[::-1]]
        top_indices = sorted_indices[:n_recommendations]
        similar_scores = cosine_sim[top_indices]
        
        # Create results dataframe
        results = self.movies_df.iloc[top_indices].copy()
        results['similarity_score'] = similar_scores
        
        # Add explanation if requested
        if include_explanation:
            explanations = []
            ref_genres = set(reference_movie['genres'].split('|'))
            
            for idx, row in results.iterrows():
                rec_genres = set(row['genres'].split('|'))
                common_genres = ref_genres.intersection(rec_genres)
                
                # Build explanation
                explanation_parts = []
                
                if common_genres:
                    genre_list = ', '.join(sorted(list(common_genres))[:3])
                    explanation_parts.append(f"Similar genres: {genre_list}")
                
                # Add similarity score
                explanation_parts.append(f"similarity: {row['similarity_score']:.3f}")
                
                explanations.append(" • ".join(explanation_parts))
            
            results['explanation'] = explanations
        
        return results[['movieId', 'title', 'genres', 'similarity_score'] + 
                      (['explanation'] if include_explanation else [])]
    
    def recommend_for_user(self, user_ratings: pd.DataFrame, 
                          n_recommendations: int = 10,
                          include_explanation: bool = False) -> pd.DataFrame:
        """
        Recommend movies for a user based on their rated movies.
        Includes explanations showing which movies influenced each recommendation.
        
        Args:
            user_ratings: DataFrame with columns: movieId, rating
            n_recommendations: Number of recommendations to return
            include_explanation: Whether to include explanation text
            
        Returns:
            DataFrame with recommended movies, scores, and optionally explanations
        """
        # Get top-rated movies by user (at least rating >= 3.5)
        top_movies = user_ratings[user_ratings['rating'] >= 3.5].nlargest(5, 'rating')['movieId'].tolist()
        
        if len(top_movies) == 0:
            return pd.DataFrame(columns=['movieId', 'title', 'genres', 'similarity_score'])
        
        # Get similar movies for each top-rated movie
        all_recommendations = []
        movie_sources = {}  # Track which movie influenced each recommendation
        
        for movie_id in top_movies:
            try:
                similar = self.get_similar_movies(movie_id, n_recommendations=20, include_explanation=False)
                # Track source movie
                source_movie = self.movies_df[self.movies_df['movieId'] == movie_id].iloc[0]
                for _, rec in similar.iterrows():
                    rec_id = rec['movieId']
                    if rec_id not in movie_sources:
                        movie_sources[rec_id] = []
                    movie_sources[rec_id].append({
                        'title': source_movie['title'],
                        'rating': user_ratings[user_ratings['movieId'] == movie_id]['rating'].iloc[0]
                    })
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
        
        # Aggregate by movie (weighted average by source movie rating)
        aggregated_list = []
        for movie_id in combined['movieId'].unique():
            movie_recs = combined[combined['movieId'] == movie_id]
            
            # Weight similarity scores by source movie ratings
            weighted_scores = []
            for _, rec in movie_recs.iterrows():
                sources = movie_sources.get(movie_id, [])
                if sources:
                    avg_source_rating = np.mean([s['rating'] for s in sources])
                    # Weight: higher source rating = more trust in similarity
                    weight = avg_source_rating / 5.0
                    weighted_scores.append(rec['similarity_score'] * weight)
                else:
                    weighted_scores.append(rec['similarity_score'])
            
            aggregated_list.append({
                'movieId': movie_id,
                'title': movie_recs.iloc[0]['title'],
                'genres': movie_recs.iloc[0]['genres'],
                'similarity_score': np.mean(weighted_scores) if weighted_scores else movie_recs['similarity_score'].mean()
            })
        
        aggregated = pd.DataFrame(aggregated_list)
        
        # Sort by weighted similarity and return top N
        aggregated = aggregated.sort_values('similarity_score', ascending=False)
        aggregated = aggregated.head(n_recommendations)
        
        # Add explanations if requested
        if include_explanation:
            explanations = []
            for _, row in aggregated.iterrows():
                rec_id = row['movieId']
                if rec_id in movie_sources:
                    sources = movie_sources[rec_id]
                    source_titles = [s['title'] for s in sources[:2]]  # Max 2 source movies
                    if len(source_titles) == 1:
                        explanations.append(
                            f"Similar to '{source_titles[0]}' (similarity: {row['similarity_score']:.3f})"
                        )
                    else:
                        explanations.append(
                            f"Similar to '{source_titles[0]}' and '{source_titles[1]}' "
                            f"(similarity: {row['similarity_score']:.3f})"
                        )
                else:
                    explanations.append(f"Similarity score: {row['similarity_score']:.3f}")
            
            aggregated['explanation'] = explanations
        
        return aggregated
    
    def get_movie_by_id(self, movie_id: int) -> pd.Series:
        """Get movie details by ID."""
        movie = self.movies_df[self.movies_df['movieId'] == movie_id]
        if len(movie) == 0:
            raise ValueError(f"Movie with ID {movie_id} not found")
        return movie.iloc[0]
