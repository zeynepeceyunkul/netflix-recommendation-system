"""
Content-Based Filtering Recommendation System.
Uses TF-IDF on semantic text (themes, mood, narrative style, genres) and cosine similarity.
Includes diversity-aware ranking and explainable recommendations.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict, Optional
import re


class ContentBasedRecommender:
    """
    Content-based recommendation using a semantic feature space: genres, themes,
    emotional tone, narrative style, and cinematography. Uses TF-IDF and cosine
    similarity with diversity-aware ranking to avoid same-genre collapse.
    """
    
    def __init__(self, movies_df: pd.DataFrame, similarity_threshold: float = 0.15,
                 max_features: int = 5000, max_same_genre_combo: int = 3):
        """
        Args:
            movies_df: DataFrame with movieId, title, genres; optional: semantic_text, themes, emotional_tone
            similarity_threshold: Minimum cosine similarity to consider
            max_features: TF-IDF max features
            max_same_genre_combo: Max recommendations per same genre combination (diversity)
        """
        self.movies_df = movies_df.copy()
        self.similarity_threshold = similarity_threshold
        self.max_features = max_features
        self.max_same_genre_combo = max_same_genre_combo
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]{2,}\b',
            ngram_range=(1, 2),
        )
        self.tfidf_matrix = None
        self._fit()
    
    def _get_text_for_tfidf(self, row: pd.Series) -> str:
        """Single blob for TF-IDF: prefer semantic_text, else genres + title."""
        if 'semantic_text' in row and pd.notna(row.get('semantic_text')) and str(row['semantic_text']).strip():
            return str(row['semantic_text']).strip()
        genres = row.get('genres', '') or ''
        genres = str(genres).replace('|', ' ').lower()
        title = str(row.get('title', '')).lower()
        title = re.sub(r'\s*\(\d{4}\)\s*', ' ', title)
        title = re.sub(r'[^\w\s]', ' ', title)
        return f"{genres} {genres} {title}".strip()
    
    def _fit(self):
        """Fit TF-IDF on semantic text (or fallback genre+title)."""
        texts = self.movies_df.apply(self._get_text_for_tfidf, axis=1)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts.astype(str))
    
    def _genre_combo_key(self, genres_str: str) -> str:
        """Normalized genre combo for diversity grouping."""
        if pd.isna(genres_str):
            return 'unknown'
        return '|'.join(sorted(g.strip() for g in str(genres_str).split('|') if g.strip()))
    
    def _diversity_rank(self, indices: np.ndarray, scores: np.ndarray, n_want: int) -> np.ndarray:
        """Select up to n_want indices, capping per genre-combo to avoid collapse."""
        df = self.movies_df.iloc[indices]
        df = df.copy()
        df['_score'] = scores
        df['_genre_combo'] = df['genres'].map(self._genre_combo_key)
        df = df.sort_values('_score', ascending=False)
        
        chosen = []
        combo_counts: Dict[str, int] = {}
        for idx in df.index:
            if len(chosen) >= n_want:
                break
            combo = df.loc[idx, '_genre_combo']
            if combo_counts.get(combo, 0) >= self.max_same_genre_combo:
                continue
            chosen.append(idx)
            combo_counts[combo] = combo_counts.get(combo, 0) + 1
        
        return np.array(chosen)
    
    def _explain_recommendation(self, reference_movie: pd.Series, rec_row: pd.Series, score: float) -> str:
        """Build honest explanation: overlapping themes, mood, similarity score."""
        parts = []
        
        ref_g = set(str(reference_movie.get('genres', '')).split('|'))
        rec_g = set(str(rec_row.get('genres', '')).split('|'))
        common_g = ref_g & rec_g
        if common_g:
            parts.append("Similar genres: " + ", ".join(sorted(common_g)[:3]))
        
        if 'themes' in reference_movie and 'themes' in rec_row:
            ref_t = set(str(reference_movie['themes']).split('|'))
            rec_t = set(str(rec_row['themes']).split('|'))
            common_t = ref_t & rec_t
            if common_t:
                parts.append("Themes: " + ", ".join(sorted(common_t)[:3]))
        
        if 'emotional_tone' in reference_movie and 'emotional_tone' in rec_row:
            ref_e = set(str(reference_movie['emotional_tone']).split('|'))
            rec_e = set(str(rec_row['emotional_tone']).split('|'))
            common_e = ref_e & rec_e
            if common_e:
                parts.append("Mood: " + ", ".join(sorted(common_e)[:3]))
        
        if 'cinematography_style' in reference_movie and 'cinematography_style' in rec_row:
            ref_c = set(str(reference_movie['cinematography_style']).split('|'))
            rec_c = set(str(rec_row['cinematography_style']).split('|'))
            common_c = ref_c & rec_c
            if common_c:
                parts.append("Style: " + ", ".join(sorted(common_c)[:3]))
        
        parts.append(f"Similarity: {score:.2f}")
        return " · ".join(parts)
    
    def get_similar_movies(self, movie_id: int, n_recommendations: int = 10,
                           include_explanation: bool = False) -> pd.DataFrame:
        """
        Get similar movies: exclude self, apply threshold, sort by similarity,
        then diversity-aware selection. Explanations mention themes and mood when available.
        """
        movie_idx = self.movies_df[self.movies_df['movieId'] == movie_id].index
        if len(movie_idx) == 0:
            raise ValueError(f"Movie with ID {movie_id} not found")
        movie_idx = movie_idx[0]
        reference_movie = self.movies_df.iloc[movie_idx]
        
        cosine_sim = cosine_similarity(
            self.tfidf_matrix[movie_idx:movie_idx + 1],
            self.tfidf_matrix,
        ).flatten()
        
        valid = np.where(cosine_sim >= self.similarity_threshold)[0]
        valid = valid[valid != movie_idx]
        if len(valid) == 0:
            fallback = max(0.08, self.similarity_threshold * 0.5)
            valid = np.where(cosine_sim >= fallback)[0]
            valid = valid[valid != movie_idx]
        if len(valid) == 0:
            return pd.DataFrame(columns=['movieId', 'title', 'genres', 'similarity_score', 'explanation'])
        
        sorted_idx = valid[np.argsort(cosine_sim[valid])[::-1]]
        scores = cosine_sim[sorted_idx]
        # Diversity-aware: take more than we need then trim
        take = min(len(sorted_idx), n_recommendations * 2)
        selected = self._diversity_rank(sorted_idx[:take], scores[:take], n_recommendations)
        if len(selected) == 0:
            selected = sorted_idx[:n_recommendations]
            scores = cosine_sim[selected]
        else:
            scores = cosine_sim[selected]
        
        results = self.movies_df.iloc[selected].copy()
        results['similarity_score'] = scores
        
        out_cols = ['movieId', 'title', 'genres', 'similarity_score']
        for extra in ('themes', 'emotional_tone', 'cinematography_style'):
            if extra in results.columns:
                out_cols.append(extra)
        if include_explanation:
            results['explanation'] = [
                self._explain_recommendation(reference_movie, results.iloc[i], results.iloc[i]['similarity_score'])
                for i in range(len(results))
            ]
            out_cols.append('explanation')
        
        return results[[c for c in out_cols if c in results.columns]]
    
    def recommend_for_user(self, user_ratings: pd.DataFrame, n_recommendations: int = 10,
                           include_explanation: bool = False) -> pd.DataFrame:
        """Recommend based on user's top-rated movies; explanations reference themes and mood."""
        top = user_ratings[user_ratings['rating'] >= 3.5].nlargest(5, 'rating')['movieId'].tolist()
        if not top:
            return pd.DataFrame(columns=['movieId', 'title', 'genres', 'similarity_score'])
        
        all_recs = []
        movie_sources = {}
        for movie_id in top:
            try:
                similar = self.get_similar_movies(movie_id, n_recommendations=20, include_explanation=False)
                src = self.movies_df[self.movies_df['movieId'] == movie_id].iloc[0]
                r = user_ratings[user_ratings['movieId'] == movie_id]['rating'].iloc[0]
                for _, rec in similar.iterrows():
                    rid = rec['movieId']
                    if rid not in movie_sources:
                        movie_sources[rid] = []
                    movie_sources[rid].append({'title': src['title'], 'rating': r})
                all_recs.append(similar)
            except ValueError:
                continue
        if not all_recs:
            return pd.DataFrame(columns=['movieId', 'title', 'genres', 'similarity_score'])
        
        combined = pd.concat(all_recs, ignore_index=True)
        rated = set(user_ratings['movieId'].tolist())
        combined = combined[~combined['movieId'].isin(rated)]
        
        agg_list = []
        for mid in combined['movieId'].unique():
            sub = combined[combined['movieId'] == mid]
            w = movie_sources.get(mid, [])
            wgt = np.mean([x['rating'] for x in w]) / 5.0 if w else 1.0
            sc = (sub['similarity_score'] * wgt).mean()
            row = sub.iloc[0]
            agg_list.append({
                'movieId': mid,
                'title': row['title'],
                'genres': row['genres'],
                'similarity_score': sc,
            })
        agg = pd.DataFrame(agg_list).sort_values('similarity_score', ascending=False).head(n_recommendations)
        
        if include_explanation:
            agg['explanation'] = [
                f"Similar themes and mood to movies you liked (score: {row['similarity_score']:.2f})"
                for _, row in agg.iterrows()
            ]
        return agg
    
    def get_movie_by_id(self, movie_id: int) -> pd.Series:
        """Get movie details by ID."""
        m = self.movies_df[self.movies_df['movieId'] == movie_id]
        if len(m) == 0:
            raise ValueError(f"Movie with ID {movie_id} not found")
        return m.iloc[0]
