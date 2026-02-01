"""
TMDB (The Movie Database) API Integration.
Optional module for fetching movie posters and metadata.
Results are cached locally to avoid API rate limits.
"""

import pandas as pd
import requests
import json
import os
from typing import Optional, Dict
import time


class TMDBApi:
    """
    TMDB API client for fetching movie metadata.
    Results are cached locally to minimize API calls.
    """
    
    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "data/tmdb_cache"):
        """
        Initialize TMDB API client.
        
        Args:
            api_key: TMDB API key (optional, can be set via environment variable TMDB_API_KEY)
            cache_dir: Directory to cache API responses
        """
        self.api_key = api_key or os.getenv('TMDB_API_KEY')
        self.base_url = "https://api.themoviedb.org/3"
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.25  # 4 requests per second max
    
    def _get_cached(self, movie_id: int) -> Optional[Dict]:
        """Get cached movie data if available."""
        cache_file = os.path.join(self.cache_dir, f"{movie_id}.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def _cache_data(self, movie_id: int, data: Dict):
        """Cache movie data locally."""
        cache_file = os.path.join(self.cache_dir, f"{movie_id}.json")
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def search_movie(self, title: str, year: Optional[int] = None) -> Optional[Dict]:
        """
        Search for a movie by title.
        
        Args:
            title: Movie title
            year: Optional release year
            
        Returns:
            Movie data dictionary or None if not found
        """
        if not self.api_key:
            return None
        
        # Check cache first
        cache_key = f"search_{title}_{year}"
        cached = self._get_cached(hash(cache_key))
        if cached:
            return cached
        
        self._rate_limit()
        
        try:
            url = f"{self.base_url}/search/movie"
            params = {
                'api_key': self.api_key,
                'query': title,
                'language': 'en-US'
            }
            if year:
                params['year'] = year
            
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            if data.get('results') and len(data['results']) > 0:
                movie_data = data['results'][0]  # Get first result
                self._cache_data(hash(cache_key), movie_data)
                return movie_data
        except Exception as e:
            print(f"TMDB API error: {e}")
        
        return None
    
    def get_movie_details(self, tmdb_id: int) -> Optional[Dict]:
        """
        Get detailed movie information by TMDB ID.
        
        Args:
            tmdb_id: TMDB movie ID
            
        Returns:
            Detailed movie data or None
        """
        if not self.api_key:
            return None
        
        # Check cache
        cached = self._get_cached(tmdb_id)
        if cached:
            return cached
        
        self._rate_limit()
        
        try:
            url = f"{self.base_url}/movie/{tmdb_id}"
            params = {
                'api_key': self.api_key,
                'language': 'en-US'
            }
            
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            self._cache_data(tmdb_id, data)
            return data
        except Exception as e:
            print(f"TMDB API error: {e}")
        
        return None
    
    def get_poster_url(self, movie_data: Dict, size: str = "w500") -> Optional[str]:
        """
        Get movie poster URL.
        
        Args:
            movie_data: Movie data from TMDB API
            size: Image size (w92, w154, w185, w342, w500, w780, original)
            
        Returns:
            Poster URL or None
        """
        poster_path = movie_data.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/{size}{poster_path}"
        return None
    
    def enrich_movies(self, movies_df: pd.DataFrame, 
                     title_column: str = 'title') -> pd.DataFrame:
        """
        Enrich movies dataframe with TMDB data (posters, descriptions, etc.).
        Only enriches if API key is available.
        
        Args:
            movies_df: Movies dataframe
            title_column: Name of column containing movie titles
            
        Returns:
            Enriched dataframe with additional columns:
            - poster_url: URL to movie poster
            - overview: Movie description
            - release_date: Release date
            - tmdb_id: TMDB movie ID
        """
        if not self.api_key:
            print("TMDB API key not found. Skipping enrichment.")
            print("To enable: Set TMDB_API_KEY environment variable or pass api_key parameter")
            return movies_df
        
        enriched = movies_df.copy()
        enriched['poster_url'] = None
        enriched['overview'] = None
        enriched['release_date'] = None
        enriched['tmdb_id'] = None
        
        print(f"Enriching {len(movies_df)} movies with TMDB data...")
        for idx, row in movies_df.iterrows():
            title = row[title_column]
            movie_data = self.search_movie(title)
            
            if movie_data:
                enriched.at[idx, 'poster_url'] = self.get_poster_url(movie_data)
                enriched.at[idx, 'overview'] = movie_data.get('overview', '')
                enriched.at[idx, 'release_date'] = movie_data.get('release_date', '')
                enriched.at[idx, 'tmdb_id'] = movie_data.get('id')
            
            # Progress indicator
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(movies_df)} movies...")
        
        print("Enrichment complete!")
        return enriched


def get_tmdb_client(api_key: Optional[str] = None) -> Optional[TMDBApi]:
    """
    Get TMDB API client if API key is available.
    
    Args:
        api_key: Optional API key (otherwise uses TMDB_API_KEY env var)
        
    Returns:
        TMDBApi instance or None if no API key
    """
    key = api_key or os.getenv('TMDB_API_KEY')
    if key:
        return TMDBApi(api_key=key)
    return None

