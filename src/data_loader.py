"""
Data loading utilities for the recommendation system.
Handles loading and generating mock MovieLens-like datasets.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import os


def generate_mock_movies(n_movies: int = 1000) -> pd.DataFrame:
    """
    Generate a mock movies dataset with genres and titles.
    Uses realistic MovieLens-style titles and ensures multi-genre diversity.
    
    Args:
        n_movies: Number of movies to generate
        
    Returns:
        DataFrame with columns: movieId, title, genres
    """
    np.random.seed(42)
    
    genres_list = [
        'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
        'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
        'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    
    # Realistic movie title templates (inspired by real MovieLens titles)
    title_templates = {
        'Action': ['The Last Stand', 'Rapid Fire', 'Final Strike', 'Dark Mission', 'Deadly Force'],
        'Adventure': ['Journey to the Unknown', 'Lost Treasure', 'Mystic Quest', 'Ancient Secrets', 'Hidden Path'],
        'Comedy': ['Laugh Out Loud', 'Funny Business', 'Comedy Central', 'The Joke', 'Happy Times'],
        'Drama': ['The Long Road', 'Broken Dreams', 'Life Stories', 'The Decision', 'Fading Light'],
        'Sci-Fi': ['Future World', 'Space Station Alpha', 'Time Paradox', 'Alien Contact', 'Digital Reality'],
        'Horror': ['The Haunting', 'Dark Shadows', 'Midnight Terror', 'The Curse', 'Nightmare'],
        'Romance': ['Love Story', 'Heartstrings', 'Forever Yours', 'The Promise', 'True Love'],
        'Thriller': ['The Chase', 'Hidden Truth', 'Deadly Game', 'The Conspiracy', 'Final Hour'],
        'Mystery': ['The Secret', 'Unsolved Case', 'Missing Piece', 'The Clue', 'Dark Mystery'],
        'Fantasy': ['Magic Realm', 'The Enchanted', 'Mystical Powers', 'The Legend', 'Ancient Magic'],
        'Crime': ['The Heist', 'Undercover', 'Street Justice', 'The Syndicate', 'Lawless'],
        'War': ['Battlefield', 'The Front Line', 'War Stories', 'Combat Zone', 'The War'],
        'Western': ['The Outlaw', 'Frontier Justice', 'Desert Storm', 'The Ranch', 'Wild West']
    }
    
    # Common words for title variation
    common_words = ['The', 'A', 'An', 'In', 'On', 'At', 'Of', 'For', 'With']
    adjectives = ['Great', 'Last', 'First', 'Final', 'Dark', 'Bright', 'Lost', 'Hidden', 'Secret']
    
    movies = []
    
    for i in range(1, n_movies + 1):
        # Ensure multi-genre (2-4 genres per movie for better diversity)
        n_genres = np.random.randint(2, 5)  # Changed from 1-4 to 2-4
        selected_genres = np.random.choice(genres_list, size=n_genres, replace=False)
        genres = '|'.join(selected_genres)
        
        # Generate realistic title based on primary genre
        primary_genre = selected_genres[0]
        
        if primary_genre in title_templates:
            base_title = np.random.choice(title_templates[primary_genre])
        else:
            # Fallback for genres without specific templates
            base_title = f"The {primary_genre} Story"
        
        # Add variation to avoid exact duplicates
        if np.random.random() < 0.3:  # 30% chance to add adjective
            adjective = np.random.choice(adjectives)
            title = f"{adjective} {base_title}"
        elif np.random.random() < 0.2:  # 20% chance to add year
            year = np.random.randint(1980, 2024)
            title = f"{base_title} ({year})"
        else:
            title = base_title
        
        # Ensure unique titles
        if i > 1:
            existing_titles = [m['title'] for m in movies]
            counter = 1
            original_title = title
            while title in existing_titles:
                title = f"{original_title} {counter}"
                counter += 1
        
        movies.append({
            'movieId': i,
            'title': title,
            'genres': genres
        })
    
    return pd.DataFrame(movies)


def generate_mock_ratings(n_users: int = 500, n_movies: int = 1000, 
                         sparsity: float = 0.95, movies_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Generate mock user ratings with realistic patterns.
    
    Args:
        n_users: Number of users
        n_movies: Number of movies
        sparsity: Proportion of missing ratings (0.95 = 95% missing)
        movies_df: Optional movies dataframe (to avoid circular dependency)
        
    Returns:
        DataFrame with columns: userId, movieId, rating, timestamp
    """
    np.random.seed(42)
    
    # Total possible ratings
    total_possible = n_users * n_movies
    n_ratings = int(total_possible * (1 - sparsity))
    
    ratings = []
    
    # Create some user preference patterns
    user_preferences = {}
    for user_id in range(1, n_users + 1):
        # Each user has preferred genres
        preferred_genres = np.random.choice(
            ['Action', 'Comedy', 'Drama', 'Romance', 'Sci-Fi', 'Thriller'],
            size=np.random.randint(1, 4)
        )
        user_preferences[user_id] = preferred_genres
    
    # Get movies dataframe if not provided
    if movies_df is None:
        movies_df = generate_mock_movies(n_movies)
    
    for _ in range(n_ratings):
        user_id = np.random.randint(1, n_users + 1)
        movie_id = np.random.randint(1, n_movies + 1)
        
        # Get movie genres
        movie = movies_df[movies_df['movieId'] == movie_id]
        if len(movie) > 0:
            movie_genres = movie.iloc[0]['genres'].split('|')
            user_pref = user_preferences[user_id]
            
            # Higher rating if genres match user preferences
            if any(genre in user_pref for genre in movie_genres):
                rating = np.random.choice([4, 5], p=[0.3, 0.7])
            else:
                rating = np.random.choice([1, 2, 3, 4, 5], p=[0.2, 0.2, 0.3, 0.2, 0.1])
        else:
            rating = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.3, 0.3, 0.2])
        
        timestamp = np.random.randint(1000000000, 1600000000)
        
        ratings.append({
            'userId': user_id,
            'movieId': movie_id,
            'rating': rating,
            'timestamp': timestamp
        })
    
    return pd.DataFrame(ratings).drop_duplicates(subset=['userId', 'movieId'])


def generate_mock_users(n_users: int = 500) -> pd.DataFrame:
    """
    Generate mock user data.
    
    Args:
        n_users: Number of users
        
    Returns:
        DataFrame with columns: userId, age, gender, occupation
    """
    np.random.seed(42)
    genders = ['M', 'F', 'Other']
    occupations = [
        'engineer', 'student', 'teacher', 'doctor', 'artist',
        'lawyer', 'scientist', 'writer', 'manager', 'other'
    ]
    
    users = []
    for i in range(1, n_users + 1):
        users.append({
            'userId': i,
            'age': np.random.randint(18, 70),
            'gender': np.random.choice(genders),
            'occupation': np.random.choice(occupations)
        })
    
    return pd.DataFrame(users)


def load_movies(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load movies dataset from file or generate if not exists.
    
    Note: Generated data uses MovieLens-format, not Netflix data.
    
    Args:
        file_path: Path to movies.csv file
        
    Returns:
        DataFrame with movie data
    """
    if file_path is None:
        # Try to find data directory relative to project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        file_path = os.path.join(project_root, 'data', 'movies.csv')
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # Ensure multi-genre for existing data
        if 'genres' in df.columns:
            # Count genres per movie
            genre_counts = df['genres'].str.count('\|') + 1
            # Filter out single-genre movies or add secondary genre
            single_genre_mask = genre_counts == 1
            if single_genre_mask.any():
                print(f"Warning: {single_genre_mask.sum()} movies have only one genre. Consider enriching data.")
        return df
    else:
        print(f"Movies file not found at {file_path}. Generating improved mock data...")
        movies_df = generate_mock_movies()
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        movies_df.to_csv(file_path, index=False)
        return movies_df


def load_ratings(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load ratings dataset from file or generate if not exists.
    
    Args:
        file_path: Path to ratings.csv file
        
    Returns:
        DataFrame with ratings data
    """
    if file_path is None:
        # Try to find data directory relative to project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        file_path = os.path.join(project_root, 'data', 'ratings.csv')
    
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print(f"Ratings file not found at {file_path}. Generating mock data...")
        # First generate movies if needed, then use them for ratings
        movies_df = generate_mock_movies()
        ratings_df = generate_mock_ratings(movies_df=movies_df)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        ratings_df.to_csv(file_path, index=False)
        return ratings_df


def load_users(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load users dataset from file or generate if not exists.
    
    Args:
        file_path: Path to users.csv file
        
    Returns:
        DataFrame with user data
    """
    if file_path is None:
        # Try to find data directory relative to project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        file_path = os.path.join(project_root, 'data', 'users.csv')
    
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print(f"Users file not found at {file_path}. Generating mock data...")
        users_df = generate_mock_users()
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        users_df.to_csv(file_path, index=False)
        return users_df


def load_all_data(data_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all datasets (movies, ratings, users).
    
    Note: This function loads MovieLens-format data, not Netflix data.
    Netflix data is proprietary and not publicly available.
    
    Args:
        data_dir: Directory containing data files (if None, uses default location)
        
    Returns:
        Tuple of (movies_df, ratings_df, users_df)
    """
    if data_dir is None:
        # Use default paths (will be resolved in individual load functions)
        movies_df = load_movies()
        ratings_df = load_ratings()
        users_df = load_users()
    else:
        movies_path = os.path.join(data_dir, 'movies.csv')
        ratings_path = os.path.join(data_dir, 'ratings.csv')
        users_path = os.path.join(data_dir, 'users.csv')
        
        movies_df = load_movies(movies_path)
        ratings_df = load_ratings(ratings_path)
        users_df = load_users(users_path)
    
    return movies_df, ratings_df, users_df
