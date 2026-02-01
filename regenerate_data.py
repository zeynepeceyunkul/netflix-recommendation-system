"""
Script to regenerate data files with improved quality.
Run this to update existing data with better titles and multi-genre movies.
"""

import os
import sys
sys.path.append('src')

from data_loader import generate_mock_movies, generate_mock_ratings, generate_mock_users

def regenerate_data():
    """Regenerate all data files with improved quality."""
    print("=" * 80)
    print("REGENERATING DATA FILES WITH IMPROVED QUALITY")
    print("=" * 80)
    
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    print("\n1. Generating improved movies...")
    movies_df = generate_mock_movies(n_movies=1000)
    movies_path = os.path.join(data_dir, 'movies.csv')
    movies_df.to_csv(movies_path, index=False)
    print(f"   ✓ Generated {len(movies_df)} movies")
    print(f"   ✓ Saved to {movies_path}")
    
    # Check genre distribution
    genre_counts = movies_df['genres'].str.count('\|') + 1
    print(f"   ✓ Movies with 2+ genres: {(genre_counts >= 2).sum()}/{len(movies_df)}")
    print(f"   ✓ Average genres per movie: {genre_counts.mean():.2f}")
    
    print("\n2. Generating ratings...")
    ratings_df = generate_mock_ratings(n_users=500, n_movies=1000, movies_df=movies_df)
    ratings_path = os.path.join(data_dir, 'ratings.csv')
    ratings_df.to_csv(ratings_path, index=False)
    print(f"   ✓ Generated {len(ratings_df)} ratings")
    print(f"   ✓ Saved to {ratings_path}")
    
    print("\n3. Generating users...")
    users_df = generate_mock_users(n_users=500)
    users_path = os.path.join(data_dir, 'users.csv')
    users_df.to_csv(users_path, index=False)
    print(f"   ✓ Generated {len(users_df)} users")
    print(f"   ✓ Saved to {users_path}")
    
    print("\n" + "=" * 80)
    print("Data regeneration complete!")
    print("=" * 80)
    print("\nSample movies:")
    print(movies_df[['title', 'genres']].head(10).to_string(index=False))
    print("\nYou can now run the Streamlit app with improved data quality!")

if __name__ == "__main__":
    regenerate_data()

