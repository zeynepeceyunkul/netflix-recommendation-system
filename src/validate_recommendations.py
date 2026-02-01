"""
Validation script for content-based recommendations.
Tests recommendation quality by manually inspecting results.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import pandas as pd
from data_loader import load_all_data
from preprocessing import preprocess_movies, preprocess_ratings
from content_based import ContentBasedRecommender


def validate_recommendations(movie_title: str = None, n_recommendations: int = 10):
    """
    Validate content-based recommendations for a specific movie.
    
    Args:
        movie_title: Title of movie to test (if None, uses first movie)
        n_recommendations: Number of recommendations to generate
    """
    print("=" * 80)
    print("CONTENT-BASED RECOMMENDATION VALIDATION")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading data...")
    movies_df, ratings_df, users_df = load_all_data()
    movies_df = preprocess_movies(movies_df, ratings_df)
    ratings_df = preprocess_ratings(ratings_df)
    
    print(f"   ✓ Loaded {len(movies_df)} movies, {len(ratings_df)} ratings")
    
    # Initialize recommender
    print("\n2. Initializing content-based recommender...")
    recommender = ContentBasedRecommender(movies_df, similarity_threshold=0.2)
    print("   ✓ Recommender initialized")
    
    # Select test movie
    if movie_title is None:
        # Pick a movie with multiple genres for better testing
        multi_genre_movies = movies_df[movies_df['genres'].str.count('\|') >= 1]
        if len(multi_genre_movies) > 0:
            test_movie = multi_genre_movies.iloc[0]
        else:
            test_movie = movies_df.iloc[0]
        movie_title = test_movie['title']
    else:
        test_movie = movies_df[movies_df['title'] == movie_title].iloc[0]
    
    movie_id = test_movie['movieId']
    
    print(f"\n3. Testing with movie: '{movie_title}'")
    print(f"   Movie ID: {movie_id}")
    print(f"   Genres: {test_movie['genres']}")
    
    # Get recommendations
    print(f"\n4. Generating {n_recommendations} recommendations...")
    recommendations = recommender.get_similar_movies(
        movie_id, 
        n_recommendations=n_recommendations,
        include_explanation=True
    )
    
    if len(recommendations) == 0:
        print("   ✗ No recommendations found!")
        return
    
    print(f"   ✓ Found {len(recommendations)} recommendations")
    
    # Analyze recommendations
    print("\n5. Recommendation Analysis:")
    print("-" * 80)
    
    ref_genres = set(test_movie['genres'].split('|'))
    print(f"\nReference Movie: '{movie_title}'")
    print(f"Reference Genres: {', '.join(sorted(ref_genres))}")
    print("\nRecommended Movies:")
    
    genre_matches = []
    similarity_scores = []
    
    for idx, (_, rec) in enumerate(recommendations.iterrows(), 1):
        rec_genres = set(rec['genres'].split('|'))
        common_genres = ref_genres.intersection(rec_genres)
        similarity = rec['similarity_score']
        
        genre_matches.append(len(common_genres))
        similarity_scores.append(similarity)
        
        # Visual indicator
        match_indicator = "✓" if len(common_genres) > 0 else "✗"
        
        print(f"\n  {idx}. {rec['title']}")
        print(f"     Genres: {rec['genres'].replace('|', ', ')}")
        print(f"     Similarity: {similarity:.3f} {match_indicator}")
        print(f"     Common genres: {', '.join(sorted(common_genres)) if common_genres else 'None'}")
        if 'explanation' in rec:
            print(f"     Explanation: {rec['explanation']}")
    
    # Statistics
    print("\n6. Quality Metrics:")
    print("-" * 80)
    print(f"   Average similarity score: {pd.Series(similarity_scores).mean():.3f}")
    print(f"   Min similarity: {min(similarity_scores):.3f}")
    print(f"   Max similarity: {max(similarity_scores):.3f}")
    print(f"   Movies with genre overlap: {sum(1 for m in genre_matches if m > 0)}/{len(genre_matches)}")
    print(f"   Average genre matches: {pd.Series(genre_matches).mean():.2f}")
    
    # Quality assessment
    print("\n7. Quality Assessment:")
    print("-" * 80)
    
    avg_sim = pd.Series(similarity_scores).mean()
    genre_overlap_ratio = sum(1 for m in genre_matches if m > 0) / len(genre_matches)
    
    if avg_sim >= 0.4 and genre_overlap_ratio >= 0.8:
        print("   ✓ EXCELLENT: High similarity scores and good genre overlap")
    elif avg_sim >= 0.3 and genre_overlap_ratio >= 0.6:
        print("   ✓ GOOD: Reasonable similarity and genre matching")
    elif avg_sim >= 0.2:
        print("   ⚠ FAIR: Low similarity but meets threshold")
    else:
        print("   ✗ POOR: Recommendations may not be meaningful")
    
    # Recommendations
    print("\n8. Recommendations:")
    print("-" * 80)
    if avg_sim < 0.3:
        print("   • Consider lowering similarity threshold or improving data quality")
    if genre_overlap_ratio < 0.7:
        print("   • Ensure movies have multiple genres for better matching")
    if len(recommendations) < n_recommendations:
        print(f"   • Only {len(recommendations)}/{n_recommendations} recommendations found")
        print("   • Consider adjusting similarity threshold")
    
    print("\n" + "=" * 80)
    print("Validation complete!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate content-based recommendations")
    parser.add_argument("--movie", type=str, help="Movie title to test")
    parser.add_argument("--n", type=int, default=10, help="Number of recommendations")
    
    args = parser.parse_args()
    
    validate_recommendations(movie_title=args.movie, n_recommendations=args.n)

