# Data Source Documentation

## Overview

This project uses the **MovieLens dataset**, an open-source research dataset provided by GroupLens Research at the University of Minnesota. It does **NOT** use Netflix's proprietary data.

## Why MovieLens?

1. **Publicly Available**: MovieLens is freely available for research and educational purposes
2. **Research Standard**: Widely used in academic research on recommendation systems
3. **Similar Structure**: Contains user ratings, movie metadata, and genres (similar to what Netflix uses)
4. **Ethical**: No privacy concerns or data access restrictions
5. **Reproducible**: Others can replicate our results using the same dataset

## Dataset Details

### MovieLens Dataset
- **Source**: GroupLens Research, University of Minnesota
- **Website**: https://grouplens.org/datasets/movielens/
- **License**: Open for research use
- **Format**: CSV files (movies.csv, ratings.csv, users.csv)

### Data Structure

**movies.csv**:
- `movieId`: Unique identifier
- `title`: Movie title
- `genres`: Pipe-separated list of genres

**ratings.csv**:
- `userId`: User identifier
- `movieId`: Movie identifier
- `rating`: Rating (typically 1-5)
- `timestamp`: Rating timestamp

**users.csv** (optional):
- `userId`: User identifier
- `age`: User age
- `gender`: User gender
- `occupation`: User occupation

## Limitations

### What We Don't Have (Netflix Data)
- ❌ Netflix's actual movie catalog
- ❌ Netflix user behavior data
- ❌ Netflix's proprietary algorithms
- ❌ Real-time viewing data
- ❌ Movie posters/thumbnails (unless using TMDB API)
- ❌ Detailed movie descriptions (unless using TMDB API)

### What We Simulate
- ✅ Recommendation algorithms (Content-Based, Collaborative Filtering)
- ✅ Similarity calculations (TF-IDF, Cosine Similarity)
- ✅ Matrix factorization (SVD)
- ✅ User clustering (K-Means)
- ✅ Evaluation metrics (RMSE, Precision@K, etc.)

## Data Generation

If MovieLens data files are not found in the `data/` directory, the system automatically generates synthetic MovieLens-like data for demonstration purposes. This ensures the project works out-of-the-box without requiring manual data download.

**Note**: Generated data is for demonstration only. For research purposes, use the official MovieLens dataset.

## Optional: TMDB API Integration

The project includes optional integration with The Movie Database (TMDB) API to fetch:
- Movie posters
- Movie descriptions
- Release dates
- Additional metadata

**To use TMDB API**:
1. Get a free API key from https://www.themoviedb.org/settings/api
2. Set environment variable: `export TMDB_API_KEY=your_key_here`
3. Or pass API key when initializing TMDBApi class

**Caching**: All TMDB API responses are cached locally to minimize API calls and respect rate limits.

## Academic Use

This dataset and project are suitable for:
- Academic research on recommendation systems
- Algorithm comparison studies
- Educational purposes
- Portfolio projects

## Citation

If using MovieLens dataset in academic work, please cite:

```
F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: 
History and Context. ACM Transactions on Interactive Intelligent Systems 
(TiiS) 5, 4, Article 19 (December 2015), 19 pages. 
DOI=http://dx.doi.org/10.1145/2827872
```

## Legal & Ethical Considerations

- ✅ All data used is publicly available
- ✅ No user privacy concerns (MovieLens data is anonymized)
- ✅ No proprietary data access
- ✅ Open source and reproducible
- ✅ Suitable for academic and educational use

---

**Last Updated**: 2024
**Dataset Version**: MovieLens Latest (varies by download)

