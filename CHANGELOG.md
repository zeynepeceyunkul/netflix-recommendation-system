# Changelog - Project Refactoring

## Major Refactoring (2024)

### ✅ Data Clarity & Transparency
- **Added clear disclaimers** about data source (MovieLens, not Netflix)
- **Created DATA_SOURCE.md** with detailed documentation
- **Updated README** with "Data Source & Limitations" section
- **Added academic tone** throughout documentation
- **Clarified** what is realistic vs simulated

### ✅ Data Improvements
- **Enhanced preprocessing** with derived features:
  - `user_activity_level` (Low/Medium/High)
  - `movie_popularity` (normalized score)
  - `average_user_rating`
- **Filtered sparse users** (minimum 5 ratings per user)
- **Normalized popularity scores** for better recommendations
- **Improved data cleaning** pipeline

### ✅ Model Enhancements
- **Content-Based Filtering**:
  - Added similarity threshold filtering (0.1 minimum)
  - Explainable recommendations with reasoning
  - "Because you liked X" explanations
  - Common genre identification
  
- **Collaborative Filtering**:
  - Added explanation support
  - Better prediction score interpretation
  - User-specific reasoning
  
- **Popularity-Based Fallback**:
  - New module for cold-start users
  - Weighted popularity score (60% rating count, 40% avg rating)
  - Automatic fallback when user has no ratings

### ✅ UI/UX Redesign
- **Netflix-style movie cards** with dark theme
- **Improved layout** with 3-column grid
- **Better visual hierarchy** with section headers
- **Netflix-like section titles**:
  - "Because you watched..."
  - "Recommended for You"
  - "Popular in Your Cluster"
  - "Trending Now"
- **Custom CSS styling** for professional appearance
- **Responsive design** with proper spacing

### ✅ Educational Mode
- **Toggle switch** for showing/hiding explanations
- **Displays**:
  - Similarity scores (Content-Based)
  - Predicted ratings (Collaborative)
  - Cluster IDs (K-Means)
  - Recommendation reasoning
  - Model performance metrics
  - PCA visualizations

### ✅ Documentation
- **Professional README** with:
  - Clear data source disclaimers
  - Algorithm explanations with formulas
  - Academic tone suitable for CV/portfolio
  - Architecture diagrams
  - Evaluation metrics explained
  
- **DATA_SOURCE.md**:
  - Detailed data source information
  - Limitations clearly stated
  - Academic citation information
  - Legal/ethical considerations

### ✅ New Features
- **Popular Movies** recommendation method
- **TMDB API integration** (optional, cached):
  - Movie posters
  - Descriptions
  - Release dates
  - Rate-limited and cached locally

### 🔧 Technical Improvements
- **Better error handling**
- **Improved code organization**
- **Type hints** throughout
- **Comprehensive docstrings**
- **Modular design** for easy extension

---

## Files Modified

### Core Modules
- `src/preprocessing.py` - Enhanced with derived features
- `src/content_based.py` - Added explainable recommendations
- `src/collaborative.py` - Added explanation support
- `src/data_loader.py` - Added data source comments

### New Modules
- `src/popularity.py` - Popularity-based recommendations
- `src/tmdb_api.py` - TMDB API integration (optional)

### UI
- `app/app.py` - Complete redesign with Netflix-style UI

### Documentation
- `README.md` - Professional rewrite with disclaimers
- `DATA_SOURCE.md` - New data source documentation
- `CHANGELOG.md` - This file

---

## Breaking Changes

None - All changes are backward compatible. Existing functionality is preserved with enhancements.

---

## Migration Guide

No migration needed. The refactored code is fully compatible with existing data files and usage patterns.

---

## Future Enhancements

See README.md "Future Enhancements" section for planned improvements.

