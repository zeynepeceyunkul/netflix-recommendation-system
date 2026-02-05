# 🎬 Netflix-like Recommendation System

A comprehensive machine learning project that implements multiple recommendation algorithms to simulate Netflix-style movie recommendations using open-source data.

---

## ⚠️ Data Source & Limitations

### **Important Disclaimer**

This project **DOES NOT use real Netflix data**. Netflix's user data, ratings, and movie catalog are proprietary and not publicly available.

### Why MovieLens Dataset?

- **Open Source**: MovieLens is a publicly available, research-oriented dataset
- **Similar Structure**: Contains user ratings, movie metadata, and genres (similar to what Netflix uses)
- **Research Standard**: Widely used in academic research for recommendation systems
- **Ethical**: No privacy concerns or data access restrictions

### What is Realistic vs Simulated?

**Realistic (Based on Real ML Principles):**
- ✅ Recommendation algorithms (Content-Based, Collaborative Filtering, SVD)
- ✅ Similarity calculations (TF-IDF, Cosine Similarity)
- ✅ Matrix factorization techniques
- ✅ User clustering approaches
- ✅ Evaluation metrics (RMSE, Precision@K, etc.)

**Simulated (Project-Specific):**
- ⚠️ Dataset: MovieLens (not Netflix's actual catalog)
- ⚠️ User behavior: Synthetic patterns based on MovieLens data
- ⚠️ Movie metadata: Limited to genre information (no posters, descriptions from Netflix)
- ⚠️ Scale: Smaller dataset compared to Netflix's millions of users

### Data Access

- **Primary Source**: MovieLens dataset (GroupLens Research, University of Minnesota)
- **Fallback**: If MovieLens data is unavailable, the system generates synthetic MovieLens-like data for demonstration
- **No Live API**: All data is processed locally; no external API dependencies during runtime

---

## 📋 Table of Contents

- [Overview](#overview)
- [Data Source & Limitations](#-data-source--limitations)
- [Architecture](#architecture)
- [Algorithms](#algorithms)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Academic & Industry Applications](#academic--industry-applications)
- [Future Enhancements](#future-enhancements)

---

## 🎯 Overview

This project implements three complementary recommendation approaches commonly used in production systems:

1. **Content-Based Filtering**: Recommends by **semantic similarity** (genres, themes, emotional tone, narrative style, cinematography) using TF-IDF and cosine similarity, with diversity-aware ranking.
2. **Collaborative Filtering**: Matrix Factorization (SVD) to predict ratings from user behavior.
3. **User Segmentation**: K-Means on user features; recommendations from the user's cluster.
4. **Popular**: Fallback by global popularity (rating count and average).

### Why This is "Netflix-like"

We simulate recommendation *logic*, not Netflix data: hybrid strategies, content and collaborative signals, and explainable suggestions. We do **not** use "Because you watched X" as a claim about real watch history; we phrase recommendations as **"Movies with similar themes, mood, and cinematic style"** to reflect how similarity is actually computed.

---

## Why Naive Genre-Only Systems Fail

- **Single-label collapse**: One genre per movie yields trivial, repetitive recommendations (e.g. all "Children").
- **No notion of mood or theme**: Two "Drama" films can be thematically opposite; genre alone cannot capture that.
- **No diversity**: Top-N by similarity often returns near-duplicates (same genre combo).
- **Weak explainability**: "Similar genre" is vague; users trust "similar themes and mood" more when it is grounded in richer features.

This project therefore **enriches** movies with derived **themes**, **emotional tone**, **narrative style**, and **cinematography style** (from genre combinations and optional metadata). Similarity is computed in this **semantic feature space**, and ranking is **diversity-aware** (capped per genre combination) so results feel thematically coherent but not identical.

---

## How Semantic Features Improve Quality

- **Themes** (e.g. isolation, revenge, coming-of-age) are derived from genres and combined so that "Sci-Fi | Drama" contributes both sci-fi and drama themes.
- **Emotional tone** (dark, hopeful, melancholic, intense) and **narrative style** (slow-burn, action-driven) are mapped from genres and used in the same TF-IDF text as genres.
- **Cosine similarity** is computed on this combined text, so recommendations align on themes and mood, not only on a single genre label.
- **Diversity**: We limit how many recommendations can share the exact same genre combination, so the list is not 10 copies of the same "type" of movie.

---

## UI Design Choices and Honesty

- **No Streamlit selectbox for primary choices**: Dropdown arrows are small and easy to misclick; dropdowns can close on click outside. We use **radio buttons** (for movie choice) and **number inputs** (for user ID) so the entire control is clearly clickable and behavior is predictable.
- **Honest copy**: We avoid "Because you watched X" (which implies watch history we do not have). We use **"Movies with similar themes, mood, and cinematic style to X"**.
- **Optional "Why was this recommended?"** toggle: When enabled, each card shows overlapping themes, mood, and similarity score so the logic is transparent and defensible in review.

---

## Validation and Sanity Check

To check that recommendations are thematically and emotionally coherent:

1. Run the app and pick a **Content-Based** method.
2. Choose a movie with a clear profile (e.g. Sci-Fi + Drama).
3. Enable **"Why was this recommended?"** and inspect overlapping themes and mood on each card.
4. Optionally run: `python src/validate_recommendations.py --movie "Your Movie Title" --n 10` to print top-N recommendations and quality metrics (genre overlap, average similarity). Use this to document findings (e.g. "Top 10 share themes X, Y and mood Z as expected").

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Streamlit Web Application                      │
│         (Interactive User Interface & Visualization)         │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Content-    │ │ Collaborative│ │   K-Means    │
│  Based       │ │  Filtering   │ │ Segmentation │
│  Filtering   │ │   (SVD)      │ │              │
│              │ │              │ │              │
│ TF-IDF +     │ │ Matrix       │ │ User         │
│ Cosine Sim   │ │ Factorization│ │ Clustering   │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
              ┌─────────▼─────────┐
              │   Data Pipeline   │
              │  (Preprocessing)  │
              │  - Feature Eng.   │
              │  - Normalization  │
              │  - Cleaning       │
              └─────────┬─────────┘
                        │
              ┌─────────▼─────────┐
              │   MovieLens Data  │
              │  (Open Source)    │
              │  - Movies         │
              │  - Ratings       │
              │  - Users         │
              └───────────────────┘
```

---

## 🧠 Algorithms

### 1. Content-Based Filtering

**Algorithm**: TF-IDF Vectorization + Cosine Similarity

**How it works:**
1. Convert movie genres into TF-IDF vectors
2. Calculate cosine similarity between movies
3. Recommend movies similar to ones the user rated highly

**Mathematical Foundation:**
- **TF-IDF**: `tf-idf(t,d) = tf(t,d) × idf(t)`
- **Cosine Similarity**: `cos(θ) = (A·B) / (||A|| × ||B||)`

**Advantages:**
- No cold start problem (works with just movie metadata)
- Explainable ("Similar to [Movie X]")
- Works well for niche content

**Limitations:**
- Limited to available features (genres only in this implementation)
- May create filter bubbles (only similar content)

---

### 2. Collaborative Filtering (SVD)

**Algorithm**: Singular Value Decomposition (Matrix Factorization)

**How it works:**
1. Create user-item rating matrix
2. Factorize matrix: `R ≈ U × Σ × V^T`
3. Predict missing ratings using latent factors
4. Recommend movies with highest predicted ratings

**Mathematical Foundation:**
- **SVD**: Decomposes matrix into three matrices capturing latent features
- **Prediction**: `r̂ᵢⱼ = uᵢ · vⱼ` (dot product of user and item factors)

**Advantages:**
- Discovers hidden patterns in user preferences
- Works well with sparse data
- Can find surprising recommendations

**Limitations:**
- Cold start problem (new users/movies)
- Computationally expensive for large datasets
- Less explainable than content-based

---

### 3. User Segmentation (K-Means)

**Algorithm**: K-Means Clustering on User Features

**How it works:**
1. Extract user features (avg rating, activity level, rating variance)
2. Apply K-Means clustering to group similar users
3. Recommend movies popular within user's cluster

**Mathematical Foundation:**
- **K-Means**: Minimizes within-cluster sum of squares
- **Objective**: `argmin Σᵢ Σₓ∈Cᵢ ||x - μᵢ||²`

**Advantages:**
- Identifies user personas
- Can combine with other approaches
- Provides insights into user behavior

**Limitations:**
- Requires sufficient user data
- Fixed number of clusters (may not fit all users)
- Less personalized than individual recommendations

---

## 📁 Project Structure

```
netflix-recommendation-system/
│
├── data/                          # Data directory
│   ├── movies.csv                 # Movie metadata (MovieLens format)
│   ├── ratings.csv                # User ratings (MovieLens format)
│   └── users.csv                  # User information (optional)
│
├── notebooks/                     # Jupyter notebooks for analysis
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   ├── 02_content_based.ipynb    # Content-based filtering
│   ├── 03_collaborative_filtering.ipynb  # Collaborative filtering
│   └── 04_kmeans_segmentation.ipynb      # User segmentation
│
├── src/                           # Source code modules
│   ├── data_loader.py            # Data loading utilities
│   ├── preprocessing.py           # Data preprocessing & feature engineering
│   ├── content_based.py          # Content-based recommender
│   ├── collaborative.py          # Collaborative filtering (SVD)
│   ├── kmeans_model.py           # K-Means segmentation
│   ├── popularity.py             # Popularity-based fallback
│   └── evaluation.py             # Evaluation metrics
│
├── app/                           # Streamlit application
│   └── app.py                    # Main UI application
│
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── .gitignore                    # Git ignore rules
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Steps

1. **Clone or navigate to the project directory:**
   ```bash
   cd netflix-recommendation-system
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app:**
   ```bash
   streamlit run app/app.py
   ```

   The app will open in your browser at `http://localhost:8501`

---

## 💻 Usage

### Streamlit Web Application

1. **Launch the app:**
   ```bash
   streamlit run app/app.py
   ```

2. **Select a recommendation method:**
   - **Content-Based Filtering**: Find movies similar to a selected movie
   - **Collaborative Filtering**: Get personalized recommendations for a user
   - **User Segmentation**: View user clusters and cluster-based recommendations

3. **Enable Educational Mode** (optional):
   - Toggle "Explain Recommendations" to see similarity scores, prediction scores, and reasoning

### Jupyter Notebooks

1. **Start Jupyter:**
   ```bash
   jupyter notebook
   ```

2. **Navigate to notebooks/** and explore:
   - `01_eda.ipynb` - Data exploration and statistics
   - `02_content_based.ipynb` - Content-based filtering analysis
   - `03_collaborative_filtering.ipynb` - Collaborative filtering analysis
   - `04_kmeans_segmentation.ipynb` - User segmentation analysis

### Python API

```python
from src.data_loader import load_all_data
from src.preprocessing import preprocess_movies, preprocess_ratings
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeFilteringRecommender

# Load data
movies_df, ratings_df, users_df = load_all_data()

# Initialize recommenders
content_recommender = ContentBasedRecommender(movies_df)
collab_recommender = CollaborativeFilteringRecommender(ratings_df)

# Get recommendations
similar_movies = content_recommender.get_similar_movies(
    movie_id=1, 
    n_recommendations=10
)
user_recommendations = collab_recommender.recommend_for_user(
    user_id=1, 
    movies_df=movies_df
)
```

---

## 📊 Evaluation Metrics

### Precision@K
Measures the proportion of recommended items that are relevant (rated ≥ 4.0).

**Formula**: `Precision@K = (# relevant items in top K) / K`

### Recall@K
Measures the proportion of relevant items that were retrieved.

**Formula**: `Recall@K = (# relevant items in top K) / (total relevant items)`

### RMSE (Root Mean Squared Error)
Measures the average magnitude of prediction errors.

**Formula**: `RMSE = √(Σ(predicted - actual)² / n)`

### MAE (Mean Absolute Error)
Measures the average absolute difference between predicted and actual ratings.

**Formula**: `MAE = Σ|predicted - actual| / n`

### Coverage
Percentage of the catalog that gets recommended.

**Formula**: `Coverage = (# unique movies recommended) / (total movies) × 100%`

### Diversity
Measures the variety of genres in recommendations.

**Formula**: `Diversity = (# unique genres) / (# recommendations)`

---

## 🎓 Academic & Industry Applications

### Research Applications

- **Algorithm Comparison**: Compare different recommendation approaches
- **Evaluation Metrics**: Study precision, recall, RMSE trade-offs
- **Cold Start Problem**: Investigate solutions for new users/movies
- **Hybrid Systems**: Combine multiple recommendation strategies

### Industry Applications

**Netflix** (Simulated):
- Content-Based: "Because you watched [Movie]"
- Collaborative: Personalized homepage
- Segmentation: User personas for content strategy

**Similar Platforms**:
- **Spotify**: Music recommendations
- **Amazon**: Product recommendations
- **YouTube**: Video recommendations
- **Goodreads**: Book recommendations

---

## ⚠️ Why Recommendations Can Look Wrong

### Data Quality Issues

**Single-Genre Movies:**
- Movies with only one genre have limited similarity signals
- Solution: Our system ensures minimum 2 genres per movie

**Sparse Data:**
- Limited user ratings reduce recommendation quality
- Solution: We filter users with < 5 ratings

**Synthetic Titles:**
- Generated titles may not capture real semantic meaning
- Solution: We use realistic MovieLens-style title templates

### Algorithm Limitations

**Content-Based Filtering:**
- Only uses available features (genres, titles)
- Cannot capture complex user preferences
- May create "filter bubbles" (only similar content)
- **Why Netflix uses hybrid systems**: Combines multiple approaches

**Similarity Thresholds:**
- Too low: Random recommendations
- Too high: No recommendations found
- Our default: 0.2 (balanced)

**TF-IDF Limitations:**
- Genre-only matching misses thematic similarities
- Solution: We combine genres + title keywords

### Best Practices

1. **Use Multi-Genre Movies**: Better for content-based matching
2. **Enable Educational Mode**: See similarity scores and explanations
3. **Try Different Methods**: Content-based vs Collaborative vs Hybrid
4. **Check Data Quality**: Ensure movies have 2+ genres

### When to Use What

- **Content-Based**: Good for niche content, explainable, no cold-start
- **Collaborative**: Better personalization, discovers hidden patterns
- **Hybrid**: Best of both worlds (future enhancement)

---

## 🔮 Future Enhancements

### Short-term
- [x] Better similarity thresholds and filtering
- [x] Popularity-based fallback for cold-start users
- [x] Combined genre + title features
- [ ] Hybrid recommendation system (weighted combination)
- [ ] TMDB API integration for movie posters/descriptions (cached)

### Medium-term
- [ ] Deep Learning models (Neural Collaborative Filtering)
- [ ] Real-time recommendation updates
- [ ] A/B testing framework
- [ ] Multi-armed bandit for exploration vs exploitation

### Long-term
- [ ] Graph-based recommendations
- [ ] Time-aware recommendations (trending, seasonal)
- [ ] Multi-objective optimization (diversity + relevance)
- [ ] Support for implicit feedback (clicks, views, time spent)

---

## 📝 License

This project is open source and available for educational and research purposes.

---

## 🙏 Acknowledgments

- **MovieLens Dataset**: GroupLens Research, University of Minnesota
- **Libraries**: scikit-learn, pandas, numpy, streamlit communities
- **Inspiration**: Netflix, Spotify, and other recommendation systems

---

## 📧 Contact & Contributions

This project is designed for:
- **Academic Research**: Algorithm comparison and evaluation
- **Portfolio Projects**: Demonstrating ML engineering skills
- **Learning**: Understanding recommendation systems

Contributions and improvements are welcome!

---

**Built with academic rigor and industry best practices** 🎓
