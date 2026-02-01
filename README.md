# 🎬 Netflix-like Recommendation System

A comprehensive end-to-end recommendation system that implements multiple machine learning approaches to suggest movies to users, similar to Netflix's recommendation engine.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Algorithms](#algorithms)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Real-World Applications](#real-world-applications)
- [Future Enhancements](#future-enhancements)

## 🎯 Overview

This project implements three different recommendation approaches:

1. **Content-Based Filtering**: Recommends movies based on genre similarity using TF-IDF and cosine similarity
2. **Collaborative Filtering**: Uses Matrix Factorization (SVD) to predict ratings based on user behavior patterns
3. **User Segmentation**: Groups users into clusters using K-Means and recommends movies based on cluster preferences

## ✨ Features

- **Multiple Recommendation Approaches**: Compare different ML algorithms
- **Interactive Streamlit UI**: Easy-to-use web interface for exploring recommendations
- **Comprehensive Evaluation**: Precision@K, RMSE, MAE, and other metrics
- **Data Visualization**: PCA visualization for user clusters, rating distributions, and more
- **Modular Design**: Clean, production-ready code structure
- **Mock Data Generation**: Automatically generates MovieLens-like datasets if data files are missing

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Web App                         │
│                  (User Interface)                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Content-    │ │ Collaborative│ │   K-Means    │
│  Based       │ │  Filtering   │ │ Segmentation │
│  Filtering   │ │   (SVD)      │ │              │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
              ┌─────────▼─────────┐
              │   Data Pipeline   │
              │  (Preprocessing)  │
              └─────────┬─────────┘
                        │
              ┌─────────▼─────────┐
              │   MovieLens Data  │
              │  (Movies/Ratings) │
              └───────────────────┘
```

## 🧠 Algorithms

### 1. Content-Based Filtering

**How it works:**
- Uses TF-IDF (Term Frequency-Inverse Document Frequency) to vectorize movie genres
- Calculates cosine similarity between movies
- Recommends movies similar to ones the user has rated highly

**Advantages:**
- No cold start problem for new users (if they rate a few movies)
- Explains recommendations (shows why movies are similar)
- Works well for niche content

**Use Cases:**
- Netflix: "Because you watched [Movie X]"
- Spotify: "Similar artists"
- Amazon: "Customers who viewed this also viewed"

### 2. Collaborative Filtering (SVD)

**How it works:**
- Creates a user-item rating matrix
- Uses Singular Value Decomposition (SVD) to factorize the matrix
- Predicts ratings for unseen movies based on latent factors
- Recommends movies with highest predicted ratings

**Advantages:**
- Discovers hidden patterns in user preferences
- Works well with sparse data
- Can find surprising recommendations

**Use Cases:**
- Netflix: Personalized homepage recommendations
- YouTube: "Recommended for you"
- E-commerce: "You might also like"

### 3. User Segmentation (K-Means)

**How it works:**
- Extracts user features (average rating, number of ratings, rating variance)
- Applies K-Means clustering to group similar users
- Recommends movies popular within the user's cluster

**Advantages:**
- Identifies user segments for targeted marketing
- Can combine with other approaches
- Provides insights into user behavior

**Use Cases:**
- Netflix: User personas and segment-based recommendations
- Spotify: Music taste clusters
- E-commerce: Customer segmentation for personalized campaigns

## 📁 Project Structure

```
netflix-recommendation-system/
│
├── data/                          # Data directory
│   ├── movies.csv                 # Movie metadata
│   ├── ratings.csv                # User ratings
│   └── users.csv                  # User information
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   ├── 02_content_based.ipynb    # Content-based filtering
│   ├── 03_collaborative_filtering.ipynb  # Collaborative filtering
│   └── 04_kmeans_segmentation.ipynb      # User segmentation
│
├── src/                           # Source code
│   ├── data_loader.py            # Data loading utilities
│   ├── preprocessing.py           # Data preprocessing
│   ├── content_based.py          # Content-based recommender
│   ├── collaborative.py          # Collaborative filtering
│   ├── kmeans_model.py           # K-Means segmentation
│   └── evaluation.py             # Evaluation metrics
│
├── app/                           # Streamlit application
│   └── app.py                    # Main Streamlit UI
│
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── .gitignore                    # Git ignore rules
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Steps

1. **Clone the repository** (or navigate to the project directory):
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

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**:
   ```bash
   streamlit run app/app.py
   ```

   The app will open in your browser at `http://localhost:8501`

## 💻 Usage

### Streamlit Web App

1. Launch the app:
   ```bash
   streamlit run app/app.py
   ```

2. Select a recommendation method from the sidebar:
   - **Content-Based Filtering**: Select a movie to find similar movies
   - **Collaborative Filtering**: Select a user to get personalized recommendations
   - **User Segmentation**: View user clusters and cluster-based recommendations

3. Explore recommendations and visualizations!

### Jupyter Notebooks

1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Navigate to the `notebooks/` directory and open:
   - `01_eda.ipynb` - Explore the dataset
   - `02_content_based.ipynb` - Content-based filtering implementation
   - `03_collaborative_filtering.ipynb` - Collaborative filtering implementation
   - `04_kmeans_segmentation.ipynb` - User segmentation analysis

### Python Scripts

You can also use the modules directly in Python:

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
similar_movies = content_recommender.get_similar_movies(movie_id=1, n_recommendations=10)
user_recommendations = collab_recommender.recommend_for_user(user_id=1, movies_df=movies_df)
```

## 📊 Evaluation Metrics

### Precision@K
Measures the proportion of recommended items that are relevant (rated ≥ 4.0).

### Recall@K
Measures the proportion of relevant items that were retrieved in the top K recommendations.

### RMSE (Root Mean Squared Error)
Measures the average magnitude of prediction errors in collaborative filtering.

### MAE (Mean Absolute Error)
Measures the average absolute difference between predicted and actual ratings.

### Coverage
Percentage of the catalog that gets recommended.

### Diversity
Measures the variety of genres in recommendations.

## 🌍 Real-World Applications

### Netflix
- **Content-Based**: "Because you watched [Movie]"
- **Collaborative**: Personalized homepage recommendations
- **Segmentation**: User personas for content strategy

### Spotify
- **Content-Based**: "Similar artists" based on audio features
- **Collaborative**: "Discover Weekly" playlist
- **Segmentation**: Music taste clusters

### Amazon
- **Content-Based**: "Customers who viewed this also viewed"
- **Collaborative**: "Recommended for you"
- **Segmentation**: Customer segments for targeted marketing

### YouTube
- **Content-Based**: "Similar videos" based on tags/description
- **Collaborative**: "Recommended for you" based on watch history
- **Segmentation**: Viewer behavior clusters

## 🔮 Future Enhancements

- [ ] Hybrid recommendation system (combining multiple approaches)
- [ ] Deep Learning models (Neural Collaborative Filtering)
- [ ] Real-time recommendation updates
- [ ] A/B testing framework
- [ ] Multi-armed bandit for exploration vs exploitation
- [ ] Explainable AI for recommendation explanations
- [ ] Support for implicit feedback (clicks, views, time spent)
- [ ] Graph-based recommendations
- [ ] Time-aware recommendations (trending, seasonal)
- [ ] Multi-objective optimization (diversity + relevance)

## 📝 License

This project is open source and available for educational purposes.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- MovieLens dataset for inspiration
- scikit-learn, Surprise, and Streamlit communities
- Netflix, Spotify, and other platforms for real-world examples

---

**Built with ❤️ for learning and experimentation**

