# Quick Start Guide

## 🚀 Get Started in 3 Steps

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit App
```bash
streamlit run app/app.py
```

The app will automatically generate mock data if data files don't exist.

### 3. Explore Recommendations
- Open your browser at `http://localhost:8501`
- Select a recommendation method from the sidebar
- Get personalized movie recommendations!

## 📓 Run Jupyter Notebooks

```bash
jupyter notebook
```

Navigate to `notebooks/` and open:
- `01_eda.ipynb` - Explore the dataset
- `02_content_based.ipynb` - Content-based filtering
- `03_collaborative_filtering.ipynb` - Collaborative filtering
- `04_kmeans_segmentation.ipynb` - User segmentation

## 💡 Tips

- Data files are automatically generated in the `data/` directory on first run
- The system uses mock MovieLens-like data by default
- You can replace the CSV files in `data/` with real MovieLens data if desired

## 🐛 Troubleshooting

**Issue**: ModuleNotFoundError
**Solution**: Make sure you're running from the project root directory

**Issue**: Streamlit can't find modules
**Solution**: The app automatically adds the src directory to the path

**Issue**: Data generation is slow
**Solution**: This is normal on first run. Data is cached after generation.

