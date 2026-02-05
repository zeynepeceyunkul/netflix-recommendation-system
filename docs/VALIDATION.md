# Validation and Sanity Checks

## Purpose

Recommendations should feel **thematically and emotionally coherent**, not random or collapsed to a single genre. This document describes how to validate that and what we check.

## What We Validate

1. **Semantic coherence**: For a given movie, top recommendations should share themes and mood (e.g. space, isolation, melancholic) when using content-based mode.
2. **Diversity**: The list should not be 10 copies of the same genre combo; we cap recommendations per genre combination.
3. **Honest explanations**: With "Why was this recommended?" on, each card should show overlapping genres/themes/mood and a similarity score.

## How to Run Validation

### In the app

1. Start the app: `streamlit run app/app.py`
2. Select **Content-Based (themes & mood)**.
3. Pick a movie (e.g. one with Sci-Fi and Drama).
4. Enable **"Why was this recommended?"** in the sidebar.
5. Click **Get recommendations**.
6. Manually inspect:
   - Do the recommended movies share themes/mood with the selected one?
   - Are explanations (genres, themes, mood) accurate and readable?
   - Is the list diverse (not all same genre combo)?

### Command-line script

From the project root:

```bash
python src/validate_recommendations.py --movie "Space Station Alpha" --n 10
```

This prints the top 10 similar movies, their genres, similarity scores, common genres, and a short quality assessment (e.g. genre overlap count, average similarity). Use the output to document findings.

## Example Finding to Document

- **Movie**: "Future World" (Sci-Fi | Drama)
- **Expected**: Recommendations should lean toward sci-fi themes (e.g. technology, space, humanity) and dramatic tone (melancholic, contemplative).
- **Check**: Run the script or app, enable explanations, and note that top results share themes like "humanity", "isolation", "technology" and mood "contemplative" or "melancholic" where present in the enriched data.

## Data / Logic Separation

- **Data**: MovieLens (or generated) movies; semantic attributes are derived in `src/semantic_features.py` and attached in `preprocessing.preprocess_movies`.
- **Logic**: Similarity and diversity are in `src/content_based.py`; explanations are built from overlapping genres, themes, and emotional_tone.
- **UI**: Only presents what the logic produces; copy is honest ("similar themes, mood, and cinematic style", not "because you watched").
