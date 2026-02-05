"""
Semantic movie representation for recommendation quality.
Derives themes, emotional tone, narrative style, and related attributes
from genres (and optional title keywords) so recommendations reflect
mood, themes, and cinematic style—not just genre labels.
"""

import pandas as pd
import numpy as np
from typing import List, Set
import re

# ---------------------------------------------------------------------------
# Rule-based mappings: genre -> semantic attributes
# (No single-label identity: we always combine with at least one other dimension.)
# ---------------------------------------------------------------------------

GENRE_TO_THEMES = {
    "Action": ["conflict", "survival", "revenge", "rescue", "justice"],
    "Adventure": ["quest", "discovery", "journey", "exploration", "treasure"],
    "Animation": ["imagination", "coming-of-age", "family", "fantasy world"],
    "Children": ["family", "innocence", "learning", "friendship", "adventure"],
    "Comedy": ["everyday life", "relationships", "misunderstanding", "redemption"],
    "Crime": ["justice", "moral ambiguity", "revenge", "betrayal", "survival"],
    "Documentary": ["reality", "investigation", "society", "history", "nature"],
    "Drama": ["human condition", "relationships", "conflict", "choices", "loss"],
    "Fantasy": ["magic", "quest", "good vs evil", "mythology", "other worlds"],
    "Film-Noir": ["moral ambiguity", "fate", "betrayal", "urban isolation"],
    "Horror": ["fear", "survival", "isolation", "unknown", "darkness"],
    "Musical": ["expression", "romance", "dreams", "community", "transformation"],
    "Mystery": ["investigation", "truth", "secrets", "puzzle", "revelation"],
    "Romance": ["love", "relationships", "sacrifice", "longing", "connection"],
    "Sci-Fi": ["technology", "space", "future", "humanity", "isolation", "exploration"],
    "Thriller": ["suspense", "danger", "chase", "secrets", "survival"],
    "War": ["conflict", "sacrifice", "brotherhood", "survival", "moral cost"],
    "Western": ["frontier", "justice", "honor", "freedom", "law and order"],
}

GENRE_TO_EMOTIONAL_TONE = {
    "Action": ["intense", "adrenaline"],
    "Adventure": ["hopeful", "wonder"],
    "Animation": ["light-hearted", "hopeful", "whimsical"],
    "Children": ["light-hearted", "hopeful", "warm"],
    "Comedy": ["light-hearted", "warm", "playful"],
    "Crime": ["dark", "tense", "gritty"],
    "Documentary": ["reflective", "sober", "informative"],
    "Drama": ["melancholic", "intense", "hopeful", "sober"],
    "Fantasy": ["wonder", "hopeful", "epic"],
    "Film-Noir": ["dark", "melancholic", "cynical"],
    "Horror": ["dark", "tense", "dread"],
    "Musical": ["light-hearted", "uplifting", "romantic"],
    "Mystery": ["tense", "curious", "atmospheric"],
    "Romance": ["warm", "hopeful", "melancholic", "romantic"],
    "Sci-Fi": ["contemplative", "awe", "melancholic", "hopeful"],
    "Thriller": ["tense", "suspenseful", "dark"],
    "War": ["dark", "intense", "sober", "hopeful"],
    "Western": ["contemplative", "gritty", "hopeful"],
}

GENRE_TO_NARRATIVE_STYLE = {
    "Action": ["action-driven", "fast-paced", "plot-driven"],
    "Adventure": ["quest-driven", "epic", "character-driven"],
    "Animation": ["character-driven", "visual storytelling"],
    "Children": ["simple narrative", "character-driven"],
    "Comedy": ["character-driven", "situation-driven"],
    "Crime": ["plot-driven", "twist-heavy", "character-driven"],
    "Documentary": ["expository", "narrative", "investigative"],
    "Drama": ["character-driven", "slow-burn", "dialogue-heavy"],
    "Fantasy": ["epic", "world-building", "quest-driven"],
    "Film-Noir": ["slow-burn", "voiceover", "moral ambiguity"],
    "Horror": ["atmosphere-driven", "slow-burn", "tension-building"],
    "Musical": ["episodic", "character-driven", "performance-driven"],
    "Mystery": ["plot-driven", "puzzle", "revelation"],
    "Romance": ["character-driven", "emotional", "relationship-focused"],
    "Sci-Fi": ["concept-driven", "slow-burn", "world-building", "idea-driven"],
    "Thriller": ["plot-driven", "fast-paced", "tension-building"],
    "War": ["ensemble", "character-driven", "intense"],
    "Western": ["slow-burn", "character-driven", "moral conflict"],
}

GENRE_TO_CINEMATOGRAPHY = {
    "Action": ["visually dynamic", "stylized", "high energy"],
    "Adventure": ["sweeping", "visually rich", "epic scale"],
    "Animation": ["visually rich", "stylized", "colorful"],
    "Children": ["bright", "clear", "friendly"],
    "Comedy": ["natural", "bright", "clear"],
    "Crime": ["gritty", "low-light", "atmospheric"],
    "Documentary": ["naturalistic", "minimalistic", "observational"],
    "Drama": ["naturalistic", "atmospheric", "restrained"],
    "Fantasy": ["visually rich", "atmospheric", "stylized"],
    "Film-Noir": ["high contrast", "shadowy", "atmospheric"],
    "Horror": ["atmospheric", "shadowy", "claustrophobic"],
    "Musical": ["visually rich", "stylized", "colorful"],
    "Mystery": ["atmospheric", "moody", "restrained"],
    "Romance": ["warm", "soft", "atmospheric"],
    "Sci-Fi": ["atmospheric", "visually rich", "minimalistic", "futuristic"],
    "Thriller": ["tense", "restrained", "atmospheric"],
    "War": ["gritty", "documentary-style", "intense"],
    "Western": ["sweeping", "minimalistic", "atmospheric"],
}

GENRE_TO_PACING = {
    "Action": "fast",
    "Adventure": "medium",
    "Animation": "medium",
    "Children": "medium",
    "Comedy": "medium",
    "Crime": "medium",
    "Documentary": "slow",
    "Drama": "slow",
    "Fantasy": "medium",
    "Film-Noir": "slow",
    "Horror": "slow",
    "Musical": "medium",
    "Mystery": "slow",
    "Romance": "slow",
    "Sci-Fi": "slow",
    "Thriller": "fast",
    "War": "medium",
    "Western": "slow",
}

GENRE_TO_TARGET_AUDIENCE = {
    "Children": "family",
    "Animation": "family",
    "Horror": "adult",
    "Crime": "adult",
    "Film-Noir": "adult",
    "War": "adult",
    "Documentary": "general",
}
# Default when not in map
DEFAULT_TARGET = "general"


def _genres_set(genres_str: str) -> Set[str]:
    """Parse pipe-separated genres into a set."""
    if pd.isna(genres_str) or not str(genres_str).strip():
        return set()
    return set(g.strip() for g in str(genres_str).split("|") if g.strip())


def _ensure_multi_label(genres_str: str) -> str:
    """If only one genre, we cannot add real second genre from data; return as-is (caller may add Unknown)."""
    s = _genres_set(genres_str)
    if len(s) < 2 and s:
        return genres_str  # Leave single; preprocessing can add "General" or keep
    return genres_str


def derive_themes(genres_str: str) -> List[str]:
    """Derive theme keywords from genre set. Multiple genres -> combined themes."""
    gs = _genres_set(genres_str)
    themes = []
    seen = set()
    for g in gs:
        for t in GENRE_TO_THEMES.get(g, []):
            if t not in seen:
                seen.add(t)
                themes.append(t)
    return themes[:8]  # Cap for diversity


def derive_emotional_tone(genres_str: str) -> List[str]:
    """Derive emotional tone from genres."""
    gs = _genres_set(genres_str)
    tones = []
    seen = set()
    for g in gs:
        for t in GENRE_TO_EMOTIONAL_TONE.get(g, []):
            if t not in seen:
                seen.add(t)
                tones.append(t)
    return tones[:5]


def derive_narrative_style(genres_str: str) -> List[str]:
    """Derive narrative style tags from genres."""
    gs = _genres_set(genres_str)
    styles = []
    seen = set()
    for g in gs:
        for s in GENRE_TO_NARRATIVE_STYLE.get(g, []):
            if s not in seen:
                seen.add(s)
                styles.append(s)
    return styles[:5]


def derive_cinematography(genres_str: str) -> List[str]:
    """Derive cinematography/atmosphere tags from genres."""
    gs = _genres_set(genres_str)
    tags = []
    seen = set()
    for g in gs:
        for c in GENRE_TO_CINEMATOGRAPHY.get(g, []):
            if c not in seen:
                seen.add(c)
                tags.append(c)
    return tags[:5]


def derive_pacing(genres_str: str) -> str:
    """Single pacing label: prefer slow/medium from dominant genre."""
    gs = list(_genres_set(genres_str))
    if not gs:
        return "medium"
    # Take first genre as primary for pacing
    return GENRE_TO_PACING.get(gs[0], "medium")


def derive_target_audience(genres_str: str) -> str:
    """Single target audience; if any genre is family/adult, use that."""
    gs = _genres_set(genres_str)
    for g in gs:
        if g in GENRE_TO_TARGET_AUDIENCE:
            return GENRE_TO_TARGET_AUDIENCE[g]
    return DEFAULT_TARGET


def build_semantic_text(row: pd.Series) -> str:
    """
    Build a single semantic text blob for TF-IDF: genres + themes + emotional_tone
    + narrative_style + cinematography + pacing + target_audience.
    """
    parts = []

    genres_str = row.get("genres", "")
    if pd.notna(genres_str) and genres_str:
        parts.append(str(genres_str).replace("|", " ").lower())

    themes = row.get("themes")
    if isinstance(themes, str) and themes:
        parts.append(themes.replace("|", " ").lower())
    elif isinstance(themes, list) and themes:
        parts.append(" ".join(t.lower() for t in themes))

    tone = row.get("emotional_tone")
    if isinstance(tone, str) and tone:
        parts.append(tone.replace("|", " ").lower())
    elif isinstance(tone, list) and tone:
        parts.append(" ".join(t.lower() for t in tone))

    narrative = row.get("narrative_style")
    if isinstance(narrative, str) and narrative:
        parts.append(narrative.replace("|", " ").lower())
    elif isinstance(narrative, list) and narrative:
        parts.append(" ".join(n.lower() for n in narrative))

    cine = row.get("cinematography_style")
    if isinstance(cine, str) and cine:
        parts.append(cine.replace("|", " ").lower())
    elif isinstance(cine, list) and cine:
        parts.append(" ".join(c.lower() for c in cine))

    pacing_val = row.get("pacing", "")
    if pacing_val:
        parts.append(str(pacing_val).lower())

    audience = row.get("target_audience", "")
    if audience:
        parts.append(str(audience).lower())

    title = row.get("title", "")
    if pd.notna(title) and title:
        t = re.sub(r"\s*\(\d{4}\)\s*", " ", str(title))
        t = re.sub(r"[^\w\s]", " ", t).lower()
        parts.append(t)

    plot_kw = row.get("plot_keywords", "")
    if pd.notna(plot_kw) and str(plot_kw).strip():
        parts.append(str(plot_kw).replace(",", " ").lower())

    # Repeat theme, mood, and cinematography so TF-IDF similarity is driven by them
    if themes := row.get("themes"):
        if isinstance(themes, str) and themes:
            parts.append(themes.replace("|", " ").lower())
        elif isinstance(themes, list) and themes:
            parts.append(" ".join(t.lower() for t in themes))
    if tone := row.get("emotional_tone"):
        if isinstance(tone, str) and tone:
            parts.append(tone.replace("|", " ").lower())
        elif isinstance(tone, list) and tone:
            parts.append(" ".join(t.lower() for t in tone))
    if cine := row.get("cinematography_style"):
        if isinstance(cine, str) and cine:
            parts.append(cine.replace("|", " ").lower())
        elif isinstance(cine, list) and cine:
            parts.append(" ".join(c.lower() for c in cine))

    return " ".join(p for p in parts if p).strip()


def enrich_movies_semantic(movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add semantic columns to movies dataframe. Ensures multi-label identity
    by deriving themes, emotional_tone, narrative_style, cinematography_style,
    pacing, target_audience, and semantic_text. Does not remove single-genre
    movies but enriches them with derived attributes.
    """
    df = movies_df.copy()

    # Ensure genres is string and at least one genre
    if "genres" not in df.columns:
        df["genres"] = "Drama"
    df["genres"] = df["genres"].fillna("Unknown").astype(str)
    # Optional: ensure 2+ genres by appending "Drama" when only one
    def ensure_two_genres(s: str) -> str:
        gs = [x.strip() for x in s.split("|") if x.strip()]
        if len(gs) < 2 and gs and gs[0].lower() != "unknown":
            return s + "|Drama"
        return s
    df["genres"] = df["genres"].apply(ensure_two_genres)

    themes_list = []
    tone_list = []
    narrative_list = []
    cine_list = []
    pacing_list = []
    audience_list = []

    for _, row in df.iterrows():
        g = row["genres"]
        themes_list.append("|".join(derive_themes(g)))
        tone_list.append("|".join(derive_emotional_tone(g)))
        narrative_list.append("|".join(derive_narrative_style(g)))
        cine_list.append("|".join(derive_cinematography(g)))
        pacing_list.append(derive_pacing(g))
        audience_list.append(derive_target_audience(g))

    df["themes"] = themes_list
    df["emotional_tone"] = tone_list
    df["narrative_style"] = narrative_list
    df["cinematography_style"] = cine_list
    df["pacing"] = pacing_list
    df["target_audience"] = audience_list

    semantic_texts = []
    for idx, row in df.iterrows():
        semantic_texts.append(build_semantic_text(row))
    df["semantic_text"] = semantic_texts

    return df
