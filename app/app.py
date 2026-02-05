"""
Netflix-like Recommendation System — Streamlit UI.
UX refactor: 3-zone layout, step-by-step flow, results only after action.
Dark neumorphic visual style. No logic changes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import streamlit as st
import pandas as pd
import numpy as np

from data_loader import load_all_data
from preprocessing import preprocess_movies, preprocess_ratings, get_user_features
from content_based import ContentBasedRecommender
from collaborative import CollaborativeFilteringRecommender
from kmeans_model import UserSegmentation
from popularity import PopularityRecommender

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# CSS — Dark neumorphism, 3-zone hierarchy, Netflix accent
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Base: dark charcoal, soft depth */
    .stApp { background: linear-gradient(165deg, #1a1c1e 0%, #0f1113 50%, #0d0d0d 100%); min-height: 100vh; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #16181a 0%, #121416 100%); }
    [data-testid="stSidebar"] .stMarkdown { color: #b0b0b0; }

    /* Zone: Context strip (main top) */
    .context-strip {
        background: rgba(22,24,26,0.85);
        border-radius: 16px;
        padding: 1rem 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 8px 32px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.03);
    }
    .context-title { font-size: 1.75rem; font-weight: 700; color: #E50914; margin: 0 0 0.25rem 0; }
    .context-subtitle { color: #808080; font-size: 0.9rem; margin: 0; }
    .context-mode {
        display: inline-block;
        margin-top: 0.5rem;
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        background: rgba(229,9,20,0.15);
        color: #e8a0a0;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Neumorphic step card */
    .step-card {
        background: linear-gradient(145deg, #1c1e21 0%, #16181b 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 10px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04);
    }
    .step-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; color: #E50914; margin-bottom: 0.5rem; font-weight: 600; }
    .step-title { font-size: 1.1rem; font-weight: 600; color: #fff; margin-bottom: 1rem; }

    /* Movie profile inside step card */
    .profile-inline {
        background: rgba(0,0,0,0.25);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-top: 1rem;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .profile-label { color: #707070; font-size: 0.7rem; text-transform: uppercase; margin-bottom: 0.2rem; }
    .profile-value { color: #d8d8d8; font-size: 0.85rem; }

    /* Primary CTA: Netflix red, raised */
    .cta-wrapper { margin: 1.5rem 0 2rem 0; }
    .stButton > button[kind="primary"] {
        width: 100%;
        padding: 0.85rem 1.5rem;
        font-size: 1.05rem;
        font-weight: 600;
        background: linear-gradient(180deg, #E50914 0%, #b80710 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 12px !important;
        box-shadow: 0 6px 24px rgba(229,9,20,0.4), 0 2px 8px rgba(0,0,0,0.3) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(229,9,20,0.5), 0 4px 12px rgba(0,0,0,0.35) !important;
    }
    .stButton > button[kind="primary"]:active {
        transform: translateY(0) !important;
    }

    /* Results zone — visually dominant after click */
    .results-zone {
        background: linear-gradient(145deg, #181a1d 0%, #141619 100%);
        border-radius: 16px;
        padding: 1.75rem;
        margin-top: 2rem;
        border: 1px solid rgba(255,255,255,0.07);
        box-shadow: 0 12px 48px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.03);
    }
    .results-header {
        font-size: 1.35rem;
        font-weight: 600;
        color: #fff;
        margin-bottom: 1.25rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    /* Netflix-style movie card in grid */
    .movie-card {
        background: linear-gradient(145deg, #1a1c1f 0%, #15171a 100%);
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .movie-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 28px rgba(0,0,0,0.4);
        border-color: rgba(229,9,20,0.35);
    }
    .movie-title { font-size: 1rem; font-weight: 600; color: #fff; margin-bottom: 0.35rem; }
    .movie-genres { color: #a0a0a0; font-size: 0.82rem; margin-bottom: 0.3rem; }
    .movie-meta { color: #707070; font-size: 0.78rem; }
    .explanation-box {
        background: rgba(0,0,0,0.3);
        border-left: 3px solid #E50914;
        padding: 0.5rem 0.75rem;
        margin-top: 0.5rem;
        border-radius: 6px;
        font-size: 0.82rem;
        color: #b8b8b8;
    }
    .tag-row { margin-top: 0.3rem; }
    .semantic-tag {
        display: inline-block;
        background: rgba(255,255,255,0.06);
        color: #a8a8a8;
        padding: 0.18rem 0.45rem;
        border-radius: 6px;
        font-size: 0.72rem;
        margin-right: 0.2rem;
        margin-bottom: 0.2rem;
    }
    .semantic-tag.tema { border-left: 3px solid #E50914; }
    .semantic-tag.duygu { border-left: 3px solid #564dff; }
    .semantic-tag.goruntu { border-left: 3px solid #46d160; }

    /* Sidebar: compact sections */
    .sidebar-block { margin-bottom: 1.25rem; }
    .sidebar-title { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; color: #606060; margin-bottom: 0.5rem; }
    div[data-testid="stMetric"] { background: rgba(255,255,255,0.03); border-radius: 8px; padding: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Data and models (cached)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    movies_df, ratings_df, users_df = load_all_data()
    ratings_df = preprocess_ratings(ratings_df, min_ratings_per_user=5)
    movies_df = preprocess_movies(movies_df, ratings_df)
    return movies_df, ratings_df, users_df

@st.cache_resource
def load_models(movies_df, ratings_df):
    content_model = ContentBasedRecommender(
        movies_df, similarity_threshold=0.15, max_same_genre_combo=2
    )
    collaborative_model = CollaborativeFilteringRecommender(ratings_df)
    popularity_model = PopularityRecommender(ratings_df, movies_df)
    user_features = get_user_features(ratings_df, movies_df)
    segmentation_model = UserSegmentation(user_features, n_clusters=5)
    return content_model, collaborative_model, popularity_model, segmentation_model

with st.spinner("Veriler yükleniyor…"):
    movies_df, ratings_df, users_df = load_data()
    content_model, collaborative_model, popularity_model, segmentation_model = load_models(movies_df, ratings_df)

# -----------------------------------------------------------------------------
# Session state: results ONLY after action; clear when method changes
# -----------------------------------------------------------------------------
if "last_method" not in st.session_state:
    st.session_state.last_method = None
if "recommendations_ready" not in st.session_state:
    st.session_state.recommendations_ready = False
if "recommendations_result" not in st.session_state:
    st.session_state.recommendations_result = None
if "recommendations_context" not in st.session_state:
    st.session_state.recommendations_context = {}

method_options = [
    "Content-Based (tema & duygu)",
    "Collaborative (benzer kullanıcılar)",
    "User Segmentation (küme)",
    "Popüler filmler",
]
method = st.sidebar.selectbox(
    "Öneri yöntemi",
    method_options,
    index=0,
    key="method_select",
    label_visibility="visible",
)
if st.session_state.last_method != method:
    st.session_state.last_method = method
    st.session_state.recommendations_ready = False
    st.session_state.recommendations_result = None
    st.session_state.recommendations_context = {}

show_why = st.sidebar.checkbox(
    "Neden önerildi? (açıklama göster)",
    value=False,
    key="show_why",
    help="Önerilen her filmde ortak tema, duygu ve benzerlik gösterilir.",
)
n_recs = st.sidebar.selectbox(
    "Öneri sayısı",
    options=[5, 10, 15, 20],
    index=1,
    key="n_recs_global",
)
st.sidebar.markdown("---")
st.sidebar.markdown('<p class="sidebar-title">Veri seti</p>', unsafe_allow_html=True)
st.sidebar.metric("Film", len(movies_df))
st.sidebar.metric("Kullanıcı", len(ratings_df["userId"].unique()))
st.sidebar.metric("Puan", len(ratings_df))

# -----------------------------------------------------------------------------
# Helpers (unchanged logic)
# -----------------------------------------------------------------------------
def _format_semantic_tags(row: pd.Series) -> str:
    parts = []
    for label, col, cls in (
        ("Tema", "themes", "tema"),
        ("Duygu", "emotional_tone", "duygu"),
        ("Görüntü", "cinematography_style", "goruntu"),
    ):
        if col not in row or pd.isna(row.get(col)) or not str(row[col]).strip():
            continue
        vals = str(row[col]).replace("|", ",").split(",")
        vals = [v.strip() for v in vals if v.strip()][:3]
        if vals:
            tags = " ".join(f'<span class="semantic-tag {cls}">{v}</span>' for v in vals)
            parts.append(f'<div class="tag-row"><span class="profile-label">{label}</span> {tags}</div>')
    return "".join(parts) if parts else ""

def _movie_profile_html(row: pd.Series) -> str:
    year_val = row.get("year")
    year_str = str(int(year_val)) if pd.notna(year_val) else "—"
    def _str(v):
        if pd.isna(v) or v is None:
            return "—"
        s = str(v).strip()
        return s.replace("|", ", ") if s else "—"
    tema = _str(row.get("themes"))
    duygu = _str(row.get("emotional_tone"))
    cine = _str(row.get("cinematography_style"))
    kw = _str(row.get("plot_keywords"))
    return (
        f'<div class="profile-inline">'
        f'<div class="profile-label">Yıl</div><div class="profile-value">{year_str}</div>'
        f'<div class="profile-label" style="margin-top:0.4rem;">Tema</div><div class="profile-value">{tema}</div>'
        f'<div class="profile-label" style="margin-top:0.4rem;">Duygu</div><div class="profile-value">{duygu}</div>'
        f'<div class="profile-label" style="margin-top:0.4rem;">Sinematografi</div><div class="profile-value">{cine}</div>'
        f'<div class="profile-label" style="margin-top:0.4rem;">Anahtar kelimeler</div><div class="profile-value">{kw}</div>'
        f'</div>'
    )

def render_movie_cards(recs: pd.DataFrame, show_explanations: bool, movies_df: pd.DataFrame = None, max_cards: int = 12):
    if recs is None or len(recs) == 0:
        st.warning("Öneri bulunamadı.")
        return
    recs = recs.head(max_cards).copy()
    profile_cols = ["year", "themes", "emotional_tone", "cinematography_style", "plot_keywords"]
    if movies_df is not None and "movieId" in recs.columns:
        add_cols = [c for c in profile_cols if c in movies_df.columns and c not in recs.columns]
        if add_cols:
            merge_df = movies_df[["movieId"] + add_cols].drop_duplicates(subset=["movieId"])
            recs = recs.merge(merge_df, on="movieId", how="left")
    n_cols = 3
    for start in range(0, len(recs), n_cols):
        cols = st.columns(n_cols)
        for i, col in enumerate(cols):
            idx = start + i
            if idx >= len(recs):
                break
            row = recs.iloc[idx]
            with col:
                title = row["title"]
                genres = (row["genres"] or "").replace("|", " · ")
                meta = []
                if "year" in row and pd.notna(row.get("year")):
                    meta.append(str(int(row["year"])))
                if "similarity_score" in row:
                    meta.append(f"Benzerlik: {row['similarity_score']:.2f}")
                if "predicted_rating" in row:
                    meta.append(f"Tahmin: {row['predicted_rating']:.2f}/5")
                meta_str = " · ".join(meta)
                tags_html = _format_semantic_tags(row)
                expander_label = f"🎬 {title}"
                if meta_str:
                    expander_label += f" — {meta_str}"
                with st.expander(expander_label, expanded=False):
                    tags_block = f'<div style="margin-top:0.35rem;">{tags_html}</div>' if tags_html else ""
                    st.markdown(
                        f'<div class="movie-card"><div class="movie-genres">{genres}</div>'
                        f'<div class="movie-meta">{meta_str}</div>{tags_block}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown("**Film profili**")
                    st.markdown(_movie_profile_html(row), unsafe_allow_html=True)
                    if show_explanations and "explanation" in row and pd.notna(row.get("explanation")):
                        st.markdown("**Neden önerildi?**")
                        st.markdown(
                            f'<div class="explanation-box">{row["explanation"]}</div>',
                            unsafe_allow_html=True,
                        )

# -----------------------------------------------------------------------------
# MAIN: Zone 1 — Context (top)
# -----------------------------------------------------------------------------
st.markdown(
    '<div class="context-strip">'
    '<h1 class="context-title">🎬 Film Öneri Sistemi</h1>'
    '<p class="context-subtitle">Tema, duygu ve sinematografiye göre benzer filmler · MovieLens verisi (simülasyon)</p>'
    f'<span class="context-mode">{method}</span>'
    '</div>',
    unsafe_allow_html=True,
)
with st.expander("Veri ve proje hakkında", expanded=False):
    st.caption("Bu uygulama **MovieLens** açık veri setini kullanır; Netflix verisi değildir. Öneriler tema, duygu ve türe göre benzerlik ile üretilir.")

# -----------------------------------------------------------------------------
# MAIN: Zone 2 — Step-by-step flow (one card + CTA)
# -----------------------------------------------------------------------------
step_card = st.container()
with step_card:
    if method == "Content-Based (tema & duygu)":
        st.markdown(
            '<div class="step-card"><p class="step-label">Adım 1</p><p class="step-title">Film seçin</p></div>',
            unsafe_allow_html=True,
        )
        movie_titles = sorted(movies_df["title"].unique().tolist())
        chosen_title = st.selectbox(
            "Film",
            options=movie_titles,
            index=0,
            key="content_movie",
            label_visibility="collapsed",
        )
        selected_movie_id = movies_df[movies_df["title"] == chosen_title]["movieId"].iloc[0]
        movie_info = movies_df[movies_df["movieId"] == selected_movie_id].iloc[0]
        st.markdown(_movie_profile_html(movie_info), unsafe_allow_html=True)
        st.markdown('<div class="cta-wrapper"></div>', unsafe_allow_html=True)
        if st.button("Önerileri getir", type="primary", key="content_btn"):
            with st.spinner("Benzer filmler aranıyor…"):
                recs = content_model.get_similar_movies(
                    selected_movie_id, n_recommendations=n_recs, include_explanation=show_why
                )
            if len(recs) > 0:
                st.session_state.recommendations_ready = True
                st.session_state.recommendations_result = recs
                st.session_state.recommendations_context = {
                    "header": f'"{chosen_title}" ile benzer tema, duygu ve sinematografiye sahip filmler',
                    "show_why": show_why,
                }
            else:
                st.session_state.recommendations_ready = False
                st.session_state.recommendations_result = None
                st.warning("Bu film için benzer bulunamadı. Başka bir film deneyin.")
            st.rerun()

    elif method == "Collaborative (benzer kullanıcılar)":
        st.markdown(
            '<div class="step-card"><p class="step-label">Adım 1</p><p class="step-title">Kullanıcı seçin</p></div>',
            unsafe_allow_html=True,
        )
        user_ids = sorted(ratings_df["userId"].unique().tolist())
        selected_user = st.selectbox(
            "Kullanıcı",
            options=user_ids,
            index=0,
            key="collab_user",
            format_func=lambda x: f"Kullanıcı {x}",
            label_visibility="collapsed",
        )
        user_ratings = collaborative_model.get_user_ratings(selected_user)
        st.markdown('<div class="cta-wrapper"></div>', unsafe_allow_html=True)
        if st.button("Önerileri getir", type="primary", key="collab_btn"):
            with st.spinner("Öneriler hesaplanıyor…"):
                if len(user_ratings) > 0:
                    recs = collaborative_model.recommend_for_user(
                        selected_user, movies_df, n_recs, include_explanation=show_why
                    )
                else:
                    recs = popularity_model.recommend(n_recs)
            st.session_state.recommendations_ready = True
            st.session_state.recommendations_result = recs
            st.session_state.recommendations_context = {
                "header": "Sizin için öneriler",
                "show_why": show_why,
            }
            if len(user_ratings) == 0:
                st.session_state.recommendations_context["info"] = "Bu kullanıcının puanı yok; popüler filmler gösteriliyor."
            st.rerun()

    elif method == "User Segmentation (küme)":
        st.markdown(
            '<div class="step-card"><p class="step-label">Adım 1</p><p class="step-title">Kullanıcı seçin</p></div>',
            unsafe_allow_html=True,
        )
        user_ids = sorted(ratings_df["userId"].unique().tolist())
        selected_user = st.selectbox(
            "Kullanıcı",
            options=user_ids,
            index=0,
            key="seg_user",
            format_func=lambda x: f"Kullanıcı {x}",
            label_visibility="collapsed",
        )
        try:
            user_cluster = segmentation_model.get_user_cluster(selected_user)
            cstats = segmentation_model.get_cluster_stats()
            crow = cstats[cstats["cluster"] == user_cluster].iloc[0]
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Küme", int(user_cluster))
            with c2:
                st.metric("Kümedeki kullanıcı", int(crow["n_users"]))
            with c3:
                st.metric("Ort. puan", f"{crow['avg_rating_mean']:.2f}")
        except ValueError:
            pass
        st.markdown('<div class="cta-wrapper"></div>', unsafe_allow_html=True)
        if st.button("Küme önerilerini getir", type="primary", key="seg_btn"):
            try:
                with st.spinner("Yükleniyor…"):
                    user_cluster = segmentation_model.get_user_cluster(selected_user)
                    recs = segmentation_model.recommend_for_cluster(
                        user_cluster, ratings_df, movies_df, n_recs
                    )
                st.session_state.recommendations_ready = True
                st.session_state.recommendations_result = recs
                st.session_state.recommendations_context = {
                    "header": "Kümenizde popüler",
                    "show_why": show_why,
                }
                st.rerun()
            except ValueError as e:
                st.session_state.recommendations_ready = False
                st.session_state.recommendations_result = None
                st.error(str(e))

    else:
        st.markdown(
            '<div class="step-card"><p class="step-label">Adım 1</p><p class="step-title">Popüler filmler</p></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<p class="profile-value">En çok puan alan filmler listelenecek.</p>', unsafe_allow_html=True)
        st.markdown('<div class="cta-wrapper"></div>', unsafe_allow_html=True)
        if st.button("Popüler filmleri göster", type="primary", key="pop_btn"):
            recs = popularity_model.recommend(n_recs)
            st.session_state.recommendations_ready = True
            st.session_state.recommendations_result = recs
            st.session_state.recommendations_context = {
                "header": "Trend filmler",
                "show_why": False,
            }
            st.rerun()

# -----------------------------------------------------------------------------
# MAIN: Zone 3 — Results (ONLY after action)
# -----------------------------------------------------------------------------
if st.session_state.recommendations_ready and st.session_state.recommendations_result is not None:
    recs = st.session_state.recommendations_result
    ctx = st.session_state.recommendations_context
    st.markdown('<div class="results-zone">', unsafe_allow_html=True)
    st.markdown(f'<div class="results-header">{ctx.get("header", "Öneriler")}</div>', unsafe_allow_html=True)
    if ctx.get("info"):
        st.info(ctx["info"])
    render_movie_cards(recs, ctx.get("show_why", False), movies_df=movies_df)
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Footer (minimal)
# -----------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center;color:#505050;font-size:0.8rem;">'
    'Tema, duygu ve stile göre öneriler · MovieLens · Netflix ile bağlantılı değildir</p>',
    unsafe_allow_html=True,
)
