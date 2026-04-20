"""
Netflix Content Clustering  ·  KMeans + PCA
============================================
Portfolio project – unsupervised machine-learning pipeline applied to the
Netflix catalogue.  Run with:  streamlit run app.py
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import os
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Netflix Clustering",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens & global CSS ────────────────────────────────────────────────
PALETTE = {
    "bg":       "#0A0A0F",
    "surface":  "#12121A",
    "card":     "#1A1A26",
    "border":   "#2A2A3A",
    "red":      "#E50914",
    "red_dim":  "#8B0000",
    "text":     "#F0F0F5",
    "muted":    "#7A7A9A",
    "accent":   "#FF6B35",
    "teal":     "#00B4D8",
}

CLUSTER_COLORS = ["#E50914", "#00B4D8", "#FF6B35", "#9B5DE5",
                  "#06D6A0", "#FFD60A", "#F72585", "#4CC9F0",
                  "#7209B7", "#3A0CA3"]

st.markdown(f"""
<style>
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=Manrope:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {{
    background-color: {PALETTE['bg']};
    color: {PALETTE['text']};
    font-family: 'Manrope', sans-serif;
}}
.stApp {{ background-color: {PALETTE['bg']}; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background-color: {PALETTE['surface']};
    border-right: 1px solid {PALETTE['border']};
}}
[data-testid="stSidebar"] * {{ color: {PALETTE['text']} !important; }}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background-color: transparent;
    border-bottom: 1px solid {PALETTE['border']};
    gap: 0;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    color: {PALETTE['muted']};
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.6rem 1.4rem;
    border-bottom: 2px solid transparent;
}}
.stTabs [aria-selected="true"] {{
    color: {PALETTE['red']} !important;
    border-bottom: 2px solid {PALETTE['red']};
    background: transparent !important;
}}

/* ── Metrics ── */
[data-testid="stMetric"] {{
    background-color: {PALETTE['card']};
    border: 1px solid {PALETTE['border']};
    border-radius: 8px;
    padding: 1rem 1.2rem;
}}
[data-testid="stMetricLabel"] {{ color: {PALETTE['muted']} !important; font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; font-family: 'DM Mono', monospace; }}
[data-testid="stMetricValue"] {{ color: {PALETTE['text']}; font-family: 'DM Serif Display', serif; font-size: 2rem; }}

/* ── DataFrames ── */
.stDataFrame {{ border: 1px solid {PALETTE['border']}; border-radius: 8px; overflow: hidden; }}
[data-testid="stDataFrameResizable"] {{ background-color: {PALETTE['card']}; }}

/* ── Expanders ── */
[data-testid="stExpander"] {{
    background-color: {PALETTE['card']};
    border: 1px solid {PALETTE['border']};
    border-radius: 8px;
}}

/* ── Info / alerts ── */
.stAlert {{ background-color: {PALETTE['card']}; border-color: {PALETTE['border']}; color: {PALETTE['text']}; border-radius: 8px; }}

/* ── Divider ── */
hr {{ border-color: {PALETTE['border']}; }}

/* ── Hero ── */
.hero {{
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid {PALETTE['border']};
    margin-bottom: 2rem;
}}
.hero-title {{
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2.2rem, 5vw, 3.6rem);
    line-height: 1.05;
    color: {PALETTE['text']};
    margin: 0 0 0.5rem;
}}
.hero-title span {{ color: {PALETTE['red']}; }}
.hero-sub {{
    font-size: 0.85rem;
    color: {PALETTE['muted']};
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.06em;
}}
.tag {{
    display: inline-block;
    background: {PALETTE['card']};
    border: 1px solid {PALETTE['border']};
    color: {PALETTE['muted']};
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.25rem 0.65rem;
    border-radius: 4px;
    margin-right: 0.4rem;
}}

/* ── Cluster badge ── */
.cluster-header {{
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid {PALETTE['border']};
    margin-bottom: 1rem;
}}

/* ── Section label ── */
.section-label {{
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {PALETTE['red']};
    margin-bottom: 0.4rem;
}}
</style>
""", unsafe_allow_html=True)

# ── Matplotlib theme ───────────────────────────────────────────────────────────
def apply_chart_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["surface"],
        "axes.edgecolor":    PALETTE["border"],
        "axes.labelcolor":   PALETTE["muted"],
        "axes.titlecolor":   PALETTE["text"],
        "xtick.color":       PALETTE["muted"],
        "ytick.color":       PALETTE["muted"],
        "text.color":        PALETTE["text"],
        "grid.color":        PALETTE["border"],
        "grid.linewidth":    0.6,
        "axes.grid":         True,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.family":       "monospace",
        "axes.titlesize":    11,
        "axes.labelsize":    9,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
    })

apply_chart_style()

# ── Constants ─────────────────────────────────────────────────────────────────
RATINGS = ["TV-MA", "TV-14", "TV-PG", "R", "PG-13", "TV-Y7", "TV-Y",
           "PG", "TV-G", "NR", "G", "NC-17"]

GENRE_POOL = [
    "International Movies", "Dramas", "Comedies", "Action & Adventure",
    "Documentaries", "Children & Family Movies", "Romantic Movies",
    "Horror Movies", "Thrillers", "Stand-Up Comedy", "Sci-Fi & Fantasy",
    "Crime TV Shows", "TV Dramas", "Reality TV", "Kids' TV",
    "Anime Series", "Classic Movies", "Music & Musicals",
    "Independent Movies", "British TV Shows",
]

COUNTRY_POOL = [
    "United States", "India", "United Kingdom", "Canada", "France",
    "Japan", "South Korea", "Spain", "Mexico", "Australia",
    "Germany", "Nigeria", "Brazil", "Turkey", "Egypt",
]

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    csv_path = os.path.join(os.path.dirname(__file__), "data", "netflix_titles.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if {"type", "title", "rating", "listed_in", "release_year"}.issubset(df.columns):
            return df

    try:
        import kagglehub
        path = kagglehub.dataset_download("shivamb/netflix-shows")
        kaggle_csv = os.path.join(path, "netflix_titles.csv")
        if os.path.exists(kaggle_csv):
            return pd.read_csv(kaggle_csv)
    except Exception:
        pass

    rng = np.random.RandomState(42)
    n = 2000
    types = rng.choice(["Movie", "TV Show"], size=n, p=[0.70, 0.30])
    release_years = rng.normal(2015, 6, n).clip(1970, 2023).astype(int)
    rating_weights = np.array([0.22, 0.21, 0.12, 0.10, 0.09, 0.06,
                               0.05, 0.05, 0.04, 0.03, 0.02, 0.01])
    ratings = rng.choice(RATINGS, size=n, p=rating_weights)
    genres_list = [
        ", ".join(rng.choice(GENRE_POOL, rng.choice([1,2,3], p=[0.3,0.5,0.2]), replace=False))
        for _ in range(n)
    ]
    durations = []
    for t in types:
        if t == "Movie":
            durations.append(f"{int(np.clip(rng.normal(100,25),30,240))} min")
        else:
            s = int(rng.choice([1,2,3,4,5], p=[0.45,0.25,0.15,0.10,0.05]))
            durations.append(f"{s} Season{'s' if s!=1 else ''}")
    fn = ["Alex","Jordan","Sam","Chris","Pat","Morgan","Taylor","Casey","Riley","Jamie"]
    ln = ["Smith","Kim","Garcia","Chen","Patel","Müller","Tanaka","Silva","Johansson","Ali"]
    countries = [rng.choice(COUNTRY_POOL) for _ in range(n)]
    directors = [f"{rng.choice(fn)} {rng.choice(ln)}" for _ in range(n)]
    cast_list = [", ".join(f"{rng.choice(fn)} {rng.choice(ln)}" for _ in range(rng.randint(2,6))) for _ in range(n)]

    return pd.DataFrame({
        "show_id": [f"s{i+1}" for i in range(n)],
        "type": types, "title": [f"Title {i+1}" for i in range(n)],
        "director": directors, "cast": cast_list, "country": countries,
        "release_year": release_years, "rating": ratings,
        "duration": durations, "listed_in": genres_list,
    })


# ── Feature engineering ───────────────────────────────────────────────────────
@st.cache_data
def engineer_features(df: pd.DataFrame) -> tuple:
    clean = df.copy()
    le = LabelEncoder()
    clean["rating_encoded"] = le.fit_transform(clean["rating"].fillna("NR"))
    top_genres = (
        clean["listed_in"].fillna("").str.split(", ").explode()
        .value_counts().head(12).index.tolist()
    )
    for g in top_genres:
        clean[f"genre_{g}"] = clean["listed_in"].fillna("").str.contains(g, regex=False).astype(int)
    genre_cols = [c for c in clean.columns if c.startswith("genre_")]
    clean["duration_num"] = (
        clean["duration"].fillna("0").str.extract(r"(\d+)", expand=False).astype(float).fillna(0)
    )
    clean["is_movie"] = (clean["type"] == "Movie").astype(int)
    feature_cols = ["release_year", "rating_encoded", "duration_num", "is_movie"] + genre_cols
    return clean, clean[feature_cols].copy(), genre_cols


@st.cache_data
def scale_features(_encoded_df: pd.DataFrame) -> tuple:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(_encoded_df)
    return scaled, _encoded_df.columns.tolist()


# ── Clustering ────────────────────────────────────────────────────────────────
@st.cache_data
def compute_metrics(_scaled: np.ndarray, k_range=(2, 11)) -> pd.DataFrame:
    rows = []
    for k in range(k_range[0], k_range[1]):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(_scaled)
        rows.append({"K": k, "Inertia": km.inertia_,
                     "Silhouette": silhouette_score(_scaled, labels)})
    return pd.DataFrame(rows)


@st.cache_data
def run_kmeans(_scaled: np.ndarray, k: int) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(_scaled)


@st.cache_data
def run_pca(_scaled: np.ndarray) -> np.ndarray:
    return PCA(n_components=2, random_state=42).fit_transform(_scaled)


# ── Charts ────────────────────────────────────────────────────────────────────
def fig_elbow(metrics: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(metrics["K"], metrics["Inertia"], "o-", color=PALETTE["red"],
            linewidth=1.8, markersize=5, markerfacecolor=PALETTE["text"])
    ax.set_xlabel("K")
    ax.set_ylabel("Inertia (WCSS)")
    ax.set_title("Elbow Method")
    ax.set_xticks(metrics["K"])
    fig.tight_layout(pad=0.8)
    return fig


def fig_silhouette(metrics: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(metrics["K"], metrics["Silhouette"], "s-", color=PALETTE["teal"],
            linewidth=1.8, markersize=5, markerfacecolor=PALETTE["text"])
    ax.set_xlabel("K")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score")
    ax.set_xticks(metrics["K"])
    fig.tight_layout(pad=0.8)
    return fig


def fig_pca(pca_result: np.ndarray, labels: np.ndarray, k: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = mpl.colors.ListedColormap(CLUSTER_COLORS[:k])
    sc = ax.scatter(
        pca_result[:, 0], pca_result[:, 1],
        c=labels, cmap=cmap, alpha=0.55,
        edgecolors="none", s=22,
    )
    cb = fig.colorbar(sc, ax=ax, label="Cluster",
                      ticks=range(k), pad=0.02)
    cb.ax.set_yticklabels([f"C{i}" for i in range(k)])
    cb.ax.tick_params(labelsize=8, colors=PALETTE["muted"])
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(f"PCA Projection  ·  K = {k}")
    fig.tight_layout(pad=0.8)
    return fig


def fig_barh(series: pd.Series, title: str, top_n: int = 10,
             color: str = None) -> plt.Figure:
    counts = series.value_counts().head(top_n).sort_values()
    color = color or PALETTE["red"]
    fig, ax = plt.subplots(figsize=(5, 0.38 * len(counts) + 0.8))
    bars = ax.barh(counts.index, counts.values, color=color, alpha=0.85)
    ax.bar_label(bars, padding=4, fontsize=7.5, color=PALETTE["muted"])
    ax.set_title(title)
    ax.set_xlabel("Count")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout(pad=0.6)
    return fig


def fig_cluster_profile(subset: pd.DataFrame, cluster_id: int) -> plt.Figure:
    """Compact 1×2 mini-profile: rating dist + genre dist."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 2.8))
    color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]

    # Ratings
    r = subset["rating"].value_counts().head(6).sort_values()
    ax1.barh(r.index, r.values, color=color, alpha=0.8)
    ax1.set_title("Rating Mix")
    ax1.spines["left"].set_visible(False)
    ax1.tick_params(axis="y", length=0)

    # Genres
    g = (subset["listed_in"].fillna("").str.split(", ").explode()
         .value_counts().head(6).sort_values())
    ax2.barh(g.index, g.values, color=PALETTE["teal"], alpha=0.8)
    ax2.set_title("Top Genres")
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(axis="y", length=0)

    fig.tight_layout(pad=0.6)
    return fig


# ── App ───────────────────────────────────────────────────────────────────────
def main() -> None:
    # ── Load & engineer ──────────────────────────────────────────────────────
    df = load_data()
    clean_df, encoded_df, genre_cols = engineer_features(df)
    scaled, feature_names = scale_features(encoded_df)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<p class="section-label">Model Settings</p>', unsafe_allow_html=True
        )
        k = st.slider("Clusters  (K)", 2, 10, 4)

        st.markdown("---")
        st.markdown('<p class="section-label">About</p>', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="font-size:0.78rem; color:#7A7A9A; line-height:1.6;">
            Unsupervised ML pipeline on the Netflix catalogue.<br><br>
            <b style="color:#F0F0F5;">Stack:</b> scikit-learn · pandas · Streamlit · Matplotlib<br>
            <b style="color:#F0F0F5;">Methods:</b> KMeans · PCA · StandardScaler
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Hero ─────────────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="hero">
          <p class="hero-sub">↳ UNSUPERVISED MACHINE LEARNING  ·  PORTFOLIO PROJECT</p>
          <h1 class="hero-title">Netflix Content<br><span>Clustering</span></h1>
          <div style="margin-top:0.9rem;">
            <span class="tag">KMeans</span>
            <span class="tag">PCA</span>
            <span class="tag">StandardScaler</span>
            <span class="tag">scikit-learn</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── KPIs ─────────────────────────────────────────────────────────────────
    labels = run_kmeans(scaled, k)
    sil = silhouette_score(scaled, labels)
    metrics_df = compute_metrics(scaled)
    best_k = int(metrics_df.loc[metrics_df["Silhouette"].idxmax(), "K"])

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Titles",   f"{len(df):,}")
    col2.metric("Movies",         f"{int((df['type']=='Movie').sum()):,}")
    col3.metric("TV Shows",       f"{int((df['type']=='TV Show').sum()):,}")
    col4.metric("Active Clusters", k)
    col5.metric("Silhouette Score", f"{sil:.3f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_overview, tab_model, tab_clusters, tab_data = st.tabs(
        ["OVERVIEW", "MODEL SELECTION", "CLUSTER ANALYSIS", "EXPLORER"]
    )

    # ── Tab 1  OVERVIEW ───────────────────────────────────────────────────────
    with tab_overview:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<p class="section-label">Content Mix</p>', unsafe_allow_html=True)
            st.pyplot(fig_barh(df["type"], "Type Distribution",
                               top_n=5, color=PALETTE["red"]))

            st.markdown('<p class="section-label">Release Years</p>', unsafe_allow_html=True)
            release_series = df["release_year"].astype(str)
            st.pyplot(fig_barh(release_series, "Top Release Years",
                               top_n=10, color=PALETTE["accent"]))

        with col_b:
            st.markdown('<p class="section-label">Genres</p>', unsafe_allow_html=True)
            genre_series = df["listed_in"].fillna("").str.split(", ").explode()
            st.pyplot(fig_barh(genre_series, "Top Genres",
                               top_n=12, color=PALETTE["teal"]))

            st.markdown('<p class="section-label">Content Ratings</p>', unsafe_allow_html=True)
            st.pyplot(fig_barh(df["rating"].fillna("NR"), "Rating Distribution",
                               top_n=10, color=PALETTE["red_dim"]))

    # ── Tab 2  MODEL SELECTION ────────────────────────────────────────────────
    with tab_model:
        st.markdown('<p class="section-label">Choosing optimal K</p>', unsafe_allow_html=True)

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.pyplot(fig_elbow(metrics_df))
        with col_m2:
            st.pyplot(fig_silhouette(metrics_df))

        st.markdown("<br>", unsafe_allow_html=True)

        # Style all rows uniformly dark; highlight best K row in red
        _base = f"background-color:{PALETTE['surface']}; color:{PALETTE['text']};"
        _best = f"background-color:{PALETTE['card']}; color:{PALETTE['red']}; font-weight:600;"

        def _row_style(row):
            s = _best if row.name == best_k else _base
            return [s] * len(row)

        styled = (
            metrics_df.set_index("K").round(4)
            .style
            .format({"Inertia": "{:,.1f}", "Silhouette": "{:.4f}"})
            .apply(_row_style, axis=1)
        )
        st.dataframe(styled, use_container_width=True)
        st.caption(f"↑ Optimal K by silhouette score:  **K = {best_k}**")

        # PCA scatter
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-label">PCA Projection</p>', unsafe_allow_html=True)
        pca_2d = run_pca(scaled)
        st.pyplot(fig_pca(pca_2d, labels, k))

    # ── Tab 3  CLUSTER ANALYSIS ───────────────────────────────────────────────
    with tab_clusters:
        analysis_df = clean_df.copy()
        analysis_df["Cluster"] = labels

        for cid in sorted(analysis_df["Cluster"].unique()):
            subset = analysis_df[analysis_df["Cluster"] == cid]
            color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]

            pct_movie = (subset["type"] == "Movie").mean()
            avg_year  = int(subset["release_year"].mean())
            top_rating = subset["rating"].mode().iloc[0] if not subset["rating"].mode().empty else "—"

            st.markdown(
                f"""
                <div class="cluster-header" style="color:#F0F0F5;">
                  <span style="color:{color}; font-size:0.9rem; font-family:'DM Mono',monospace;
                               letter-spacing:0.1em; vertical-align:middle;">●</span>
                  &nbsp;<span style="color:#FFFFFF; font-family:'DM Serif Display',serif;
                                     font-size:1.4rem;">Cluster {cid}</span>
                  <span style="font-size:0.75rem; color:{PALETTE['accent']};
                               font-family:'DM Mono',monospace; margin-left:0.8rem;">
                    {len(subset):,} titles
                  </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Avg Release Year", avg_year)
            c2.metric("% Movies",         f"{pct_movie:.0%}")
            c3.metric("Top Rating",       top_rating)

            st.pyplot(fig_cluster_profile(subset, cid))

            with st.expander("Sample titles"):
                st.dataframe(
                    subset[["title", "type", "rating", "release_year", "listed_in"]]
                    .head(12),
                    use_container_width=True,
                )
            st.markdown("<br>", unsafe_allow_html=True)

    # ── Tab 4  EXPLORER ───────────────────────────────────────────────────────
    with tab_data:
        analysis_df = clean_df.copy()
        analysis_df["Cluster"] = labels

        col_f1, col_f2 = st.columns([1, 3])
        with col_f1:
            cluster_filter = st.selectbox(
                "Filter by cluster",
                ["All"] + [f"Cluster {i}" for i in sorted(analysis_df["Cluster"].unique())],
            )
            type_filter = st.selectbox("Filter by type", ["All", "Movie", "TV Show"])

        view = analysis_df.copy()
        if cluster_filter != "All":
            cid = int(cluster_filter.split()[-1])
            view = view[view["Cluster"] == cid]
        if type_filter != "All":
            view = view[view["type"] == type_filter]

        st.caption(f"{len(view):,} titles shown")
        cols_show = ["title", "type", "rating", "release_year", "listed_in", "Cluster"]
        st.dataframe(
            view[cols_show].reset_index(drop=True),
            use_container_width=True,
            height=480,
        )


if __name__ == "__main__":
    main()