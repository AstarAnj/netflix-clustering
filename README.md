# Netflix Content Clustering · KMeans + PCA

A portfolio project that applies an unsupervised machine learning pipeline to the Netflix catalogue using metadata-driven feature engineering, KMeans clustering, and PCA-based visualisation.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=flat-square)

**[→ Live Demo](https://netflix-clustering-bdjyf9k5b3tcnxom9xdmnh.streamlit.app)**

---

## Project Overview

This project clusters ~8,800 Netflix titles into interpretable content segments using metadata such as content type, maturity rating, duration, release year, and genre indicators.

The goal is to demonstrate a complete unsupervised ML workflow — from raw CSV to interactive dashboard — including data cleaning, feature engineering, model selection, and result interpretation.

---

## Dashboard Tabs

| Tab | What it shows |
|---|---|
| **Overview** | Dataset distributions — genres, ratings, content type, release years |
| **Model Selection** | Elbow curve, silhouette scores, PCA scatter plot |
| **Cluster Analysis** | Per-cluster profiles with rating mix, top genres, and key metrics |
| **Explorer** | Filterable table to browse titles by cluster and content type |

---

## ML Pipeline

### 1 · Feature Engineering

Each title is converted into a numeric feature vector:

| Feature | Method |
|---|---|
| `release_year` | Raw integer |
| `rating` | Label encoding (e.g. TV-MA → 7) |
| `duration_num` | Minutes extracted for movies; seasons for TV shows |
| `is_movie` | Binary flag — 1 for Movie, 0 for TV Show |
| `genre_*` | One-hot columns for the top 12 genres |

### 2 · Feature Scaling

`StandardScaler` is applied before clustering. Without scaling, `release_year` (range ~1970–2023) would dominate Euclidean distance calculations and suppress the contribution of binary features like `is_movie`. Scaling centres each feature to mean = 0, std = 1.

### 3 · Choosing K

Two complementary methods are used:

- **Elbow Method** — plots inertia (within-cluster sum of squares) vs K. The point where the curve flattens suggests diminishing returns from adding more clusters.
- **Silhouette Score** — measures how well each point fits its own cluster compared to the nearest neighbouring cluster. Ranges from −1 to +1; higher is better. The app automatically highlights the optimal K.

### 4 · Clustering & Visualisation

KMeans is fit on the scaled feature matrix for the selected K. PCA reduces the feature space to 2 principal components so cluster assignments can be plotted on a 2D scatter chart.

---

## Tech Stack

| Layer | Library |
|---|---|
| Language | Python 3.9+ |
| ML | scikit-learn |
| Data | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Dashboard | Streamlit |
| Dataset | kagglehub / local CSV / synthetic fallback |

---

## Project Structure

```
netflix-clustering/
│
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── README.md
├── .gitignore
└── data/
    └── netflix_titles.csv  # Optional — auto-downloaded or synthesised if absent
```

---

## Run Locally

```bash
# 1. Clone
git clone https://github.com/AstarAnj/netflix-clustering.git
cd netflix-clustering

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

> **Dataset** — place `netflix_titles.csv` inside the `data/` folder (available on [Kaggle](https://www.kaggle.com/datasets/shivamb/netflix-shows)). If absent, the app falls back to a Kaggle auto-download or generates a synthetic dataset of ~2,000 titles.

---

## Dataset

- **Source:** [Netflix Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows) — Shivam Bansal on Kaggle
- **Size:** ~8,800 titles
- **Features used:** `type`, `rating`, `listed_in`, `release_year`, `duration`

---

## License

MIT — free to use, modify, and distribute.