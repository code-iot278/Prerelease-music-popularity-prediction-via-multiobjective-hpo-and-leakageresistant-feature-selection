# =====================================================================
# ONE-CELL FEATURE ENGINEERING PIPELINE (AUDIO + LYRICS + METADATA)
# Section 2.4 – Expressive, Interpretable, Leakage-Safe
# =====================================================================

# ---------------------------
# 1. INSTALL & IMPORTS
# ---------------------------
# !pip install librosa numpy pandas scipy scikit-learn networkx textstat nltk sentence-transformers

import numpy as np
import pandas as pd
import librosa
import librosa.display
import scipy.stats as stats
import networkx as nx
import nltk
import textstat

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sentence_transformers import SentenceTransformer

nltk.download("punkt")

# ---------------------------
# 2. HELPER FUNCTIONS
# ---------------------------
def robust_stats(x):
    return {
        "mean": np.mean(x),
        "median": np.median(x),
        "iqr": stats.iqr(x),
        "skew": stats.skew(x),
        "kurtosis": stats.kurtosis(x)
    }

def aggregate_feature(prefix, stats_dict):
    return {f"{prefix}_{k}": v for k, v in stats_dict.items()}

# ---------------------------
# 3. AUDIO FEATURE EXTRACTION
# ---------------------------
def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, mono=True, sr=None)
    hop = 512 if sr <= 22050 else 1024

    features = {}

    # ---- Spectral ----
    features.update(aggregate_feature(
        "spectral_centroid",
        robust_stats(librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0])
    ))
    features.update(aggregate_feature(
        "spectral_bandwidth",
        robust_stats(librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop)[0])
    ))
    features.update(aggregate_feature(
        "rolloff",
        robust_stats(librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop)[0])
    ))
    features.update(aggregate_feature(
        "flatness",
        robust_stats(librosa.feature.spectral_flatness(y=y)[0])
    ))
    features.update(aggregate_feature(
        "zcr",
        robust_stats(librosa.feature.zero_crossing_rate(y, hop_length=hop)[0])
    ))

    # ---- MFCCs + deltas ----
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop)
    delta = librosa.feature.delta(mfcc)

    for i in range(20):
        features.update(aggregate_feature(f"mfcc{i+1}", robust_stats(mfcc[i])))
        features.update(aggregate_feature(f"mfcc{i+1}_delta", robust_stats(delta[i])))

    # ---- Harmony & Tonality ----
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    features["pitch_class_entropy"] = stats.entropy(np.mean(chroma, axis=1))
    features["tonic_stability"] = np.max(np.mean(chroma, axis=1))

    # ---- Rhythm & Tempo ----
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
    features["tempo_bpm"] = tempo
    features["beat_strength"] = np.mean(librosa.onset.onset_strength(y=y, sr=sr))
    features["onset_rate"] = len(librosa.onset.onset_detect(y=y, sr=sr)) / (len(y)/sr)

    # ---- Dynamics ----
    rms = librosa.feature.rms(y=y)[0]
    features["rms_mean"] = rms.mean()
    features["dynamic_range"] = np.percentile(rms, 95) - np.percentile(rms, 5)
    features["crest_factor"] = np.max(np.abs(y)) / (rms.mean() + 1e-6)

    # ---- Structure (approximate) ----
    features["section_count"] = len(np.unique(beats))
    features["section_length_dispersion"] = np.std(np.diff(beats)) if len(beats) > 1 else 0

    return features

# ---------------------------
# 4. LYRICS & SEMANTICS
# ---------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def extract_lyrics_features(text):
    if pd.isna(text):
        return {"lyrics_missing": 1}

    tokens = nltk.word_tokenize(text.lower())
    unique_tokens = set(tokens)

    features = {
        "lyrics_missing": 0,
        "type_token_ratio": len(unique_tokens) / max(len(tokens), 1),
        "syllable_rate": textstat.syllable_count(text) / max(len(tokens), 1),
        "readability": textstat.flesch_reading_ease(text),
        "pronoun_ratio": sum(t in ["i", "we", "you", "they"] for t in tokens) / max(len(tokens), 1)
    }

    # ---- Sentiment (proxy via readability & structure) ----
    features["valence_proxy"] = np.tanh(features["readability"] / 100)

    # ---- Topic / Semantic Embedding ----
    emb = embedder.encode(text)
    for i, v in enumerate(emb[:384]):
        features[f"emb_{i}"] = v

    return features

# ---------------------------
# 5. METADATA & CONTEXT
# ---------------------------
def extract_metadata_features(row, artist_history, collab_graph):
    artist = row["artist"]

    features = {
        "prior_releases": artist_history.get(artist, {}).get("releases", 0),
        "historical_hit_rate": artist_history.get(artist, {}).get("hit_rate", 0),
        "time_since_last_release": artist_history.get(artist, {}).get("gap_days", 0),
        "num_collaborators": collab_graph.degree(artist) if artist in collab_graph else 0,
        "pagerank": nx.pagerank(collab_graph).get(artist, 0),
        "track_duration": row["duration"],
        "explicit_flag": row["explicit"],
        "release_weekday": row["release_date"].weekday(),
        "release_quarter": (row["release_date"].month - 1) // 3 + 1
    }
    return features

# ---------------------------
# 6. BUILD FEATURE MATRIX
# ---------------------------
df = pd.read_csv("music_dataset.csv", parse_dates=["release_date"])

audio_features = []
lyrics_features = []

for _, r in df.iterrows():
    audio_features.append(extract_audio_features(r["audio_path"]))
    lyrics_features.append(extract_lyrics_features(r["lyrics"]))

audio_df = pd.DataFrame(audio_features)
lyrics_df = pd.DataFrame(lyrics_features)

# Dummy artist history & collaboration graph (pre-release)
artist_history = {}
collab_graph = nx.Graph()

meta_features = df.apply(
    lambda r: extract_metadata_features(r, artist_history, collab_graph),
    axis=1
)
meta_df = pd.DataFrame(list(meta_features))

# ---------------------------
# 7. FEATURE HYGIENE
# ---------------------------
X = pd.concat([audio_df, lyrics_df, meta_df], axis=1)

# Remove near-constant features
vt = VarianceThreshold(threshold=1e-4)
X = X.loc[:, vt.fit(X).get_support()]

# Remove highly collinear features
corr = X.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
X = X.drop(columns=[c for c in upper.columns if any(upper[c] > 0.95)])

# PCA for embeddings
emb_cols = [c for c in X.columns if c.startswith("emb_")]
X_emb = PCA(n_components=64).fit_transform(X[emb_cols])
X_emb = pd.DataFrame(X_emb, columns=[f"emb_pca_{i}" for i in range(64)])

X = pd.concat([X.drop(columns=emb_cols), X_emb], axis=1)

# ---------------------------
# 8. FINAL OUTPUT
# ---------------------------
print("Final feature matrix shape:", X.shape)
X.head()
