# ============================================================
# FULL ONE-CELL DATA CLEANING, NORMALIZATION & LEAKAGE CONTROL
# ============================================================

# -----------------------------
# 1. Install & Imports
# -----------------------------
# !pip install pandas numpy scikit-learn category_encoders rapidfuzz

import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import GroupKFold
from category_encoders import TargetEncoder

# -----------------------------
# 2. Load Dataset
# -----------------------------
df = pd.read_csv(
    "music_dataset.csv",
    parse_dates=["release_date", "feature_timestamp"]
)

# -----------------------------
# 3. De-duplication
# -----------------------------
# 3.1 Exact ISRC matches (keep latest canonical)
df = (
    df.sort_values("feature_timestamp")
      .drop_duplicates(subset="isrc", keep="last")
)

# 3.2 Fuzzy artist + title matching
def fuzzy_deduplicate(df, threshold=95):
    keep_idx = []
    seen = []

    for i, r in df.iterrows():
        key = f"{r['artist']} {r['title']}".lower()
        if not any(fuzz.token_sort_ratio(key, s) >= threshold for s in seen):
            seen.append(key)
            keep_idx.append(i)

    return df.loc[keep_idx]

df = fuzzy_deduplicate(df)

# -----------------------------
# 4. Time Alignment (No Future Leakage)
# -----------------------------
df = df[df["feature_timestamp"] <= df["release_date"]]

# Event-time raster (days before release)
df["event_time_days"] = (
    df["release_date"] - df["feature_timestamp"]
).dt.days

# -----------------------------
# 5. Missing Data Handling
# -----------------------------
# Lyrics missing flag
df["lyrics_missing"] = df["lyrics"].isna().astype(int)

# Artist–Genre–Year stratification
df["release_year"] = df["release_date"].dt.year
group_cols = ["artist", "genre", "release_year"]

# Numeric imputation (median)
num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = (
    df.groupby(group_cols)[num_cols]
      .transform(lambda x: x.fillna(x.median()))
)

# Categorical imputation (mode)
cat_cols = df.select_dtypes(include="object").columns
df[cat_cols] = (
    df.groupby(group_cols)[cat_cols]
      .transform(lambda x: x.fillna(
          x.mode().iloc[0] if not x.mode().empty else "unknown"
      ))
)

# -----------------------------
# 6. Scaling & Encoding
# -----------------------------
target = "popularity_score"

heavy_tailed = ["stream_count", "followers"]
standard_numeric = ["tempo", "loudness", "energy"]
low_cardinality = ["genre"]
high_cardinality = ["artist", "label"]

# 6.1 Leakage-safe target encoding (OOF)
def oof_target_encode(df, col, target, groups, n_splits=5):
    encoded = pd.Series(index=df.index, dtype=float)
    gkf = GroupKFold(n_splits=n_splits)

    for tr, va in gkf.split(df, df[target], groups):
        te = TargetEncoder(cols=[col])
        te.fit(df.iloc[tr][[col]], df.iloc[tr][target])
        encoded.iloc[va] = te.transform(df.iloc[va][[col]])[col]

    return encoded

for col in high_cardinality:
    df[col + "_te"] = oof_target_encode(
        df, col, target, df["artist"]
    )

# 6.2 One-hot encoding (≤20 categories)
df = pd.get_dummies(df, columns=low_cardinality, drop_first=True)

# 6.3 Scaling
df[standard_numeric] = StandardScaler().fit_transform(df[standard_numeric])
df[heavy_tailed] = RobustScaler().fit_transform(df[heavy_tailed])

# -----------------------------
# 7. Chronological Train / Val / Test
# -----------------------------
df = df.sort_values("release_date")

train_df = df[df["release_date"] < "2019-01-01"]
val_df = df[
    (df["release_date"] >= "2019-01-01") &
    (df["release_date"] < "2021-01-01")
]
test_df = df[df["release_date"] >= "2021-01-01"]

# -----------------------------
# 8. Artist-aware Cross-Validation
# -----------------------------
X = train_df.drop(columns=[target])
y = train_df[target]
groups = train_df["artist"]

gkf = GroupKFold(n_splits=5)

for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
    print(
        f"Fold {fold} | "
        f"Train artists: {X.iloc[tr].artist.nunique()} | "
        f"Val artists: {X.iloc[va].artist.nunique()}"
    )

# -----------------------------
# 9. Final Outputs
# -----------------------------
print("Train shape:", train_df.shape)
print("Validation shape:", val_df.shape)
print("Test shape:", test_df.shape)
