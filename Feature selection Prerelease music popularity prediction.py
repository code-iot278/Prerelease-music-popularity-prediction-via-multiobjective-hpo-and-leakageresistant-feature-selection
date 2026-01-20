# =====================================================================================
# ONE-CELL FULL PIPELINE: FEATURE SELECTION, NESTED CV, MODELS, HPO, EVALUATION
# Sections 2.5 – 2.14 (Leakage-safe, Group-aware, Temporal)
# =====================================================================================

import numpy as np
import pandas as pd
import random
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# Sklearn imports
# -----------------------------
from sklearn.model_selection import GroupKFold, ParameterSampler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression, VarianceThreshold, RFE
from sklearn.linear_model import ElasticNet, Ridge, LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    roc_auc_score, average_precision_score
)
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr
from statsmodels.stats.multitest import fdrcorrection

# =====================================================================================
# 1. LOAD FEATURE MATRIX (output of previous cell)
# =====================================================================================
df = pd.read_csv("", parse_dates=["release_date"])

TARGET = "popularity_score"
GROUP = "artist"
TIME = "release_date"

X_full = df.drop(columns=[TARGET])
y_full = df[TARGET]
groups = df[GROUP]

# =====================================================================================
# 2. TEMPORAL BLOCKING (OUTER CV)
# =====================================================================================
df["year"] = df[TIME].dt.year
years = sorted(df["year"].unique())

outer_blocks = [(years[i], years[i+1]) for i in range(len(years)-1)]

# =====================================================================================
# 3. FILTER STAGE (Spearman + MI) — TRAIN FOLD ONLY
# =====================================================================================
def filter_stage(X, y):
    spearman_p = []
    spearman_r = []

    for col in X.columns:
        r, p = spearmanr(X[col], y)
        spearman_r.append(r)
        spearman_p.append(p)

    _, fdr_mask = fdrcorrection(spearman_p, alpha=0.05)
    mi = mutual_info_regression(X, y, random_state=SEED)

    keep = [
        col for col, ok, m in zip(X.columns, fdr_mask, mi)
        if ok and m > 1e-4
    ]
    return keep

# =====================================================================================
# 4. EMBEDDED + WRAPPER FEATURE SELECTION
# =====================================================================================
def embedded_wrapper_selection(X, y, groups, max_feats):
    gkf = GroupKFold(n_splits=5)

    # ---- Embedded: Elastic Net ----
    enet = ElasticNet(alpha=0.01, l1_ratio=0.7, random_state=SEED)
    enet.fit(X, y)
    nonzero = X.columns[np.abs(enet.coef_) > 1e-6]

    X_reduced = X[nonzero]

    # ---- Wrapper: RFE + Permutation ----
    base = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=SEED,
        n_jobs=-1
    )

    rfe = RFE(
        base,
        n_features_to_select=min(max_feats, X_reduced.shape[1]),
        step=0.2
    )
    rfe.fit(X_reduced, y)

    return X_reduced.columns[rfe.support_]

# =====================================================================================
# 5. MODEL ZOO
# =====================================================================================
MODELS = {
    "ridge": Ridge(),
    "elastic": ElasticNet(),
    "svr": SVR(kernel="rbf"),
    "rf": RandomForestRegressor(random_state=SEED),
    "gbr": GradientBoostingRegressor(random_state=SEED),
    "mlp": MLPRegressor(max_iter=500, random_state=SEED)
}

PARAM_SPACES = {
    "elastic": {
        "model__alpha": np.logspace(-3, 0, 20),
        "model__l1_ratio": np.linspace(0.1, 0.9, 10)
    },
    "rf": {
        "model__n_estimators": [200, 400],
        "model__max_depth": [6, 10, None]
    },
    "gbr": {
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__max_depth": [3, 5]
    }
}

# =====================================================================================
# 6. NESTED CV LOOP
# =====================================================================================
results = []

for train_year, val_year in outer_blocks[:-1]:

    train_idx = df["year"] <= train_year
    val_idx = df["year"] == val_year

    X_train, y_train = X_full[train_idx], y_full[train_idx]
    X_val, y_val = X_full[val_idx], y_full[val_idx]
    g_train = groups[train_idx]

    # -----------------------------
    # FILTER STAGE
    # -----------------------------
    keep_feats = filter_stage(X_train, y_train)
    X_train_f = X_train[keep_feats]
    X_val_f = X_val[keep_feats]

    # -----------------------------
    # INNER CV (Group-aware)
    # -----------------------------
    inner_gkf = GroupKFold(n_splits=5)

    for model_name, model in MODELS.items():

        param_space = PARAM_SPACES.get(model_name, {})
        param_list = list(ParameterSampler(
            param_space, n_iter=10, random_state=SEED
        )) or [{}]

        for params in param_list:

            rmses = []

            for tr, va in inner_gkf.split(X_train_f, y_train, g_train):
                X_tr, X_va = X_train_f.iloc[tr], X_train_f.iloc[va]
                y_tr, y_va = y_train.iloc[tr], y_train.iloc[va]

                # ---- Wrapper selection inside fold ----
                sel_feats = embedded_wrapper_selection(
                    X_tr, y_tr, g_train.iloc[tr], max_feats=50
                )

                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", model)
                ])
                pipe.set_params(**params)

                pipe.fit(X_tr[sel_feats], y_tr)
                preds = pipe.predict(X_va[sel_feats])

                rmses.append(mean_squared_error(y_va, preds, squared=False))

            score = np.mean(rmses)

            # -----------------------------
            # OUTER VALIDATION
            # -----------------------------
            sel_feats = embedded_wrapper_selection(
                X_train_f, y_train, g_train, max_feats=50
            )

            pipe.fit(X_train_f[sel_feats], y_train)
            val_preds = pipe.predict(X_val_f[sel_feats])

            results.append({
                "train_until": train_year,
                "validate_on": val_year,
                "model": model_name,
                "rmse": mean_squared_error(y_val, val_preds, squared=False),
                "mae": mean_absolute_error(y_val, val_preds),
                "spearman": spearmanr(y_val, val_preds)[0],
                "num_features": len(sel_feats)
            })

# =====================================================================================
# 7. FINAL REPORT
# =====================================================================================
results_df = pd.DataFrame(results)
print(results_df.groupby("model")[["rmse", "mae", "spearman"]].mean())
