"""
train_model.py
--------------
Loads data/dip_diet.csv, encodes categorical features, trains and compares
multiple classifiers for DIP Diet profile classification, selects the best
model, and saves the full pipeline to models/model.pkl.

DIP Diet Profiles:
  - Green_Warrior   : Healthy lifestyle, high natural food intake
  - Balanced_Builder: Moderate health habits, room for improvement
  - Detox_Starter   : High processed food / low activity — needs reset
"""

import pathlib
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = pathlib.Path(__file__).parent
DATA_PATH  = BASE_DIR / "data" / "dip_diet.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"

MODELS_DIR.mkdir(exist_ok=True)

# ── 1. Load dataset ───────────────────────────────────────────────────────────
print(f"Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
print(df.head(3))

# ── 2. Define features and target ─────────────────────────────────────────────
TARGET_COL = "diet_profile"

NUMERIC_COLS = [
    "age", "weight_kg", "height_cm", "bmi",
    "activity_level", "sleep_hours", "water_intake_liters",
    "fruits_per_day", "vegetables_per_day",
    "processed_food_score", "sugar_intake_score", "stress_level",
]
CATEGORICAL_COLS = ["health_goal"]

FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

print(f"\nTarget        : {TARGET_COL}")
print(f"Numeric feats : {NUMERIC_COLS}")
print(f"Categorical   : {CATEGORICAL_COLS}")

X = df[FEATURE_COLS].copy()
y = df[TARGET_COL].copy()

# Encode target
le = LabelEncoder()
y_enc = le.fit_transform(y)
classes = le.classes_
print(f"\nClasses ({len(classes)}): {classes}")

# ── 3. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
)
print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")

# ── 4. Preprocessor (numeric → scale; categorical → one-hot) ─────────────────
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
    ]
)

# ── 5. Define candidate classifiers ──────────────────────────────────────────
candidates = {
    "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42),
    "K-Nearest Neighbors":  KNeighborsClassifier(),
    "Decision Tree":        DecisionTreeClassifier(random_state=42),
    "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM (RBF)":            SVC(kernel="rbf", probability=True, random_state=42),
}

# ── 6. 5-fold cross-validation on training set ────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("\n-- Cross-validation results (5-fold, training set) --")
for name, clf in candidates.items():
    pipe = Pipeline([("pre", preprocessor), ("clf", clf)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
    results[name] = scores.mean()
    print(f"  {name:<25}  CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

# ── 7. Select best model ──────────────────────────────────────────────────────
best_name = max(results, key=results.get)
print(f"\nBest model: {best_name}  (CV accuracy = {results[best_name]:.4f})")

# ── 8. Retrain best model on full training set ────────────────────────────────
best_pipeline = Pipeline([("pre", preprocessor), ("clf", candidates[best_name])])
best_pipeline.fit(X_train, y_train)

y_pred   = best_pipeline.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)

print(f"\nTest-set accuracy: {test_acc:.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=classes))

# ── 9. Save pipeline + metadata ──────────────────────────────────────────────
artifact = {
    "pipeline":      best_pipeline,
    "label_encoder": le,
    "feature_cols":  FEATURE_COLS,
    "numeric_cols":  NUMERIC_COLS,
    "categorical_cols": CATEGORICAL_COLS,
    "target_col":    TARGET_COL,
    "classes":       classes.tolist(),
    "model_name":    best_name,
    "test_accuracy": round(test_acc, 4),
    "cv_accuracy":   round(results[best_name], 4),
}

with open(MODEL_PATH, "wb") as f:
    pickle.dump(artifact, f)

print(f"\nModel saved -> {MODEL_PATH}")
print("Training complete.")
