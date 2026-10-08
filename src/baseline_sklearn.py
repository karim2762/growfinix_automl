import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import ASSETS_DIR, DATA_PATH, SEED, TARGET, ensure_dirs

ensure_dirs()
df = pd.read_csv(DATA_PATH)
X, y = df.drop(columns=[TARGET]), df[TARGET]

# stratify keeps the survived ratio the same in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

num_cols = X.select_dtypes(include="number").columns
cat_cols = X.select_dtypes(exclude="number").columns

# the model can't take NaNs or strings, so fix both by hand
prep = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([
        ("fill", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), cat_cols),
])

model = Pipeline([
    ("prep", prep),
    ("rf", RandomForestClassifier(n_estimators=200, random_state=SEED)),
])

model.fit(X_train, y_train)
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]
metrics = {
    "model": "RandomForest (manual sklearn)",
    "accuracy": round(accuracy_score(y_test, pred), 4),
    "auc": round(roc_auc_score(y_test, proba), 4),
    "f1": round(f1_score(y_test, pred), 4),
}
print(metrics)

# the pycaret script reads this to draw the baseline line
(ASSETS_DIR / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))
print("One model, no tuning. AutoML tries about 15 and tunes the best one.")
