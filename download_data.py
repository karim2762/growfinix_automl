import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from sklearn.datasets import fetch_openml

from config import DATA_PATH, TARGET, ensure_dirs

ensure_dirs()

raw = fetch_openml("titanic", version=1, as_frame=True, parser="auto").frame

# boat and body are only known after the sinking, they leak the label.
# the other four are near-unique text, just noise here.
df = raw.drop(columns=["boat", "body", "name", "ticket", "cabin", "home.dest"])

df[TARGET] = df[TARGET].astype(int)

df.to_csv(DATA_PATH, index=False)
print(f"Saved {len(df)} rows x {df.shape[1]} columns to {DATA_PATH}")
print(df.head())
