from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ASSETS_DIR = ROOT / "assets"
MODELS_DIR = ROOT / "models"

DATA_PATH = DATA_DIR / "titanic.csv"
MODEL_PATH = MODELS_DIR / "pycaret_best"  # pycaret adds .pkl
TARGET = "survived"

# fixed so splits give the same numbers every run
SEED = 42


def ensure_dirs():
    for folder in (DATA_DIR, ASSETS_DIR, MODELS_DIR):
        folder.mkdir(parents=True, exist_ok=True)
