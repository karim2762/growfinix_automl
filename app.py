import importlib
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config import SEED

st.set_page_config(page_title="Mini AutoML")
st.title("Mini AutoML with PyCaret")


def run_automl(df, target, task, fast, folds, ignore):
    pc = importlib.import_module(f"pycaret.{task}")
    df = df.dropna(subset=[target])  # rows without a label are useless
    pc.setup(df, target=target, session_id=SEED, fold=folds, ignore_features=ignore or None,
             html=False, verbose=False)
    include = ["lr", "dt", "rf", "et", "knn"] if fast else None
    best = pc.compare_models(include=include, n_select=1)
    board = pc.pull()
    final = pc.finalize_model(best)
    with tempfile.TemporaryDirectory() as tmp:
        pc.save_model(final, str(Path(tmp) / "model"))
        model_bytes = (Path(tmp) / "model.pkl").read_bytes()
    return {"board": board, "best": type(best).__name__, "bytes": model_bytes}


file = st.file_uploader("Upload a CSV", type="csv")
if file:
    data = pd.read_csv(file)
    st.dataframe(data.head())
    target = st.selectbox("Target column", data.columns, index=len(data.columns) - 1)

    # text or few distinct values -> classification
    y = data[target]
    guess = 0 if (not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 20) else 1
    task = st.radio("Task", ["classification", "regression"], index=guess, horizontal=True)

    # id columns are unique per row and inflate the score
    id_like = [c for c in data.columns if c != target and data[c].nunique() == len(data)]
    ignore = st.multiselect("Columns to ignore", [c for c in data.columns if c != target],
                            default=id_like)
    fast = st.checkbox("Fast mode (5 models)", value=True)
    folds = st.slider("CV folds", 3, 10, 5)

    if st.button("Run AutoML", type="primary"):
        with st.spinner("Training..."):
            try:
                st.session_state["result"] = run_automl(data, target, task, fast, folds, ignore)
            except Exception as err:
                st.error(f"AutoML failed: {err}")

result = st.session_state.get("result")
if result:
    st.subheader("Leaderboard")
    st.dataframe(result["board"])
    st.success(f"Best model: {result['best']}")
    st.download_button("Download trained model (.pkl)", result["bytes"], file_name="best_model.pkl")
    st.caption("Load it with pycaret's load_model('best_model')")
