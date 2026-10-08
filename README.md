# task5_automl

AutoML on the Titanic data with PyCaret (main) and H2O (alternative), compared with a plain scikit-learn RandomForest.

**Python 3.10 or 3.11 is required for PyCaret 3.3.x.** It won't install on 3.12+. H2O also needs Java 8-17 (`java -version`).

## Run it

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python download_data.py          # saves data/titanic.csv
python src/baseline_sklearn.py   # run before pycaret, the chart uses it
python src/automl_pycaret.py
python src/automl_h2o.py         # optional

streamlit run app.py
jupyter notebook notebooks/exploration.ipynb
```

## Files

- `download_data.py`: gets Titanic from scikit-learn, saves it to `data/`
- `src/automl_pycaret.py`: the short PyCaret version, plus the saved leaderboard, charts and SHAP
- `src/automl_h2o.py`: H2OAutoML with `max_models=10`
- `src/baseline_sklearn.py`: manual RandomForest, to see what AutoML does for you
- `app.py`: Streamlit app, upload a CSV, pick the target, run, download the model
- `assets/`: leaderboard CSV and PNG, feature importance, SHAP, baseline metrics
- `notebooks/exploration.ipynb`: EDA, step-by-step run, leakage test

Seed is 42 everywhere (see `src/config.py`).

## When not to trust AutoML

**Data leakage.** AutoML maximizes the score it's given, leaked columns included. `boat` and `body` in Titanic give away the answer, so `download_data.py` drops them. IDs and anything recorded after the event do the same. A score that looks too good is a warning sign.

**Interpretability.** The best model is often a boosted or stacked ensemble you can't easily explain. For credit, health or hiring, use SHAP and feature importance, or choose a simpler model.

**Cost.** Many models times cross-validation times tuning uses a lot of CPU and time, more on big data. A tuned logistic regression is often only a point or two behind. Limit it with `max_models`, `max_runtime_secs` or fast mode in the app.

**Wrong setup.** It can't fix a bad target, biased samples, tiny data, or random splits on time series data.

**Leaderboard is not production.** Keep a hold-out set nobody touched, and watch for drift after deploying.

## Problems

- PyCaret won't install: check `python --version`
- H2O won't start: install Java 17
- LightGBM error on macOS: `brew install libomp`
- Pip version conflict: loosen the pin it names
