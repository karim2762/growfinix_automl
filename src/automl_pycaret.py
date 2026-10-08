import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pycaret.classification import (compare_models, create_model, evaluate_model,
                                    finalize_model, interpret_model, plot_model,
                                    predict_model, pull, save_model, setup,
                                    tune_model)

from config import ASSETS_DIR, DATA_PATH, MODEL_PATH, SEED, TARGET, ensure_dirs

ensure_dirs()

df = pd.read_csv(DATA_PATH)  # load the csv
setup(df, target=TARGET, session_id=SEED, train_size=0.8, html=False, verbose=False)  # cleaning + split + seed
best = compare_models(n_select=1, sort="AUC")  # cross-validate ~15 models, keep the top one
leaderboard = pull()  # the table compare_models just showed
tuned = tune_model(best, optimize="AUC")  # random search on its hyperparameters
try:
    evaluate_model(tuned)  # widget, only works in jupyter
except Exception:
    pass  # fine in a plain script, plots are saved below
holdout = predict_model(tuned); holdout_metrics = pull()  # score on the 20% held out
final = finalize_model(tuned)  # refit on all the rows
save_model(final, str(MODEL_PATH))  # writes models/pycaret_best.pkl

leaderboard.to_csv(ASSETS_DIR / "leaderboard_pycaret.csv")

chart = leaderboard.set_index("Model")[["AUC", "Accuracy"]].sort_values("AUC")
ax = chart.plot.barh(figsize=(9, 7), title="PyCaret leaderboard (5-fold CV)")
base_file = ASSETS_DIR / "baseline_metrics.json"
if base_file.exists():
    base_acc = json.loads(base_file.read_text())["accuracy"]
    ax.axvline(base_acc, color="red", ls="--", label=f"manual RF accuracy {base_acc}")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(ASSETS_DIR / "leaderboard_pycaret.png", dpi=150)
plt.close()

# shap needs a tree model, the winner might not be one
tree = create_model("rf", verbose=False)
for name, action in {
    "feature_importance.png": lambda: plot_model(tree, plot="feature", save=str(ASSETS_DIR)),
    "shap_summary.png": lambda: interpret_model(tree, plot="summary", save=str(ASSETS_DIR)),
}.items():
    try:
        action()
    except Exception as err:
        print(f"could not make {name}: {err}")

# pycaret picks its own file names
for old, new in {"Feature Importance.png": "feature_importance.png",
                 "SHAP summary.png": "shap_summary.png"}.items():
    if (ASSETS_DIR / old).exists():
        (ASSETS_DIR / old).replace(ASSETS_DIR / new)

print(leaderboard.head(10))
print("Hold-out metrics of tuned model:\n", holdout_metrics)
