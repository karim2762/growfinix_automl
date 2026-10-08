import h2o
from h2o.automl import H2OAutoML

from config import ASSETS_DIR, DATA_PATH, MODELS_DIR, SEED, TARGET, ensure_dirs

ensure_dirs()
h2o.init(nthreads=-1)

frame = h2o.import_file(str(DATA_PATH))
frame[TARGET] = frame[TARGET].asfactor()  # factor = classification

train, test = frame.split_frame(ratios=[0.8], seed=SEED)
features = [c for c in frame.columns if c != TARGET]

# max_runtime_secs so it can't hang on a slow laptop
aml = H2OAutoML(max_models=10, seed=SEED, max_runtime_secs=300, sort_metric="AUC")
aml.train(x=features, y=TARGET, training_frame=train)

board = aml.leaderboard.as_data_frame()
print(board)
board.to_csv(ASSETS_DIR / "leaderboard_h2o.csv", index=False)

print("Best model on unseen test data:")
print(aml.leader.model_performance(test))

h2o.save_model(aml.leader, path=str(MODELS_DIR), force=True)
h2o.cluster().shutdown()
