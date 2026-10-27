import numpy as np

from fw.surrogate.dataset import load_raw
from fw.surrogate.inference_surrogate import SurrogatePredictor


predictor = SurrogatePredictor("checkpoints/best_surrogate.pth")

# load ONE raw sample from a dataset file, in the checkpoint scenario's raw
# layout (load_raw checks the file's identity and projects if allowed)
X, Y, identity = load_raw("data/ship_dynamics_dataset_test.npz", predictor.scenario)
mismatch = predictor.data_mismatch(identity)
if mismatch:
    raise SystemExit(f"data/ship_dynamics_dataset_test.npz: {mismatch}.")

sample_idx = 0
x_raw = X[sample_idx]
y_true = Y[sample_idx]

y_pred = predictor.predict_single(x_raw)

np.set_printoptions(precision=5, suppress=True)

print(f"Checkpoint: scenario={predictor.scenario.name}, env_encoding={predictor.env_encoding}, "
      f"ship_model={predictor.ship_model}; data: {identity}")
print(f"True Y: {y_true}")
print(f"Pred Y: {y_pred}")
print(f"Abs error: {abs(y_pred - y_true)}")
