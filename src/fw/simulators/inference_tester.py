import numpy as np

from inference_surrogate import SurrogatePredictor


# load ONE raw sample from your original dataset
d = np.load("data/ship_dynamics_dataset_test.npz")
X = d["X"]
Y = d["Y"]

sample_idx = 0
x_raw = X[sample_idx]
y_true = Y[sample_idx]

predictor = SurrogatePredictor("checkpoints/best_surrogate.pth")

y_pred = predictor.predict_single(x_raw)

np.set_printoptions(precision=5, suppress=True)

print(f"True Y: {y_true}")
print(f"Pred Y: {y_pred}")
print(f"Abs error: {abs(y_pred - y_true)}")
