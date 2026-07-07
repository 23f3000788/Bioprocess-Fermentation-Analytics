"""
Bioprocess Data Analytics - Yield Prediction
-----------------------------------------------
Predicts fermentation yield from process parameters using a Random Forest
regressor. Given the modest dataset size (168 hourly samples), evaluation
uses K-Fold cross-validation in addition to a held-out test split, for a
more reliable estimate of generalization performance than a single split.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "fermentation_data.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

FEATURES = ["Temperature_C", "pH", "Dissolved_Oxygen_%", "Biomass_g/L"]
TARGET = "Yield_g/L"

X = df[FEATURES]
y = df[TARGET]

# --- Held-out test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
test_rmse = mean_squared_error(y_test, y_pred) ** 0.5
test_mae = mean_absolute_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

# --- 5-Fold cross-validation on the full dataset for a more robust estimate ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmse_scores = -cross_val_score(model, X, y, cv=kf, scoring="neg_root_mean_squared_error")
cv_r2_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")

print("Held-out test set performance:")
print(f"  RMSE : {test_rmse:.4f} g/L")
print(f"  MAE  : {test_mae:.4f} g/L")
print(f"  R2   : {test_r2:.4f}")
print(f"\n5-Fold CV performance:")
print(f"  RMSE : {cv_rmse_scores.mean():.4f} +/- {cv_rmse_scores.std():.4f} g/L")
print(f"  R2   : {cv_r2_scores.mean():.4f} +/- {cv_r2_scores.std():.4f}")

with open(os.path.join(RESULTS_DIR, "yield_prediction_metrics.txt"), "w") as f:
    f.write("Random Forest Yield Prediction - Evaluation\n")
    f.write("Held-out test set:\n")
    f.write(f"  RMSE: {test_rmse:.4f} g/L\n")
    f.write(f"  MAE : {test_mae:.4f} g/L\n")
    f.write(f"  R2  : {test_r2:.4f}\n")
    f.write("5-Fold Cross-Validation:\n")
    f.write(f"  RMSE: {cv_rmse_scores.mean():.4f} +/- {cv_rmse_scores.std():.4f} g/L\n")
    f.write(f"  R2  : {cv_r2_scores.mean():.4f} +/- {cv_r2_scores.std():.4f}\n")

# --- Feature importance plot ---
importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
plt.figure(figsize=(7, 5))
importances.plot(kind="barh", color="seagreen")
plt.xlabel("Importance")
plt.title("Feature Importance in Yield Prediction (Random Forest)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=150)
plt.close()

# --- Predicted vs Actual plot ---
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, color="steelblue", edgecolor="k", alpha=0.8)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, "r--", label="Ideal (y=x)")
plt.xlabel("Actual Yield (g/L)")
plt.ylabel("Predicted Yield (g/L)")
plt.title(f"Predicted vs Actual Yield (R2={test_r2:.3f})")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "predicted_vs_actual_yield.png"), dpi=150)
plt.close()

print(f"\nPlots and metrics saved to {RESULTS_DIR}/")
