"""
Bioprocess Data Analytics - Biomass Forecasting with ARIMA
------------------------------------------------------------
Fits an ARIMA model on a training window of the biomass time series and
evaluates forecast accuracy on a held-out window, rather than forecasting
blindly past the end of the observed data with no ground truth to check against.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "fermentation_data.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
series = df["Biomass_g/L"]

# --- Train/test split: hold out the final 24 hours to validate the forecast ---
HORIZON = 24
train, test = series[:-HORIZON], series[-HORIZON:]

model = ARIMA(train, order=(2, 1, 2))
model_fit = model.fit()
print(model_fit.summary())

forecast_result = model_fit.get_forecast(steps=HORIZON)
forecast = forecast_result.predicted_mean
conf_int = forecast_result.conf_int(alpha=0.05)

# --- Evaluate against the actual held-out values ---
mae = mean_absolute_error(test, forecast)
rmse = mean_squared_error(test, forecast) ** 0.5
mape = float(np.mean(np.abs((test.values - forecast.values) / test.values)) * 100)

print(f"\nHeld-out forecast evaluation (last {HORIZON} hours):")
print(f"  MAE  : {mae:.4f} g/L")
print(f"  RMSE : {rmse:.4f} g/L")
print(f"  MAPE : {mape:.2f}%")

with open(os.path.join(RESULTS_DIR, "arima_metrics.txt"), "w") as f:
    f.write(f"ARIMA(2,1,2) Biomass Forecast Evaluation (held-out last {HORIZON} hours)\n")
    f.write(f"MAE  : {mae:.4f} g/L\n")
    f.write(f"RMSE : {rmse:.4f} g/L\n")
    f.write(f"MAPE : {mape:.2f}%\n")

# --- Plot observed vs forecast with confidence interval ---
plt.figure(figsize=(10, 6))
plt.plot(train.index, train, label="Training Data", color="steelblue")
plt.plot(test.index, test, label="Actual (Held-out)", color="black", linewidth=2)
plt.plot(test.index, forecast, label="ARIMA Forecast", color="tomato", linestyle="--")
plt.fill_between(test.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                  color="tomato", alpha=0.15, label="95% CI")
plt.xlabel("Time (hours)")
plt.ylabel("Biomass (g/L)")
plt.title(f"Biomass Forecasting with ARIMA (MAE={mae:.3f}, RMSE={rmse:.3f})")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "biomass_forecast.png"), dpi=150)
plt.close()

print(f"\nForecast plot and metrics saved to {RESULTS_DIR}/")
