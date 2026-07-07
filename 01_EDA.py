"""
Bioprocess Data Analytics - Exploratory Data Analysis
-------------------------------------------------------
Loads the fermentation dataset, summarizes key statistics, and visualizes
process parameter trends and correlations to identify relationships relevant
to biomass growth and yield.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "fermentation_data.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nSummary statistics:\n", df.describe())
print("\nMissing values:\n", df.isnull().sum())

# --- Correlation heatmap ---
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Process Parameters")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "correlation_heatmap.png"), dpi=150)
plt.close()

# --- Biomass growth curve ---
plt.figure(figsize=(8, 5))
plt.plot(df["Time_hours"], df["Biomass_g/L"], color="darkgreen")
plt.xlabel("Time (hours)")
plt.ylabel("Biomass (g/L)")
plt.title("Biomass Growth Over Time")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "biomass_growth_curve.png"), dpi=150)
plt.close()

# --- Process parameters over time ---
fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
axes[0].plot(df["Time_hours"], df["Temperature_C"], color="tomato")
axes[0].set_ylabel("Temperature (°C)")
axes[1].plot(df["Time_hours"], df["pH"], color="steelblue")
axes[1].set_ylabel("pH")
axes[2].plot(df["Time_hours"], df["Dissolved_Oxygen_%"], color="darkorange")
axes[2].set_ylabel("Dissolved O2 (%)")
axes[2].set_xlabel("Time (hours)")
fig.suptitle("Process Parameters Over Time")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "process_parameters_over_time.png"), dpi=150)
plt.close()

print(f"\nEDA complete. Plots saved to {RESULTS_DIR}/")
