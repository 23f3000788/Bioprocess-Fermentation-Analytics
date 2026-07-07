# Bioprocess Data Analytics for Fermentation Optimization

Simulated fermentation dataset analyzed with time-series forecasting and machine
learning regression to monitor key process parameters and predict fermentation
outcomes, supported by an interactive dashboard.

## Project Structure
```
├── data/
│   └── fermentation_data.csv       # 168 hourly process readings
├── notebooks/
│   ├── 01_EDA.py                   # Exploratory data analysis
│   ├── 02_TimeSeries.py            # ARIMA biomass forecasting
│   └── 03_YieldPrediction.py       # Random Forest yield prediction
├── dashboard/
│   └── app.py                      # Plotly Dash interactive dashboard
└── results/                        # Generated plots and metrics
```

## Methods & Results

**Biomass Forecasting (ARIMA(2,1,2))**
Forecast evaluated against a held-out 24-hour window (not just projected blindly
past the end of the series):
- MAE: 0.081 g/L
- RMSE: 0.086 g/L
- MAPE: 0.93%

**Yield Prediction (Random Forest Regressor)**
Evaluated with both a held-out test split and 5-fold cross-validation:
- Held-out test R²: 0.981, RMSE: 0.270 g/L, MAE: 0.181 g/L
- 5-fold CV R²: 0.978 ± 0.002, RMSE: 0.292 ± 0.020 g/L
- Dissolved Oxygen and Biomass were the strongest predictors of yield (see `results/feature_importance.png`)

## Skills Demonstrated
- Data cleaning & exploratory data analysis
- Time-series forecasting with proper train/test evaluation (ARIMA)
- Regression modeling with cross-validation (Random Forest)
- Interactive data visualization (Plotly Dash)

## Running the project
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels dash plotly
python notebooks/01_EDA.py
python notebooks/02_TimeSeries.py
python notebooks/03_YieldPrediction.py
python dashboard/app.py   # then open http://127.0.0.1:8050
```
