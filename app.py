"""
Bioprocess Data Analytics Dashboard
--------------------------------------
Interactive Plotly Dash app to explore fermentation process parameters,
biomass growth, and yield over time.

Run with: python app.py
Then open http://127.0.0.1:8050 in your browser.
"""
import os
import pandas as pd
import dash
from dash import dcc, html
import plotly.express as px

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "fermentation_data.csv")
df = pd.read_csv(DATA_PATH)

app = dash.Dash(__name__)
app.title = "Bioprocess Data Analytics Dashboard"

app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif", "padding": "20px", "maxWidth": "1000px", "margin": "0 auto"},
    children=[
        html.H1("Bioprocess Data Analytics Dashboard", style={"textAlign": "center"}),
        html.P(
            "Monitoring key fermentation process parameters, biomass growth, and yield over time.",
            style={"textAlign": "center", "color": "#555"},
        ),
        dcc.Graph(
            id="biomass_trend",
            figure=px.line(
                df, x="Time_hours", y="Biomass_g/L",
                title="Biomass Growth Over Time",
                labels={"Time_hours": "Time (hours)", "Biomass_g/L": "Biomass (g/L)"},
            ).update_traces(line_color="#2E7D32"),
        ),
        dcc.Graph(
            id="yield_trend",
            figure=px.line(
                df, x="Time_hours", y="Yield_g/L",
                title="Yield Over Time",
                labels={"Time_hours": "Time (hours)", "Yield_g/L": "Yield (g/L)"},
            ).update_traces(line_color="#1565C0"),
        ),
        dcc.Graph(
            id="parameter_trends",
            figure=px.line(
                df, x="Time_hours", y=["Temperature_C", "pH", "Dissolved_Oxygen_%"],
                title="Process Parameters Over Time",
                labels={"Time_hours": "Time (hours)", "value": "Value", "variable": "Parameter"},
            ),
        ),
    ],
)

if __name__ == "__main__":
    app.run(debug=True)
