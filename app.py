from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Fuel Price Prediction System",
    page_icon="⛽",
    layout="wide",
)

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "Data"
MODEL_DIR = BASE / "models"

CITIES = ["Chennai", "Delhi", "Mumbai", "Kolkata"]

@st.cache_data
def load_data():
    fuel = pd.read_csv(DATA_DIR / "india_fuel_prices_daily.csv")
    crude = pd.read_csv(DATA_DIR / "crude-oil-price.csv")
    metrics = pd.read_csv(MODEL_DIR / "model_comparison_metrics.csv")
    predictions = pd.read_csv(MODEL_DIR / "test_predictions.csv")

    fuel["date"] = pd.to_datetime(fuel["date"], errors="coerce")
    crude["date"] = pd.to_datetime(
        crude["date"], errors="coerce", utc=True
    ).dt.tz_localize(None)

    predictions["date"] = pd.to_datetime(
        predictions["date"], errors="coerce"
    )
    predictions["target_date"] = pd.to_datetime(
        predictions["target_date"], errors="coerce"
    )

    # Reproduce the correction used during model development.
    correction = (
        fuel["date"].between("2022-06-01", "2022-06-30")
        & fuel["city"].isin(CITIES)
    )
    old_petrol = fuel.loc[
        correction, "petrol_price_inr_per_litre"
    ].copy()

    fuel.loc[correction, "petrol_price_inr_per_litre"] = (
        fuel.loc[correction, "diesel_price_inr_per_litre"].values
    )
    fuel.loc[correction, "diesel_price_inr_per_litre"] = old_petrol.values

    return fuel, crude, metrics, predictions

st.title("⛽ Fuel Price Prediction System")
st.markdown(
    "Explore historical fuel prices, evaluate model performance, "
    "and investigate future price scenarios."
)

try:
    fuel, crude, metrics, predictions = load_data()
except Exception as exc:
    st.error(f"Unable to load project files: {exc}")
    st.stop()

latest_dataset_date = fuel["date"].max()
st.info(
    f"Historical fuel dataset ends on {latest_dataset_date:%d %B %Y}. "
    "Forecasts below are based on historical data, not live market prices."
)

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("Forecast settings")
city = st.sidebar.selectbox("City", CITIES)
fuel_type = st.sidebar.selectbox("Fuel type", ["Petrol", "Diesel"])

price_col = (
    "petrol_price_inr_per_litre"
    if fuel_type == "Petrol"
    else "diesel_price_inr_per_litre"
)

target_col = (
    "target_petrol_tomorrow"
    if fuel_type == "Petrol"
    else "target_diesel_tomorrow"
)

pred_col = (
    "predicted_petrol"
    if fuel_type == "Petrol"
    else "predicted_diesel"
)

city_history = (
    fuel.loc[
        (fuel["city"] == city) & fuel[price_col].notna(),
        ["date", price_col],
    ]
    .dropna()
    .sort_values("date")
    .drop_duplicates("date", keep="last")
)

if city_history.empty:
    st.error("No historical prices are available for this city.")
    st.stop()

last_row = city_history.iloc[-1]
last_date = last_row["date"]
last_price = float(last_row[price_col])

starting_price = st.sidebar.number_input(
    "Starting price (₹ per litre)",
    min_value=0.01,
    max_value=10000.0,
    value=last_price,
    step=0.10,
    help="Change this to explore a what-if scenario.",
)

horizon = st.sidebar.slider(
    "Forecast horizon (days)", min_value=1, max_value=30, value=7
)

daily_change_pct = st.sidebar.slider(
    "Assumed daily change (%)",
    min_value=-2.0,
    max_value=2.0,
    value=0.0,
    step=0.1,
    help="Used only for the what-if scenario, not the ML forecast.",
)

# ---------------- OVERVIEW ----------------
st.subheader(f"{city} {fuel_type} dashboard")

a, b, c = st.columns(3)
a.metric("Latest recorded price", f"₹{last_price:.2f}/L")
b.metric("User-selected starting price", f"₹{starting_price:.2f}/L")
c.metric("Historical data through", last_date.strftime("%d %b %Y"))

st.caption(
    "The forecast starts after the latest date in the dataset. "
    "The user-selected starting price is used for the what-if scenario."
)

# ---------------- HISTORICAL AND FUTURE GRAPHS ----------------
future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=horizon,
    freq="D",
)

# Selected model is the persistence baseline. It carries forward the
# last recorded price; it does not independently predict price changes.
model_prices = [last_price] * horizon

# Scenario compounds the user-selected assumed daily percentage change.
scenario_prices = []
scenario_price = starting_price

for _ in range(horizon):
    scenario_price *= 1 + daily_change_pct / 100
    scenario_prices.append(scenario_price)

history_window = city_history.tail(180)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=history_window["date"],
        y=history_window[price_col],
        mode="lines",
        name="Historical actual price",
        line=dict(width=2),
    )
)
fig.add_trace(
    go.Scatter(
        x=future_dates,
        y=model_prices,
        mode="lines+markers",
        name="Baseline model forecast",
        line=dict(dash="dash", width=3),
    )
)
fig.add_trace(
    go.Scatter(
        x=future_dates,
        y=scenario_prices,
        mode="lines+markers",
        name="User what-if scenario",
        line=dict(dash="dot", width=3),
    )
)
fig.add_vline(
    x=last_date.timestamp() * 1000,
    line_dash="dash",
    annotation_text="Forecast begins",
)
fig.update_layout(
    title=f"{city} {fuel_type}: historical prices and future scenarios",
    xaxis_title="Date",
    yaxis_title="Price (₹ per litre)",
    hovermode="x unified",
    legend_title="Series",
)
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
left.metric(
    f"Baseline estimate after {horizon} days",
    f"₹{model_prices[-1]:.2f}/L",
)
right.metric(
    f"What-if price after {horizon} days",
    f"₹{scenario_prices[-1]:.2f}/L",
    delta=f"{scenario_prices[-1] - starting_price:+.2f} ₹/L",
)

st.warning(
    "The baseline forecast assumes no price change. The what-if line "
    "applies your assumed daily percentage and is not an ML prediction."
)

future_table = pd.DataFrame({
    "Forecast date": future_dates,
    "Baseline forecast (₹/L)": model_prices,
    "What-if scenario (₹/L)": scenario_prices,
})
st.subheader("Future price table")
st.dataframe(future_table, use_container_width=True, hide_index=True)

# ---------------- MODEL METRICS ----------------
st.divider()
st.header("Model evaluation")
st.write(
    "MAE and RMSE are computed from the held-out test period. "
    "Lower values indicate smaller prediction errors."
)

fuel_metrics = metrics[
    metrics["fuel"].astype(str).str.lower() == fuel_type.lower()
]

if not fuel_metrics.empty:
    st.dataframe(fuel_metrics, use_container_width=True, hide_index=True)

    baseline = fuel_metrics[
        fuel_metrics["model"] == "PersistenceBaseline"
    ]
    if not baseline.empty:
        m1, m2 = st.columns(2)
        m1.metric("Test MAE", f"₹{float(baseline.iloc[0]['test_MAE']):.4f}/L")
        m2.metric("Test RMSE", f"₹{float(baseline.iloc[0]['test_RMSE']):.4f}/L")
else:
    st.warning("No saved metrics were found for this fuel type.")

# ---------------- ACTUAL VS PREDICTED TEST DATA ----------------
st.subheader("Actual vs. predicted on the historical test set")

if {"city", "target_date", target_col, pred_col}.issubset(predictions.columns):
    test_city = predictions[
        predictions["city"] == city
    ].dropna(subset=["target_date", target_col, pred_col]).sort_values(
        "target_date"
    )

    if not test_city.empty:
        test_fig = go.Figure()
        test_fig.add_trace(
            go.Scatter(
                x=test_city["target_date"],
                y=test_city[target_col],
                mode="lines",
                name="Actual",
            )
        )
        test_fig.add_trace(
            go.Scatter(
                x=test_city["target_date"],
                y=test_city[pred_col],
                mode="lines",
                name="Predicted",
                line=dict(dash="dash"),
            )
        )
        test_fig.update_layout(
            title=f"{city} {fuel_type}: held-out test predictions",
            xaxis_title="Date",
            yaxis_title="Price (₹ per litre)",
            hovermode="x unified",
        )
        st.plotly_chart(test_fig, use_container_width=True)

        errors = test_city[pred_col] - test_city[target_col]
        mae = errors.abs().mean()
        rmse = (errors.pow(2).mean()) ** 0.5
        x, y = st.columns(2)
        x.metric("Selected city test MAE", f"₹{mae:.4f}/L")
        y.metric("Selected city test RMSE", f"₹{rmse:.4f}/L")
    else:
        st.warning("No saved test predictions are available for this city.")
else:
    st.warning(
        "The test prediction file has different column names; "
        "check its schema before plotting."
    )

# ---------------- CRUDE OIL ----------------
st.divider()
st.header("Crude oil historical trend")
crude_valid = crude.dropna(subset=["date", "price"]).sort_values("date")

if not crude_valid.empty:
    crude_fig = go.Figure()
    crude_fig.add_trace(
        go.Scatter(
            x=crude_valid["date"],
            y=crude_valid["price"],
            mode="lines",
            name="Crude oil price",
        )
    )
    crude_fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Source price units",
        hovermode="x unified",
    )
    st.plotly_chart(crude_fig, use_container_width=True)
    st.caption(
        "Crude oil is shown in the units supplied by the source CSV; "
        "it is not assumed to be INR per litre."
    )

st.caption(
    "Research limitation: the historical dataset has few price-change "
    "events. A low test error does not guarantee accurate future price "
    "changes, and this app is not connected to a live fuel-price feed."
)
