from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Fuel Price Prediction System",
    page_icon="⛽",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Data"
MODEL_DIR = ROOT / "models"

@st.cache_data
def load_data():
    fuel = pd.read_csv(DATA_DIR / "india_fuel_prices_daily.csv")
    crude = pd.read_csv(DATA_DIR / "crude-oil-price.csv")
    metrics = pd.read_csv(MODEL_DIR / "model_comparison_metrics.csv")
    predictions = pd.read_csv(MODEL_DIR / "test_predictions.csv")

    fuel["date"] = pd.to_datetime(fuel["date"], errors="coerce")
    crude["date"] = pd.to_datetime(crude["date"], errors="coerce", utc=True)
    crude["date"] = crude["date"].dt.tz_localize(None)
    predictions["date"] = pd.to_datetime(predictions["date"], errors="coerce")
    predictions["target_date"] = pd.to_datetime(
        predictions["target_date"], errors="coerce"
    )

    # Apply the same in-memory correction used during model development.
    # The original CSV is never changed.
    correction = (
        (fuel["date"] >= "2022-06-01")
        & (fuel["date"] <= "2022-06-30")
        & (fuel["city"].isin(["Chennai", "Delhi", "Mumbai", "Kolkata"]))
    )
    petrol_values = fuel.loc[
        correction, "petrol_price_inr_per_litre"
    ].copy()
    fuel.loc[correction, "petrol_price_inr_per_litre"] = fuel.loc[
        correction, "diesel_price_inr_per_litre"
    ].values
    fuel.loc[correction, "diesel_price_inr_per_litre"] = petrol_values.values

    return fuel, crude, metrics, predictions

try:
    fuel, crude, metrics, predictions = load_data()
except Exception as exc:
    st.error(f"Unable to load project files: {exc}")
    st.stop()

st.title("⛽ Fuel Price Prediction System")
st.write(
    "Explore historical petrol and diesel prices and view a next-day "
    "estimate based on the latest available observation in the dataset."
)

st.info(
    "Dataset limitation: these predictions are based on historical data, "
    "not live fuel prices. The latest recorded date is shown below."
)

cities = ["Chennai", "Delhi", "Mumbai", "Kolkata"]
city = st.sidebar.selectbox("Select city", cities)
fuel_type = st.sidebar.radio("Select fuel type", ["Petrol", "Diesel"])

price_column = (
    "petrol_price_inr_per_litre"
    if fuel_type == "Petrol"
    else "diesel_price_inr_per_litre"
)

city_data = (
    fuel.loc[
        (fuel["city"] == city) & fuel[price_column].notna(),
        ["date", price_column],
    ]
    .dropna()
    .sort_values("date")
    .drop_duplicates("date", keep="last")
)

if city_data.empty:
    st.error("No price data is available for this selection.")
    st.stop()

latest = city_data.iloc[-1]
latest_date = latest["date"]
latest_price = float(latest[price_column])
prediction_date = latest_date + pd.Timedelta(days=1)

# The selected model is the persistence baseline: tomorrow's estimate
# equals the latest recorded price.
predicted_price = latest_price

st.subheader(f"{city} — {fuel_type} overview")

col1, col2, col3 = st.columns(3)
col1.metric("Latest recorded price", f"₹{latest_price:.2f}/L")
col2.metric("Next-day estimate", f"₹{predicted_price:.2f}/L")
col3.metric("Latest data date", latest_date.strftime("%d %b %Y"))

st.caption(
    f"Estimated date: {prediction_date.strftime('%d %b %Y')}. "
    "The estimate assumes no price change and is not a live market forecast."
)

st.divider()
st.subheader("Historical fuel price")

fig = px.line(
    city_data,
    x="date",
    y=price_column,
    title=f"{city} {fuel_type.lower()} price history",
    labels={
        "date": "Date",
        price_column: "Price (₹ per litre)",
    },
)
fig.update_layout(hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Recent price records")
recent = city_data.tail(15).sort_values("date", ascending=False).copy()
recent["date"] = recent["date"].dt.strftime("%Y-%m-%d")
recent = recent.rename(
    columns={price_column: f"{fuel_type} price (₹/L)"}
)
st.dataframe(recent, use_container_width=True, hide_index=True)

# --------------------------------------------------
# Crude oil history
# --------------------------------------------------
st.divider()
st.subheader("Crude oil price history")

crude_plot = crude.dropna(subset=["date", "price"]).sort_values("date")
if not crude_plot.empty:
    crude_fig = px.line(
        crude_plot,
        x="date",
        y="price",
        title="Historical crude oil prices from the supplied CSV",
        labels={"date": "Date", "price": "Price (source units)"},
    )
    st.plotly_chart(crude_fig, use_container_width=True)
    st.caption(
        "Crude oil is shown as a historical supporting indicator. "
        "Its source units are not assumed to be INR per litre."
    )
else:
    st.warning("No usable crude oil observations were found.")

# --------------------------------------------------
# Model evaluation
# --------------------------------------------------
st.divider()
st.subheader("Model evaluation")

if {"fuel", "model", "validation_MAE", "test_MAE"}.issubset(metrics.columns):
    st.write(
        "MAE measures the average absolute difference between actual and "
        "predicted prices. Lower is better."
    )
    st.dataframe(metrics, use_container_width=True, hide_index=True)

    selected_metrics = metrics[
        metrics["fuel"].astype(str).str.lower() == fuel_type.lower()
    ]
    if not selected_metrics.empty:
        st.write(f"**Evaluation for {fuel_type.lower()}:**")
        st.dataframe(
            selected_metrics,
            use_container_width=True,
            hide_index=True,
        )
else:
    st.warning("The model comparison file does not contain the expected columns.")

st.subheader("Held-out test predictions")
st.caption(
    "These are saved historical test predictions, not predictions for today's "
    "market. The test set contains very few actual price-change events."
)
if not predictions.empty:
    if "city" in predictions.columns:
        displayed_predictions = predictions[
            predictions["city"] == city
        ].copy()
    else:
        displayed_predictions = predictions.copy()

    st.dataframe(
        displayed_predictions.tail(30),
        use_container_width=True,
        hide_index=True,
    )

st.divider()
st.caption(
    "Project note: the selected persistence baseline carries forward the "
    "last recorded price. Validation data contained no price changes, so "
    "perfect validation error does not establish reliable performance "
    "during future price changes."
)
