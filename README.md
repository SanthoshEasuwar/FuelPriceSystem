# Fuel Price Prediction System

A Streamlit dashboard for exploring historical petrol, diesel, and crude oil
prices using the supplied CSV files.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Important limitation

The next-day estimate uses a persistence baseline: the latest recorded price
is carried forward unchanged. The dashboard is not connected to a live fuel
price API. The supplied fuel dataset ends on 17 May 2026, so the displayed
estimate is for the day after the last available observation for the selected
city.

The supplied crude oil data is shown as a historical supporting indicator.
