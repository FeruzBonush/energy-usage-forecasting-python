import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
from pathlib import Path


@st.cache_data
def load_raw_data(filepath: str):
    """
    Load the original minute-level dataset.
    """
    df = pd.read_csv(
        filepath,
        sep=';',
        na_values='?',
        low_memory=False,
        parse_dates={'datetime': ['Date', 'Time']},
        dayfirst=True,
    )
    df = df.dropna(subset=['Global_active_power'])
    df = df.set_index('datetime').sort_index()
    return df


@st.cache_data
def prepare_hourly_data(df: pd.DataFrame):
    """
    Resample to hourly and create feature dataframe.
    """
    hourly = df['Global_active_power'].resample('H').mean()
    hourly_df = pd.DataFrame({'Global_active_power': hourly}).dropna()

    data = hourly_df.copy()
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    data['month'] = data.index.month

    data['lag_1'] = data['Global_active_power'].shift(1)
    data['lag_2'] = data['Global_active_power'].shift(2)
    data['lag_24'] = data['Global_active_power'].shift(24)
    data['rolling_mean_24'] = data['Global_active_power'].rolling(window=24).mean()

    data = data.dropna()

    feature_cols = [
        'hour', 'day_of_week', 'month',
        'lag_1', 'lag_2', 'lag_24',
        'rolling_mean_24'
    ]

    X = data[feature_cols]
    y = data['Global_active_power']

    return data, X, y, feature_cols


@st.cache_resource
def train_model(X, y, split_date="2010-09-01"):
    """
    Time-based split and Random Forest training.
    """
    X_train = X[X.index < split_date]
    X_test = X[X.index >= split_date]
    y_train = y[y.index < split_date]
    y_test = y[y.index >= split_date]

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    baseline_pred = X_test['lag_1']
    model_pred = model.predict(X_test)

    return X_train, X_test, y_train, y_test, baseline_pred, model_pred, model


def print_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def main():
    st.title("🔋 Energy Usage Forecasting Dashboard")
    st.write(
        "This app uses historical smart meter data to forecast "
        "hourly household electricity consumption."
    )

    data_path = Path("data/household_power_consumption.txt")

    if not data_path.exists():
        st.error(
            "Dataset not found.\n\n"
            "Please download `household_power_consumption.txt` and place it in the `data/` folder."
        )
        st.stop()

    with st.spinner("Loading and preparing data..."):
        raw_df = load_raw_data(str(data_path))
        data, X, y, feature_cols = prepare_hourly_data(raw_df)
        X_train, X_test, y_train, y_test, baseline_pred, model_pred, model = train_model(X, y)

    st.subheader("1. Data Overview")
    st.write("**Date range (hourly data):**", data.index.min(), "to", data.index.max())
    st.write("**Total hours:**", len(data))

    st.line_chart(data['Global_active_power'].rename("Global_active_power (kW)"))

    st.subheader("2. Model Performance")

    baseline_mae, baseline_rmse = print_metrics(y_test, baseline_pred)
    model_mae, model_rmse = print_metrics(y_test, model_pred)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Baseline (lag_1)**")
        st.write(f"MAE: {baseline_mae:.4f}")
        st.write(f"RMSE: {baseline_rmse:.4f}")
    with col2:
        st.markdown("**Random Forest**")
        st.write(f"MAE: {model_mae:.4f}")
        st.write(f"RMSE: {model_rmse:.4f}")

    st.subheader("3. Actual vs Predicted (Test Period)")

    results = pd.DataFrame({
        'Actual': y_test,
        'Baseline': baseline_pred,
        'RandomForest': model_pred
    }, index=y_test.index)

    # Select a date range to visualise
    min_date = results.index.min()
    max_date = results.index.max()

    st.write("Select a date range from the test period to visualise predictions:")

    date_range = st.slider(
        "Date range",
        min_value=min_date.to_pydatetime(),
        max_value=max_date.to_pydatetime(),
        value=(min_date.to_pydatetime(), (min_date + pd.Timedelta(days=7)).to_pydatetime()),
        format="YYYY-MM-DD"
    )

    start, end = date_range
    mask = (results.index >= start) & (results.index <= end)
    plot_data = results.loc[mask]

    st.line_chart(plot_data)

    st.caption(
        "This dashboard trains a Random Forest model on historical hourly energy usage and "
        "compares it to a naive baseline that simply uses the previous hour's value."
    )


if __name__ == "__main__":
    main()
