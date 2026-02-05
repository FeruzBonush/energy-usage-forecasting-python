# Energy Usage Forecasting with Python

This project focuses on forecasting household electricity consumption using historical smart meter data. Accurate energy demand forecasting is critical for energy providers and building managers to optimize capacity planning, reduce peak load costs, and support sustainable energy usage.

## 🔍 Project Overview

Using several years of minute-level electricity consumption data, the project:

- Aggregates raw measurements into **hourly** energy usage
- Performs **time series feature engineering** (time, lag, rolling features)
- Trains a **Random Forest regression model**
- Compares the model against a **naive baseline** (previous hour’s value)
- Evaluates performance using **MAE** and **RMSE**
- Visualises actual vs predicted energy usage over time

## 🧾 Dataset

This project uses the **Individual Household Electric Power Consumption** dataset from the UCI Machine Learning Repository.

The dataset is **not included** in this repository due to its size.

To run the notebook:

1. Download the dataset from the UCI repository  
2. Extract `household_power_consumption.txt`
3. Place it in the `data/` folder

Expected path:

```text
data/household_power_consumption.txt

