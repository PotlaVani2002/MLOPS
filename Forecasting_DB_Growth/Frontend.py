import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from statsmodels.tsa.arima.model import ARIMA
import logging

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

st.set_page_config(page_title="DB Growth Forecast", layout="wide")

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("db_growth_data.csv", parse_dates=["Date"])
        return df
    except FileNotFoundError:
        st.error("CSV file not found. Ensure 'db_growth_data.csv' is present.")
        st.stop()

df = load_data()

# -------------------------------
# Summary Section
# -------------------------------
st.title("📊 Current Database Usage Summary")

latest = df.groupby('Server')['Date'].max().reset_index()
latest_data = pd.merge(df, latest, on=['Server', 'Date'], how='inner')
server_sizes = latest_data.groupby('Server')['DB_Size_GB'].sum().reset_index()
total_size = server_sizes['DB_Size_GB'].sum()

for _, row in server_sizes.iterrows():
    st.write(f"Server: **{row['Server']}** — Total DB Size: **{row['DB_Size_GB']:.2f} GB**")

st.write(f"### Total Across All Servers: **{total_size:.2f} GB**")
st.markdown("---")

# -------------------------------
# Inputs
# -------------------------------
st.title("📈 Forecast DB Growth with ARIMA")
months_to_forecast = st.slider("Months to Forecast", min_value=1, max_value=36, value=12)
chart_type = st.selectbox("Chart Type", ["Line Chart", "Bar Chart"])

server_list = ["All Servers"] + sorted(df["Server"].unique())
selected_server = st.selectbox("Select Server", server_list)

if selected_server != "All Servers":
    db_list = ["All Databases"] + sorted(df[df["Server"] == selected_server]["Database"].unique())
else:
    db_list = ["All Databases"]

selected_database = st.selectbox("Select Database", db_list)

capacity_limit = st.slider("⚠️ Capacity Limit (GB)", min_value=10, max_value=1000, value=500)

# -------------------------------
# ARIMA Forecast Function
# -------------------------------
def forecast_arima(ts, periods):
    model = ARIMA(ts, order=(1, 1, 1))
    model_fit = model.fit()
    forecast_result = model_fit.get_forecast(steps=periods)
    forecast_mean = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()
    future_dates = pd.date_range(ts.index[-1] + timedelta(days=1), periods=periods, freq="MS")
    return pd.DataFrame({
        "Date": future_dates,
        "Forecast_GB": forecast_mean.values,
        "Lower_Bound": conf_int.iloc[:, 0].values,
        "Upper_Bound": conf_int.iloc[:, 1].values
    })

# -------------------------------
# Trigger Forecast
# -------------------------------
if st.button("🔮 Generate Forecast"):

    if selected_server == "All Servers" and selected_database == "All Databases":
        ts = df.resample("MS", on="Date")["DB_Size_GB"].sum().asfreq("MS").fillna(method="ffill")
        title = "All Servers - All Databases"
    elif selected_server != "All Servers" and selected_database == "All Databases":
        filtered = df[df["Server"] == selected_server]
        ts = filtered.resample("MS", on="Date")["DB_Size_GB"].sum().asfreq("MS").fillna(method="ffill")
        title = f"{selected_server} - All Databases"
    else:
        filtered = df[(df["Server"] == selected_server) & (df["Database"] == selected_database)]
        ts = filtered.resample("MS", on="Date")["DB_Size_GB"].last().asfreq("MS").fillna(method="ffill")
        title = f"{selected_server} - {selected_database}"

    # Generate forecast
    forecast_df = forecast_arima(ts, months_to_forecast)

    # -------------------------------
    # Plotting
    # -------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ts.index, ts.values, label="Historical", color="blue", linewidth=2)
    ax.plot(forecast_df["Date"], forecast_df["Forecast_GB"], label="Forecast", color="red", linestyle="--", linewidth=2)
    ax.fill_between(forecast_df["Date"], forecast_df["Lower_Bound"], forecast_df["Upper_Bound"],
                    color='red', alpha=0.2, label="Confidence Interval")

    # Capacity limit line
    ax.axhline(y=capacity_limit, color='black', linestyle='--', label=f"Capacity Limit ({capacity_limit} GB)")

    # Check if forecast exceeds capacity
    if (forecast_df["Forecast_GB"] > capacity_limit).any():
        st.warning("⚠️ Forecast exceeds the capacity limit. Consider scaling or optimizing storage.")

    ax.set_title(f"Forecast: {title}")
    ax.set_xlabel("Date")
    ax.set_ylabel("DB Size (GB)")
    ax.legend()
    plt.xticks(rotation=45)

    st.pyplot(fig)

    # -------------------------------
    # Forecast Table
    # -------------------------------
    st.subheader("📅 Forecast Table")
    st.dataframe(forecast_df)
