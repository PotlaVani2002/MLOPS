import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from statsmodels.tsa.arima.model import ARIMA
import pyodbc
import logging

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

st.set_page_config(page_title="DB Growth Forecast", layout="wide")

# -------------------------------
# SQL Server: Load Data
# -------------------------------
@st.cache_data
def load_data_from_sql():
    server = '172.17.35.52'
    database = 'AdventureWorks2022'
    username = 'mlops'
    password = 'mlops2k25'

    try:
        conn = pyodbc.connect(
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"UID={username};"
            f"PWD={password}"
        )

        query = """
        SELECT
            ServerName AS [Server],
            DatabaseName AS [Database],
            RecordDate AS [Date],
            SizeMB
        FROM dbo.database_info_tb
        """
        df = pd.read_sql(query, conn)
        df['Date'] = pd.to_datetime(df['Date'])
        df['DB_Size_GB'] = df['SizeMB'] / 1024  # Convert MB to GB
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error connecting to SQL Server: {e}")
        st.stop()

# Load data
df = load_data_from_sql()

# -------------------------------
# Current Usage Summary
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
# User Inputs
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

    forecast_df = forecast_arima(ts, months_to_forecast)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ts.index, ts.values, label="Historical", color="blue")
    ax.plot(forecast_df["Date"], forecast_df["Forecast_GB"], label="Forecast", color="red", linestyle="--")
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

    # Forecast Table
    st.subheader("📅 Forecast Table")
    st.dataframe(forecast_df)
