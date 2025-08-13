import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from datetime import timedelta
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="DB Growth Forecast", layout="wide")

# -------------------------------
# Load data
# -------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("db_growth_data.csv", parse_dates=["Date"])
        return df
    except FileNotFoundError:
        st.error("CSV file not found! Make sure db_growth_data.csv is in the app folder.")
        st.stop()  # Stop execution if file is missing


df = load_data()

# -------------------------------
# Current DB Usage Summary
# -------------------------------
st.title("📊 Current Database Usage Summary")

latest_dates = df.groupby('Server')['Date'].max().reset_index()
latest_per_server = pd.merge(df, latest_dates, on=['Server', 'Date'], how='inner')
total_usage_per_server = latest_per_server.groupby('Server')['DB_Size_GB'].sum().reset_index()
overall_total = total_usage_per_server['DB_Size_GB'].sum()

for _, row in total_usage_per_server.iterrows():
    server_date = latest_dates[latest_dates['Server'] == row['Server']]['Date'].values[0]
    st.write(f"Server: **{row['Server']}** (as of {pd.to_datetime(server_date).date()}) — Total DB Size: **{row['DB_Size_GB']:.2f} GB**")

st.write("---")
st.write(f"### Overall Total DB Size Across All Servers: **{overall_total:.2f} GB**")

# -------------------------------
# Load ARIMA models
# -------------------------------
@st.cache_resource
def load_models():
    try:
        with open("arima_models.pkl", "rb") as f:
            models = pickle.load(f)
    except FileNotFoundError:
        st.warning("ARIMA models pickle file not found! Forecasting will fit new models on the fly.")
        models = {}
    return models  # <-- This was missing


models = load_models()

# -------------------------------
# Forecasting Section
# -------------------------------
st.title("📈 Database Size Growth Forecasting")

months_to_forecast = st.number_input("Months to Forecast", min_value=1, max_value=48, value=6)
chart_type = st.selectbox("Chart Type", ["Line Chart", "Bar Chart"])

server_options = ["All Servers"] + sorted(df["Server"].unique().tolist())
selected_server = st.selectbox("Select Server", server_options)

if selected_server != "All Servers":
    db_options = ["All Databases"] + sorted(df[df["Server"] == selected_server]["Database"].unique().tolist())
else:
    db_options = ["All Databases"]
selected_database = st.selectbox("Select Database", db_options)

# -------------------------------
# Forecast function
# -------------------------------
def forecast_series(ts, model_key):
    if model_key in models:
        model_fit = models[model_key]
    else:
        model_fit = ARIMA(ts, order=(1, 1, 1)).fit()
    forecast = model_fit.forecast(steps=months_to_forecast)
    forecast_dates = pd.date_range(ts.index[-1] + timedelta(days=1), periods=months_to_forecast, freq="MS")
    return pd.DataFrame({"Date": forecast_dates, "Predicted_Size_GB": forecast.values})

# -------------------------------
# Prepare data for plotting
# -------------------------------
plot_data = []
all_forecast_rows = []

def prepare_forecast(ts, server_name, db_name):
    forecast_df = forecast_series(ts, (server_name, db_name))
    plot_data.append((server_name, db_name, ts, forecast_df))
    forecast_df["Server"] = server_name
    forecast_df["Database"] = db_name
    all_forecast_rows.append(forecast_df)

if st.button("Generate Forecast"):
    if selected_server == "All Servers" and selected_database == "All Databases":
        agg_df = df.resample("MS", on="Date")["DB_Size_GB"].sum().asfreq("MS").fillna(method="ffill")
        prepare_forecast(agg_df, "All Servers", "All Databases")
    elif selected_server != "All Servers" and selected_database == "All Databases":
        server_df = df[df["Server"] == selected_server]
        monthly_total = server_df.resample("MS", on="Date")["DB_Size_GB"].sum().asfreq("MS").fillna(method="ffill")
        prepare_forecast(monthly_total, selected_server, "All Databases")
    else:
        db_df = df[(df["Server"] == selected_server) & (df["Database"] == selected_database)]
        ts = db_df.resample("MS", on="Date")["DB_Size_GB"].last().asfreq("MS").fillna(method="ffill")
        prepare_forecast(ts, selected_server, selected_database)

    # -------------------------------
    # Plotting
    # -------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10.colors

    for idx, (server, db, ts, forecast_df) in enumerate(plot_data):
        color = colors[idx % len(colors)]
        if chart_type == "Line Chart":
            ax.plot(ts.index, ts.values, label=f"{server} ({db}) - Historical", color="blue", linestyle="-")
            ax.plot(forecast_df["Date"], forecast_df["Predicted_Size_GB"], label=f"{server} ({db}) - Forecast", color="red", linestyle="--", linewidth=2)
        else:
            ax.bar(ts.index, ts.values, label=f"{server} ({db}) - Historical", color="blue", alpha=0.6)
            ax.bar(forecast_df["Date"], forecast_df["Predicted_Size_GB"], label=f"{server} ({db}) - Forecast", color="red", alpha=0.9, width=5)

    ax.set_title("Database Growth Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Database Size (GB)")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    if all_forecast_rows:
        combined_forecast_df = pd.concat(all_forecast_rows)[["Server", "Database", "Date", "Predicted_Size_GB"]]
        st.write("📅 **Complete Forecast Table**")
        st.dataframe(combined_forecast_df.reset_index(drop=True))
