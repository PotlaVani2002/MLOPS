import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from datetime import timedelta
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv("db_growth_data.csv", parse_dates=["Date"])

# Get the latest date per server
latest_dates = df.groupby('Server')['Date'].max().reset_index()

# Merge to get the latest records per server
latest_per_server = pd.merge(df, latest_dates, on=['Server', 'Date'], how='inner')

# Sum DB_Size_GB per server for their latest dates
total_usage_per_server = latest_per_server.groupby('Server')['DB_Size_GB'].sum().reset_index()

# Calculate overall total usage
overall_total = total_usage_per_server['DB_Size_GB'].sum()

# Display summary as text
st.title("📊 Current Database Usage Summary")

# Show per-server usage with their latest date
for _, row in total_usage_per_server.iterrows():
    server_date = latest_dates[latest_dates['Server'] == row['Server']]['Date'].values[0]
    st.write(f"Server: **{row['Server']}** (as of {pd.to_datetime(server_date).date()}) — Total DB Size: **{row['DB_Size_GB']:.2f} GB**")

# Show overall total usage
st.write("---")
st.write(f"### Overall Total DB Size Across All Servers: **{overall_total:.2f} GB**")


# Load saved ARIMA models
with open("arima_models.pkl", "rb") as f:
    models = pickle.load(f)

# Load historical data
df = pd.read_csv("db_growth_data.csv", parse_dates=["Date"])
df = df.sort_values(["Server", "Database", "Date"])

st.title("📈 Database Size Growth Forecasting")

# User inputs
months_to_forecast = st.number_input("Months to Forecast", min_value=1, max_value=48, value=6)
chart_type = st.selectbox("Chart Type", ["Line Chart", "Bar Chart"])

# Server selection
server_options = ["All Servers"] + sorted(df["Server"].unique().tolist())
selected_server = st.selectbox("Select Server", server_options)

# Database selection
if selected_server != "All Servers":
    db_options = ["All Databases"] + sorted(df[df["Server"] == selected_server]["Database"].unique().tolist())
else:
    db_options = ["All Databases"]
selected_database = st.selectbox("Select Database", db_options)


def forecast_series(ts, model_key):
    """Return forecast DataFrame given a time series and model key."""
    if model_key in models:
        model_fit = models[model_key]
    else:
        model_fit = ARIMA(ts, order=(1, 1, 1)).fit()

    forecast = model_fit.forecast(steps=months_to_forecast)
    forecast_dates = pd.date_range(ts.index[-1] + timedelta(days=1), periods=months_to_forecast, freq="MS")
    return pd.DataFrame({"Date": forecast_dates, "Predicted_Size_GB": forecast.values})


plot_data = []
all_forecast_rows = []

if selected_server == "All Servers" and selected_database == "All Databases":
    # Aggregate all servers + all dbs
    agg_df = df.resample("MS", on="Date")["DB_Size_GB"].sum().asfreq("MS").fillna(method="ffill")
    forecast_df = forecast_series(agg_df, ("ALL_SERVERS", "ALL_DBS"))
    plot_data.append(("All Servers", "All Databases", agg_df, forecast_df))
    forecast_df["Server"] = "All Servers"
    forecast_df["Database"] = "All Databases"
    all_forecast_rows.append(forecast_df)

elif selected_server != "All Servers" and selected_database == "All Databases":
    # Aggregate all DBs of the selected server
    server_df = df[df["Server"] == selected_server]
    monthly_total = server_df.resample("MS", on="Date")["DB_Size_GB"].sum().asfreq("MS").fillna(method="ffill")
    forecast_df = forecast_series(monthly_total, (selected_server, "ALL_DBS"))
    plot_data.append((selected_server, "All Databases", monthly_total, forecast_df))
    forecast_df["Server"] = selected_server
    forecast_df["Database"] = "All Databases"
    all_forecast_rows.append(forecast_df)

else:
    # Single server + single DB
    db_df = df[(df["Server"] == selected_server) & (df["Database"] == selected_database)]
    ts = db_df.resample("MS", on="Date")["DB_Size_GB"].last().asfreq("MS").fillna(method="ffill")
    forecast_df = forecast_series(ts, (selected_server, selected_database))
    plot_data.append((selected_server, selected_database, ts, forecast_df))
    forecast_df["Server"] = selected_server
    forecast_df["Database"] = selected_database
    all_forecast_rows.append(forecast_df)


# Plotting
fig, ax = plt.subplots(figsize=(12, 6))
colors = plt.cm.tab10.colors

for idx, (server, db, ts, forecast_df) in enumerate(plot_data):
    color = colors[idx % len(colors)]

    if chart_type == "Line Chart":
        # Plot historical data solid line
        ax.plot(ts.index, ts.values, label=f"{server} ({db}) - Historical", color="blue", linestyle="-")
        # Plot forecast data as continuation with dashed line (no scatter/dots)
        ax.plot(forecast_df["Date"], forecast_df["Predicted_Size_GB"], label=f"{server} ({db}) - Forecast", color="red", linestyle="--",linewidth=2)
    else:
        # For bar chart, plot historical and forecast side-by-side with different transparency
        ax.bar(ts.index, ts.values, label=f"{server} ({db}) - Historical", color="blue", alpha=0.6)
        ax.bar(forecast_df["Date"], forecast_df["Predicted_Size_GB"], label=f"{server} ({db}) - Forecast", color="red", alpha=0.9,width=5)

ax.set_title("Database Growth Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Database Size (GB)")
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)

# Show combined forecast table with server, database, date, predicted values
if all_forecast_rows:
    combined_forecast_df = pd.concat(all_forecast_rows)[["Server", "Database", "Date", "Predicted_Size_GB"]]
    st.write("📅 **Complete Forecast Table**")
    st.dataframe(combined_forecast_df.reset_index(drop=True))
