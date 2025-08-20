import streamlit as st
import pandas as pd
import pyodbc
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.express as px
import warnings

warnings.filterwarnings("ignore")

# ---------------------------
# Load Data from SQL Server
# ---------------------------
@st.cache_data
def load_growth_percentage():
    conn_str = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=172.17.35.52;"
        "DATABASE=AdventureWorks2022;"
        "UID=mlops;"
        "PWD=mlops2k25;"
    )
    conn = pyodbc.connect(conn_str)

    query = """  select b.servername,b.databasename,b.used, b.yr,b.mn,((b.used*1.0)/70000)* 100 as per from
    (SELECT a.SERVERNAME,a.databasename, SUM(a.used)as used, a.yr,a.mn from  
  (  select servername,databasename,year(recorddate)as yr  ,month(recorddate) as mn, sum(growthmb) as used
  from db_info_tb
  group by servername,databasename,year(recorddate)  ,month(recorddate))a
  group by a.SERVERNAME,a.databasename, a.yr,a.mn)b
  group by b.servername,b.databasename,b.used, b.yr,b.mn
  order by b.servername,b.databasename, b.yr,b.mn
  ;

    """
    df = pd.read_sql(query, conn)
    conn.close()

    df["Date"] = pd.to_datetime(df["yr"].astype(str) + "-" + df["mn"].astype(str) + "-01")
    return df

# ---------------------------
# ARIMA Model Selection
# ---------------------------
def select_best_arima(train, p_range=(0, 6), d_range=(0, 2), q_range=(0, 6)):
    best_aic = float("inf")
    best_order = None
    best_model = None
    for p in range(p_range[0], p_range[1] + 1):
        for d in range(d_range[0], d_range[1] + 1):
            for q in range(q_range[0], q_range[1] + 1):
                try:
                    model = ARIMA(train, order=(p, d, q))
                    results = model.fit()
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_order = (p, d, q)
                        best_model = results
                except:
                    continue
    return best_order, best_model

# ---------------------------
# SARIMA Model Selection
# ---------------------------
def select_best_sarima(train, 
                       p_range=(0, 2), d_range=(0, 1), q_range=(0, 2),
                       P_range=(0, 2), D_range=(0, 1), Q_range=(0, 2), s=12):
    best_aic = float("inf")
    best_order = None
    best_seasonal_order = None
    best_model = None

    for p in range(p_range[0], p_range[1] + 1):
        for d in range(d_range[0], d_range[1] + 1):
            for q in range(q_range[0], q_range[1] + 1):
                for P in range(P_range[0], P_range[1] + 1):
                    for D in range(D_range[0], D_range[1] + 1):
                        for Q in range(Q_range[0], Q_range[1] + 1):
                            try:
                                model = SARIMAX(
                                    train,
                                    order=(p, d, q),
                                    seasonal_order=(P, D, Q, s),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False
                                )
                                results = model.fit(disp=False)
                                if results.aic < best_aic:
                                    best_aic = results.aic
                                    best_order = (p, d, q)
                                    best_seasonal_order = (P, D, Q, s)
                                    best_model = results
                            except:
                                continue
    return best_order, best_seasonal_order, best_model

# ---------------------------
# Streamlit UI
# ---------------------------
df = load_growth_percentage()

servers = ["All Servers"] + sorted(df['servername'].unique())
selected_server = st.sidebar.selectbox("Select Server", servers)

if selected_server != "All Servers":
    db_list = df[df["servername"] == selected_server]['databasename'].unique()
else:
    db_list = df['databasename'].unique()

databases = ["All Databases"] + sorted(db_list)
selected_db = st.sidebar.selectbox("Select Database", databases)

forecast_months = st.sidebar.number_input("Months to Forecast", min_value=1, value=6, step=1)

selected_model = st.sidebar.selectbox("Select Model", ["ARIMA", "SARIMA"])
chart_type = st.sidebar.selectbox("Select Chart Type", ["Line Chart", "Bar Chart"])

if selected_server != "All Servers":
    df = df[df["servername"] == selected_server]

if selected_db != "All Databases":
    df = df[df["databasename"] == selected_db]

st.subheader("Database Growth Forecast")

metrics_list = []
plot_data = pd.DataFrame()

# ---------------------------
# Forecasting Logic
# ---------------------------
def forecast_group(group, server_name, db_name):
    ts = group.set_index("Date")["per"].asfreq("MS").fillna(method="ffill")
    if len(ts) < 8:
        st.warning(f"Not enough data for {server_name} | {db_name} (need >= 8 months).")
        return pd.DataFrame()
    
    train_size = max(int(len(ts) * 0.8), 6)
    train, test = ts.iloc[:train_size], ts.iloc[train_size:]
    
    try:
        if selected_model == "ARIMA":
            best_order, model_fit = select_best_arima(train)
            if best_order is None:
                raise ValueError("No valid ARIMA model found.")
            pred = model_fit.forecast(steps=len(test))
            final_fit = ARIMA(ts, order=best_order).fit()
        else:
            best_order, best_seasonal_order, model_fit = select_best_sarima(train)
            if best_order is None or best_seasonal_order is None:
                raise ValueError("No valid SARIMA model found.")
            pred = model_fit.forecast(steps=len(test))
            final_fit = SARIMAX(ts, order=best_order, seasonal_order=best_seasonal_order).fit()

        
        # metrics
        y_true, y_pred = np.asarray(test, dtype=float), np.asarray(pred, dtype=float)
        mse = mean_squared_error(y_true, y_pred)
        metrics_list.append({
            "RMSE": round(np.sqrt(mse), 3),
            "MAE": round(mean_absolute_error(y_true, y_pred), 3),
            "MAPE (%)": round(np.mean(np.abs((y_true - y_pred) / np.where(np.abs(y_true) < 1e-8, 1e-8, np.abs(y_true)))) * 100, 3),
            "Model": selected_model
        })
        
        hist_df = group.copy()
        hist_df["Type"] = "Historical"
        
        # forecast
        future = final_fit.forecast(steps=forecast_months)
        future_idx = pd.date_range(ts.index[-1] + pd.offsets.MonthBegin(1),
                                   periods=forecast_months, freq="MS")
        forecast_df = pd.DataFrame({
            "Date": future_idx,
            "per": future.values,
            "servername": server_name,
            "databasename": db_name,
            "Type": "Forecast"
        })
        forecast_df["yr"] = forecast_df["Date"].dt.year
        forecast_df["mn"] = forecast_df["Date"].dt.month
        
        # calculate used from growth %
        last_used = group["used"].iloc[-1]
        used_forecast, current_used = [], last_used
        for perc in forecast_df["per"]:
            growth_value = (perc / 100.0) * current_used
            current_used += growth_value
            used_forecast.append(current_used)
        forecast_df["used"] = used_forecast
        
        combined = pd.concat([hist_df, forecast_df])
        return combined
    except Exception as e:
        st.warning(f"Could not build {selected_model} model for {server_name} | {db_name}: {e}")
        return pd.DataFrame()

if selected_db == "All Databases":
    group = df.groupby("Date", as_index=False)[["per", "used"]].sum()
    group["servername"] = selected_server if selected_server != "All Servers" else "All Servers"
    group["databasename"] = "All Databases"
    plot_data = forecast_group(group, selected_server if selected_server != "All Servers" else "All Servers", "All Databases")
else:
    for (server, db), group in df.groupby(["servername", "databasename"]):
        group = group.sort_values("Date")
        plot_data = pd.concat([plot_data, forecast_group(group, server, db)])

if not plot_data.empty:
    # Add display column for MB/GB in hover
    plot_data["used_display"] = plot_data["used"].apply(lambda x: f"{x:,.2f} MB" if x < 1024 else f"{x/1024:,.2f} GB")
    
    plot_data["Label"] = plot_data["servername"] + " | " + plot_data["databasename"]

# ---------------------------
# Dynamic Capacity Selection
    # ---------------------------
    if selected_server == "All Servers" and selected_db == "All Databases":
        server_capacity = 4   # overall server capacity
    else:
        server_capacity = 2    # per-database capacity (percentage)

    
    if chart_type == "Line Chart":
        fig = px.line(
            plot_data,
            x="Date",
            y="per",
            color="Type",
            line_group="Label",
            markers=True,
            hover_data={
                "servername": True,
                "databasename": True,
                "per": ':.2f',
                "used_display": True,
                "Type": True,
            },
            title="Server-Database Growth Forecast",
            color_discrete_map={"Historical": "#1f77b4", "Forecast": "#ff7f0e"}
        )
    else:
        fig = px.bar(
            plot_data,
            x="Date",
            y="per",
            color="Type",
            barmode="group",
            hover_data={
                "servername": True,
                "databasename": True,
                "per": ':.2f',
                "used_display": True,
                "Type": True,
            },
            title="Server-Database Growth Forecast",
            color_discrete_map={"Historical": "#1f77b4", "Forecast": "#ff7f0e"}
        )
    
    fig.add_hline(y=0, line_color="black")
    fig.add_vline(x=plot_data["Date"].min(), line_color="black")
    fig.add_trace(
        px.line(
            x=[plot_data["Date"].min(), plot_data["Date"].max()],
            y=[server_capacity, server_capacity],
        ).data[0]
    )

    # style it like the other traces
    fig.data[-1].update(
        name="DB Limit",
        mode="lines",
        line=dict(color="red", dash="dot", width=2),
        showlegend=True
    )
    min_date = plot_data["Date"].min()
    max_date = plot_data["Date"].max()
    total_days = (max_date - min_date).days
    pad = pd.Timedelta(days=int(total_days * 0.05))
    fig.update_xaxes(range=[min_date - pad, max_date + pad])
    # ---------------------------
    # Dynamic Y-axis Tick Interval
    # ---------------------------
    if selected_server == "All Servers" and selected_db == "All Databases":
        y_tick_interval = 0.5
    else:
        y_tick_interval = 0.5

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Growth Percentage (%)",
        hovermode="x unified",
        yaxis=dict(dtick=y_tick_interval)  # 👈 set y-axis difference
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Metrics Table
# ---------------------------
st.subheader("Forecast Model Metrics")
if metrics_list:
    metrics_df = pd.DataFrame(metrics_list)[["Model", "RMSE", "MAE", "MAPE (%)"]]
    st.dataframe(metrics_df, use_container_width=True)
    st.markdown("""
    **Metric Definitions:**
    - **RMSE (Root Mean Square Error):** Square root of average squared differences between actual and predicted values.
    - **MAE (Mean Absolute Error):** Average of absolute differences between actual and predicted values.
    - **MAPE (Mean Absolute Percentage Error):** Average percentage error between actual and predicted values.
    """)
else:
    st.info("No metrics to display (insufficient data).")

# ---------------------------
# Raw + Predicted Data
# ---------------------------
st.subheader("Raw + Predicted Data")
raw_df = df.copy()
raw_df["Type"] = "Historical"
pred_df = plot_data[plot_data["Type"] == "Forecast"].copy()
combined_df = pd.concat([raw_df, pred_df], ignore_index=True)
combined_df = combined_df.sort_values(["servername", "databasename", "Date"])

# Remove Year-Month column if exists
if "Year-Month" in combined_df.columns:
    combined_df = combined_df.drop(columns=["Year-Month"])

st.dataframe(combined_df, use_container_width=True)
