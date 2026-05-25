import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# 1. Page Configuration
st.set_page_config(page_title="Predictive Analytics Dashboard", layout="wide")
st.title("🔮 Predictive Analytics & Trend Forecasting")
st.markdown("Leverage historical business data to generate data-driven future forecasts.")

# 2. Generate Synthetic Historical Data (3 Years of Daily Data)
@st.cache_data
def generate_historical_data():
    np.random.seed(42)
    # 3 years of daily dates leading up to May 2026
    date_range = pd.date_range(start="2023-05-01", end="2026-05-01", freq="D")
    n_days = len(date_range)
    
    # Components of our time series:
    # Baseline growth trend + Weekly Seasonality (Saturdays are quiet) + Yearly Holiday Spike (Dec) + Random Noise
    baseline = np.linspace(5000, 8500, n_days)
    weekly_pattern = np.array([200, 400, 300, 100, -100, -800, -600]) # Mon-Sun variation
    weekly_seasonality = np.array([weekly_pattern[d.weekday()] for d in date_range])
    
    yearly_seasonality = np.zeros(n_days)
    for i, d in enumerate(date_range):
        if d.month == 12 and d.day > 15: # Holiday rush
            yearly_seasonality[i] = 2500
            
    noise = np.random.normal(0, 300, n_days)
    
    revenue = baseline + weekly_seasonality + yearly_seasonality + noise
    
    df = pd.DataFrame({"Date": date_range, "Revenue": revenue})
    return df

df_raw = generate_historical_data()

# 3. Sidebar - Forecasting Parameters
st.sidebar.header("⏳ Forecast Settings")
forecast_days = st.sidebar.slider("Days to Forecast into Future:", min_value=30, max_value=180, value=90, step=30)
confidence_interval = st.sidebar.slider("Uncertainty Interval (Confidence):", min_value=0.70, max_value=0.95, value=0.80, step=0.05)

# 4. Preparing Data for Prophet
# Prophet strictly requires two columns: 'ds' (datestamp) and 'y' (target metric)
df_prophet = df_raw.rename(columns={"Date": "ds", "Revenue": "y"})

# Split into Train/Test to calculate performance metrics (Holdout method: last 30 days)
train_df = df_prophet.iloc[:-30]
test_df = df_prophet.iloc[-30:]

# 5. Training the Predictive Model
@st.cache_resource
def train_and_forecast(train_data, full_data, periods, interval):
    # Fit model on training data for evaluation
    model_eval = Prophet(interval_width=interval, yearly_seasonality=True, weekly_seasonality=True)
    model_eval.fit(train_data)
    
    # Fit model on full data for actual future prediction
    model_final = Prophet(interval_width=interval, yearly_seasonality=True, weekly_seasonality=True)
    model_final.fit(full_data)
    
    # Generate future dates dataframe
    future_dates = model_final.make_future_dataframe(periods=periods, freq='D')
    forecast = model_final.predict(future_dates)
    
    # Evaluate backtest predictions
    eval_future = model_eval.make_future_dataframe(periods=30, freq='D')
    eval_forecast = model_eval.predict(eval_future)
    
    return forecast, eval_forecast, model_final

forecast, eval_forecast, final_model = train_and_forecast(train_df, df_prophet, forecast_days, confidence_interval)

# 6. Evaluation Metrics Row
st.subheader("🎯 Model Accuracy & Diagnostics")
col1, col2, col3 = st.columns(3)

# Calculate error metrics over the last 30 days of actual data
y_true = test_df['y'].values
y_pred = eval_forecast.iloc[-30:]['yhat'].values

mae = mean_absolute_error(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred) * 100
accuracy_score = 100 - mape

with col1:
    st.metric(label="Mean Absolute Error (MAE)", value=f"${mae:,.2f}", help="Average amount by which predictions miss actual values.")
with col2:
    st.metric(label="Mean Absolute Percentage Error (MAPE)", value=f"{mape:.2f}%", help="Average percentage variance from actuals.")
with col3:
    st.metric(label="Model Predictive Accuracy", value=f"{accuracy_score:.2f}%")

st.markdown("---")

# 7. Interactive Forecast Visualization
st.subheader("📈 Revenue Forecast & Confidence Bands")
st.markdown(f"Displaying historical records alongside the generated **{forecast_days}-day business forecast**.")

fig = go.Figure()

# Plot historical actuals
fig.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines', name='Historical Actuals', line=dict(color='#2b2b2b')))

# Plot future predictions
future_forecast = forecast.iloc[-forecast_days:]
fig.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat'], mode='lines', name='Predicted Forecast', line=dict(color='#00b4d8', width=3)))

# Upper and Lower Confidence Bounds
fig.add_trace(go.Scatter(
    x=future_forecast['ds'].tolist() + future_forecast['ds'].tolist()[::-1],
    y=future_forecast['yhat_upper'].tolist() + future_forecast['yhat_lower'].tolist()[::-1],
    fill='toself',
    fillcolor='rgba(0, 180, 216, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=True,
    name='Uncertainty Interval'
))

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Daily Revenue ($)",
    template="plotly_white",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# 8. Under-the-Hood: Trend Components
st.markdown("---")
st.subheader("🕵️ Trend Breakdown (Decomposition)")
st.markdown("Time-series models work by isolating foundational elements. See what is driving your numbers:")

# Extract underlying trend and weekly cyclic patterns
component_col1, component_col2 = st.columns(2)

with component_col1:
    st.markdown("**Long-Term Macro Trend** (Where the business is heading overall)")
    fig_macro = go.Figure()
    fig_macro.add_trace(go.Scatter(x=forecast['ds'], y=forecast['trend'], line=dict(color='#7209b7')))
    fig_macro.update_layout(template="plotly_white", margin=dict(t=10, b=10))
    st.plotly_chart(fig_macro, use_container_width=True)

with component_col2:
    st.markdown("**Weekly Operations Seasonality** (Impact of the day of the week)")
    # Grouping to show cleanly a representation of a week
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    # Extract structural coefficients from Prophet internal indices 
    weekly_impacts = [forecast[forecast['ds'].dt.day_name() == day]['weekly'].iloc[0] for day in days_of_week]
    
    fig_week = go.Scatter(x=days_of_week, y=weekly_impacts, mode='lines+markers', fill='tozeroy', line=dict(color='#4361ee'))
    fig_week_layout = go.Figure(data=fig_week)
    fig_week_layout.update_layout(template="plotly_white", margin=dict(t=10, b=10))
    st.plotly_chart(fig_week_layout, use_container_width=True)
