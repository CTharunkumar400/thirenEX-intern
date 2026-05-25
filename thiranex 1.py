import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Sales & Revenue Dashboard", layout="wide")
st.title("📊 Sales & Revenue Analysis Dashboard")
st.markdown("Gain actionable insights into your business performance.")

# 2. Helper Function to Load Mock Data (If no file is uploaded)
@st.cache_data
def load_mock_data():
    np.random.seed(42)
    dates = pd.date_range(start="2025-01-01", end="2025-12-31", freq="D")
    products = ["Laptop", "Smartphone", "Tablet", "Smartwatch", "Headphones"]
    regions = ["North", "South", "East", "West"]
    
    mock_data = pd.DataFrame({
        "Date": np.random.choice(dates, size=500),
        "Product": np.random.choice(products, size=500),
        "Region": np.random.choice(regions, size=500),
        "Units Sold": np.random.randint(1, 10, size=500),
        "Unit Price": np.random.choice([1200, 800, 400, 250, 150], size=500)
    })
    mock_data["Revenue"] = mock_data["Units Sold"] * mock_data["Unit Price"]
    mock_data["Date"] = pd.to_datetime(mock_data["Date"])
    return mock_data

# 3. Sidebar - Data Import
st.sidebar.header("📥 Data Source")
uploaded_file = st.sidebar.file_uploader("Upload your Excel or CSV file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.sidebar.success("File uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")
        st.stop()
else:
    st.sidebar.info("Using sample mock data. Upload your own file to customize!")
    df = load_mock_data()

# Ensure Date column is datetime
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"])

# 4. Sidebar - Interactive Filters / Slicers
st.sidebar.header("🎛️ Filters & Slicers")

# Date Range Filter
if "Date" in df.columns:
    min_date = df["Date"].min().to_pydatetime()
    max_date = df["Date"].max().to_pydatetime()
    start_date, end_date = st.sidebar.date_input("Select Date Range", [min_date, max_date])
    df = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]

# Product Filter
if "Product" in df.columns:
    selected_products = st.sidebar.multiselect("Select Products", options=df["Product"].unique(), default=df["Product"].unique())
    df = df[df["Product"].isin(selected_products)]

# Region Filter
if "Region" in df.columns:
    selected_regions = st.sidebar.multiselect("Select Regions", options=df["Region"].unique(), default=df["Region"].unique())
    df = df[df["Region"].isin(selected_regions)]

# 5. KPI Metrics Row
st.subheader("📈 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

total_revenue = df["Revenue"].sum() if "Revenue" in df.columns else 0
total_units = df["Units Sold"].sum() if "Units Sold" in df.columns else 0
avg_order_val = total_revenue / len(df) if len(df) > 0 else 0
unique_products = df["Product"].nunique() if "Product" in df.columns else 0

with col1:
    st.metric(label="Total Revenue", value=f"${total_revenue:,.2f}")
with col2:
    st.metric(label="Units Sold", value=f"{total_units:,}")
with col3:
    st.metric(label="Avg Order Value", value=f"${avg_order_val:,.2f}")
with col4:
    st.metric(label="Product Categories", value=unique_products)

st.markdown("---")

# 6. Visualizations
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📅 Revenue Trend Over Time")
    if "Date" in df.columns and "Revenue" in df.columns:
        # Group by Month-Year for cleaner trends
        df_trend = df.set_index("Date").resample("ME")["Revenue"].sum().reset_index()
        fig_trend = px.line(df_trend, x="Date", y="Revenue", markers=True, 
                            labels={"Revenue": "Revenue ($)", "Date": "Month"},
                            template="plotly_white", color_discrete_sequence=["#1f77b4"])
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.warning("Ensure 'Date' and 'Revenue' columns exist.")

with col_right:
    st.subheader("🏆 Top Performing Products")
    if "Product" in df.columns and "Revenue" in df.columns:
        df_prod = df.groupby("Product")["Revenue"].sum().sort_values(ascending=True).reset_index()
        fig_prod = px.bar(df_prod, x="Revenue", y="Product", orientation='h',
                          labels={"Revenue": "Total Revenue ($)"},
                          template="plotly_white", color="Revenue", color_continuous_scale="Viridis")
        st.plotly_chart(fig_prod, use_container_width=True)
    else:
        st.warning("Ensure 'Product' and 'Revenue' columns exist.")

# 7. Raw Data Breakdown
st.markdown("---")
st.subheader("📋 Filtered Transaction Data")
st.dataframe(df, use_container_width=True)
