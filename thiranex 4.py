import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px

# 1. Page Configuration
st.set_page_config(page_title="Data Cleaning & Reporting Automation", layout="wide")
st.title("🤖 Data Cleaning & Reporting Automation Pipeline")
st.markdown("Instantly transform dirty raw source data into production-ready analytical assets.")

# 2. Generate a "Dirty" Sample Dataset for Simulation
@st.cache_data
def load_dirty_data():
    np.random.seed(101)
    n_rows = 200
    
    # Standard template base
    dates = pd.date_range(start="2026-01-01", periods=15, freq="D")
    products = ["  Laptop ", "Smartphone", "Tablet", "smartwatch", "Headphones", "SMARTPHONE"] # Added spaces & casing inconsistency
    regions = ["North", "South", "East", "West", "UNKNOWN"]
    
    df = pd.DataFrame({
        "Transaction_ID": [f"TXN-{1000 + i}" for i in range(n_rows)],
        "Date": np.random.choice(dates, size=n_rows),
        "Product": np.random.choice(products, size=n_rows),
        "Region": np.random.choice(regions, size=n_rows),
        "Units_Sold": np.random.choice([1, 2, 3, 4, np.nan, 5], size=n_rows), # Contains Null values
        "Revenue": np.random.choice([1200, 800, 400, 150, 0, np.nan], size=n_rows) # Contains Nulls and Zeroes
    })
    
    # Explicitly inject exact duplicate rows to simulate human data entry errors
    duplicates = df.sample(n=12, random_state=42)
    df = pd.concat([df, duplicates], ignore_index=True)
    
    return df

df_dirty = load_dirty_data()

# 3. Sidebar - Data Ingestion Flow
st.sidebar.header("📥 Data Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload Raw Data Sheet", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)
else:
    st.sidebar.info("Processing synthetic dirty dataset. Upload an file to run custom pipelines.")
    df_raw = df_dirty.copy()

# Display Initial States
col_raw, col_clean = st.columns(2)

with col_raw:
    st.subheader("❌ Raw Ingested Data Auditing")
    st.markdown("Initial diagnostic properties of uncurated data stream:")
    
    # Audit Statistics calculations
    total_rows = len(df_raw)
    duplicate_count = df_raw.duplicated().sum()
    null_counts = df_raw.isna().sum().sum()
    
    st.error(f"⚠️ Total Records: {total_rows} | Duplicates Detected: {duplicate_count} | Total Missing Fields: {null_counts}")
    st.dataframe(df_raw.head(10), use_container_width=True)

# 4. Automation Processing Pipeline Engine
def run_automated_pipeline(input_df):
    cleaned_df = input_df.copy()
    
    # Step A: Drop absolute duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()
    
    # Step B: Standardize textual structural items (String Trimming & Title Casing)
    if "Product" in cleaned_df.columns:
        cleaned_df["Product"] = cleaned_df["Product"].astype(str).str.strip().str.title()
        
    if "Region" in cleaned_df.columns:
        cleaned_df["Region"] = cleaned_df["Region"].astype(str).str.strip().str.upper()
        # Handle placeholders like "UNKNOWN" as true missing indicators
        cleaned_df["Region"] = cleaned_df["Region"].replace("UNKNOWN", np.nan)
    
    # Step C: Intelligent Imputation / Handling Missing Numbers
    if "Units_Sold" in cleaned_df.columns:
        # Fallback to median value for realistic whole counts
        median_units = cleaned_df["Units_Sold"].median()
        cleaned_df["Units_Sold"] = cleaned_df["Units_Sold"].fillna(median_units)
        
    if "Revenue" in cleaned_df.columns:
        # Fallback to mean value for continuous currency structures
        mean_rev = cleaned_df["Revenue"].mean()
        cleaned_df["Revenue"] = cleaned_df["Revenue"].fillna(mean_rev)
        
    # Step D: Enforce Static Strict Datatypes
    if "Date" in cleaned_df.columns:
        cleaned_df["Date"] = pd.to_datetime(cleaned_df["Date"])
        
    return cleaned_df

# Run execution
df_clean_data = run_automated_pipeline(df_raw)

with col_clean:
    st.subheader("✅ Automated Data Cleaning Output")
    st.markdown("Refined and optimized table structure post-pipeline execution:")
    
    clean_total_rows = len(df_clean_data)
    clean_dupes = df_clean_data.duplicated().sum()
    clean_nulls = df_clean_data.isna().sum().sum()
    
    st.success(f"💎 Cleaned Records: {clean_total_rows} | Duplicates Remaining: {clean_dupes} | Empty Blocks: {clean_nulls}")
    st.dataframe(df_clean_data.head(10), use_container_width=True)

st.markdown("---")

# 5. Automated Summary Reporting Visualization
st.subheader("📊 Executive Reporting Visual Summaries")
report_col1, report_col2 = st.columns([1, 2])

with report_col1:
    st.markdown("**Core Summary Metrics Table**")
    if "Product" in df_clean_data.columns and "Revenue" in df_clean_data.columns:
        summary_metrics = df_clean_data.groupby("Product").agg(
            Total_Units=("Units_Sold", "sum"),
            Total_Revenue=("Revenue", "sum")
        ).reset_index()
        st.dataframe(summary_metrics, use_container_width=True)

with report_col2:
    if "Product" in df_clean_data.columns and "Revenue" in df_clean_data.columns:
        fig_pie = px.pie(
            summary_metrics, 
            values="Total_Revenue", 
            names="Product", 
            title="Revenue Contribution Breakdown by Product Category",
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# 6. Automated Report File Generator (Excel Export Routine)
st.markdown("---")
st.subheader("💾 Generate Downstream Export Package")
st.markdown("Download a pristine, auto-generated copy of this analysis directly into Excel.")

# Buffer data stream pipeline creation
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
    df_clean_data.to_excel(writer, sheet_name="Cleaned Data", index=False)
    if 'summary_metrics' in locals():
        summary_metrics.to_excel(writer, sheet_name="Executive Summary", index=False)

st.download_button(
    label="📥 Download Production-Ready Excel File",
    data=buffer.getvalue(),
    file_name="Automated_Cleaned_Sales_Report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
