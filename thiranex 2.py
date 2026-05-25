import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Page Configuration
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")
st.title("👥 Customer Segmentation Analytics")
st.markdown("Group customers by demographics and behavioral patterns using Machine Learning (K-Means).")

# 2. Generate Synthetic Data
@st.cache_data
def load_customer_data():
    np.random.seed(42)
    n_customers = 400
    
    # Create 3 distinct structural groups to simulate real world clusters
    # Group 1: Young, lower income, high spending (Impulsive buyers)
    # Group 2: Middle-aged, high income, high spending (Target/VIP)
    # Group 3: Older, medium income, low spending (Frugal/Conservative)
    
    g1_age = np.random.randint(18, 30, 130)
    g1_income = np.random.randint(15, 45, 130)
    g1_spending = np.random.randint(60, 95, 130)
    
    g2_age = np.random.randint(30, 50, 140)
    g2_income = np.random.randint(70, 120, 140)
    g2_spending = np.random.randint(65, 95, 140)
    
    g3_age = np.random.randint(45, 70, 130)
    g3_income = np.random.randint(40, 75, 130)
    g3_spending = np.random.randint(10, 40, 130)
    
    df = pd.DataFrame({
        "Customer_ID": [f"CUST-{i:03d}" for i in range(1, n_customers + 1)],
        "Age": np.concatenate([g1_age, g2_age, g3_age]),
        "Annual_Income_k$": np.concatenate([g1_income, g2_income, g3_income]),
        "Spending_Score": np.concatenate([g1_spending, g2_spending, g3_spending]),
        "Gender": np.random.choice(["Male", "Female"], size=n_customers)
    })
    return df

df_raw = load_customer_data()

# 3. Sidebar Configuration
st.sidebar.header("⚙️ Model Configuration")
st.sidebar.markdown("Adjust settings to retrain the K-Means clustering algorithm instantly.")

# Slicer for Number of Clusters
num_clusters = st.sidebar.slider("Select Number of Clusters (K)", min_value=2, max_value=6, value=3)

# Feature Selection for Clustering
features = ["Age", "Annual_Income_k$", "Spending_Score"]
selected_features = st.sidebar.multiselect("Features to include in analysis:", features, default=features)

if len(selected_features) < 2:
    st.error("Please select at whom at least 2 features to perform clustering.")
    st.stop()

# 4. Machine Learning Pipeline (Scaling & Clustering)
@st.cache_data
def perform_clustering(data, selected_feats, n_clst):
    df_cluster = data[selected_feats].copy()
    
    # Feature Scaling (Crucial for distance-based algorithms like K-Means)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_cluster)
    
    # Fit K-Means
    kmeans = KMeans(n_clusters=n_clst, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)
    
    # Append labels back to original data
    output_df = data.copy()
    output_df["Segment"] = [f"Segment {i+1}" for i in cluster_labels]
    return output_df

df_analyzed = perform_clustering(df_raw, selected_features, num_clusters)

# 5. Dashboard Metrics (KPI Summary Table)
st.subheader("📊 Segment Profiling Matrix")
st.markdown("Understand the core behaviors of each identified customer group:")

# Generate grouped summaries
summary_df = df_analyzed.groupby("Segment").agg({
    "Customer_ID": "count",
    "Age": "mean",
    "Annual_Income_k$": "mean",
    "Spending_Score": "mean"
}).rename(columns={"Customer_ID": "Total Customers", "Age": "Avg Age", "Annual_Income_k$": "Avg Income (k$)", "Spending_Score": "Avg Spending Score"})

st.dataframe(summary_df.style.format("{:.1f}").background_gradient(cmap="Blues", axis=0), use_container_width=True)

st.markdown("---")

# 6. Interactive Visualizations
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 3D Customer Cluster Space")
    if len(selected_features) == 3:
        fig_3d = px.scatter_3d(
            df_analyzed, 
            x="Annual_Income_k$", 
            y="Spending_Score", 
            z="Age",
            color="Segment",
            hover_name="Customer_ID",
            opacity=0.8,
            title="Customer Distribution Across 3 Dimensions",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        # Fallback 2D plot if they deselect a feature
        fig_2d = px.scatter(
            df_analyzed,
            x=selected_features[0],
            y=selected_features[1],
            color="Segment",
            hover_name="Customer_ID",
            title=f"2D Segmentation Mapping ({selected_features[0]} vs {selected_features[1]})",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        st.plotly_chart(fig_2d, use_container_width=True)

with col2:
    st.subheader("📦 Segment Demographics Breakdown")
    # Box plot to show spread of attributes within segments
    attribute_to_plot = st.selectbox("Select attribute to view variance:", ["Annual_Income_k$", "Spending_Score", "Age"])
    
    fig_box = px.box(
        df_analyzed,
        x="Segment",
        y=attribute_to_plot,
        color="Segment",
        points="all",
        title=f"Distribution of {attribute_to_plot} by Segment",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    st.plotly_chart(fig_box, use_container_width=True)

# 7. Segment Deep Dive Lookup Table
st.markdown("---")
st.subheader("🔍 Export Segment Lists")
selected_segment = st.selectbox("Filter table to extract a specific group:", options=sorted(df_analyzed["Segment"].unique()))
filtered_segment_data = df_analyzed[df_analyzed["Segment"] == selected_segment]

st.dataframe(filtered_segment_data, use_container_width=True)
