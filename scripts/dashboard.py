import streamlit as st
import pandas as pd
from deltalake import DeltaTable

st.set_page_config(
    page_title="Enterprise Fraud Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛡️ Institutional Credit Fraud Analytics Hub")
st.markdown("### End-to-End Multi-Source Lakehouse Ingestion & Inference Command Center")
st.divider()

@st.cache_data
def load_gold_features():
    try:
        # Pulls structural columns (like lat, lon, amount) directly from the base Gold layer
        dt = DeltaTable("./data/gold/")
        return dt.to_pandas()
    except Exception as e:
        st.error(f"Error loading base Gold features: {e}")
        return pd.DataFrame()

@st.cache_data
def load_scores():
    try:
        dt = DeltaTable("./data/gold/fraud_scores")
        return dt.to_pandas()
    except Exception as e:
        st.error(f"Error loading fraud scores: {e}")
        return pd.DataFrame()

@st.cache_data
def load_alerts():
    try:
        dt = DeltaTable("./data/gold/fraud_alerts")
        return dt.to_pandas()
    except Exception as e:
        st.error(f"Error loading fraud alerts: {e}")
        return pd.DataFrame()

# Load all three layers seamlessly
gold_df = load_gold_features()
scores_df = load_scores()
alerts_df = load_alerts()

# Calculate Summary Metrics Safely
if not scores_df.empty:
    total_tx = len(scores_df)
    total_alerts = len(alerts_df)
    ground_fraud_rate = (total_alerts / total_tx * 100) if total_tx > 0 else 0.0
    
    # Extract total monitored capital directly from the feature dataset
    total_volume_usd = float(gold_df["amount"].sum()) if "amount" in gold_df.columns else 0.0
else:
    total_tx, total_alerts, ground_fraud_rate, total_volume_usd = 0, 0, 0.0, 0.0

# Metric Banner Cards
m_col1, m_col2, m_col3, m_col4 = st.columns(4)
with m_col1:
    st.metric(label="Total Audited Population", value=f"{total_tx:,}")
with m_col2:
    st.metric(label="Isolated High-Risk Anomalies", value=f"{total_alerts:,}")
with m_col3:
    st.metric(label="Pipeline System Fraud Rate", value=f"{ground_fraud_rate:.4f}%")
with m_col4:
    st.metric(label="Total Capital Monitored", value=f"${total_volume_usd:,.2f}")

st.divider()

# Upper Layout Split: Geospatial Hotspots Map & Confidence Distribution Chart
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("🗺️ Live Geospatial Fraud Hotspots")
    
    # We pull the coordinates directly from the feature-rich gold_df dataframe!
    if not gold_df.empty and not alerts_df.empty:
        lat_col = next((c for c in ['lat', 'latitude', 'merch_lat'] if c in gold_df.columns), None)
        lon_col = next((c for c in ['lon', 'lng', 'longitude', 'merch_long'] if c in gold_df.columns), None)
        
        if lat_col and lon_col:
            # Filter down to coordinates belonging to high-risk transactions
            alert_ids = alerts_df["transaction_id"].unique() if "transaction_id" in alerts_df.columns else []
            geo_alerts = gold_df[gold_df["transaction_id"].isin(alert_ids)] if "transaction_id" in gold_df.columns else gold_df
            
            map_data = geo_alerts[[lat_col, lon_col]].dropna().rename(columns={lat_col: 'latitude', lon_col: 'longitude'})
            st.map(map_data, zoom=1, use_container_width=True)
        else:
            st.info("Geospatial data fields not found in gold storage layer tables.")
    else:
        st.success("Operational status clear: No anomalous tracking metrics available.")

with chart_col2:
    st.subheader("🎯 Model Prediction Confidence Distribution")
    if not scores_df.empty and "fraud_probability" in scores_df.columns:
        prob_counts = scores_df["fraud_probability"].round(1).value_counts().sort_index()
        prob_counts.index = [f"Prob {idx:.1f}" for idx in prob_counts.index]
        st.bar_chart(prob_counts, color="#ff7f0e", use_container_width=True)
    else:
        st.warning("Prediction score confidence vectors unavailable.")

st.divider()

# Lower Layout Split: Category Profiling Area Chart & Dynamic Alerts Queue Dataframe
data_col1, data_col2 = st.columns([1, 1.5])

with data_col1:
    st.subheader("💸 Volume Distribution by Channel")
    if not gold_df.empty and "transaction_type" in gold_df.columns:
        type_vol = gold_df.groupby("transaction_type")["amount"].sum().sort_values(ascending=False)
        st.area_chart(type_vol, color="#2ca02c", use_container_width=True)
    else:
        st.info("Transaction channel categorizations unavailable.")

with data_col2:
    st.subheader("⚠️ Active High-Priority Fraud Alert Stream")
    if not alerts_df.empty:
        display_cols = [col for col in ["transaction_id", "timestamp", "amount", "transaction_type", "fraud_probability"] if col in alerts_df.columns]
        sorted_alerts = alerts_df[display_cols].sort_values(by="fraud_probability", ascending=False).head(100)
        
        st.dataframe(
            sorted_alerts,
            column_config={
                "amount": st.column_config.NumberColumn("Transaction Amount", format="$%.2f"),
                "fraud_probability": st.column_config.ProgressColumn("Risk Confidence", min_value=0.0, max_value=1.0, format="%.2f"),
                "timestamp": "Execution Time",
                "transaction_id": "Alert Token"
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.success("Operational status clear: No anomalous entries detected above verification limit.")