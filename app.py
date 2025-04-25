import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from PIL import Image


logo = Image.open("logo.png")  # Ensure logo.png is in the same directory
st.image(logo, width=120)
st.markdown("<h1 style='text-align: center;'>📊 Sales Intelligence & Product Recommendation Dashboard</h1>", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv("/content/drive/MyDrive/Data Analysis - Sample File.csv")
    df['Redistribution Value'] = df['Redistribution Value'].str.replace(',', '').astype(float)
    df['Delivered_date'] = pd.to_datetime(df['Delivered_date'], errors='coerce')
    df['Month'] = df['Delivered_date'].dt.to_period('M')
    return df

df = load_data()

# --- Sidebar Navigation ---
st.sidebar.title("📂 Navigation")
section = st.sidebar.radio("Choose a Section", [
    "📊 EDA Overview",
    "📉 Drop Detection",
    "👤 Customer Profiling",
    "🔁 Cross-Selling",
    "🔗 Brand Correlation",
    "🥇 Buyer Analysis",
    "📈 Retention & Moving Average",
    "🤖 Recommender System"
])

# Shared aggregations
df['Month'] = df['Delivered_date'].dt.to_period('M')
monthly_summary = df.groupby('Month')['Redistribution Value'].sum()

# --- EDA Overview ---
if section == "📊 EDA Overview":
    st.subheader("Sales Trends Over Time")
    monthly_orders = df.groupby(df['Delivered_date'].dt.to_period('M'))['Order_Id'].nunique()
    monthly_orders.index = monthly_orders.index.to_timestamp()
    monthly_summary.index = monthly_summary.index.to_timestamp()
    st.line_chart(monthly_summary)

    st.subheader("Top-Selling Products")
    top_products = df.groupby('SKU_Code')['Redistribution Value'].sum().sort_values(ascending=False).head(10)
    st.bar_chart(top_products)

    st.subheader("Top Brands")
    top_brands = df.groupby('Brand')['Redistribution Value'].sum().sort_values(ascending=False).head(10)
    st.bar_chart(top_brands)

# --- Drop Detection ---
elif section == "📉 Drop Detection":
    st.subheader("Brand-Level Month-over-Month Drop Flags")
    brand_monthly = df.groupby(['Brand', 'Month'])['Redistribution Value'].sum().unstack(fill_value=0)
    brand_mom_change = brand_monthly.pct_change(axis=1) * 100
    drop_flags = brand_mom_change < -30
    drop_summary = brand_mom_change.round(1).astype(str)
    drop_summary[drop_flags] += "% 🔻"
    drop_summary[~drop_flags] = ""
    st.dataframe(drop_summary)

# --- Customer Profiling (RFM) ---
elif section == "👤 Customer Profiling":
    st.subheader("Customer RFM Segmentation")
    ref_date = df['Delivered_date'].max()
    rfm = df.groupby('Customer_Phone').agg({
        'Delivered_date': lambda x: (ref_date - x.max()).days,
        'Order_Id': 'nunique',
        'Redistribution Value': 'sum'
    }).reset_index()
    rfm.columns = ['Customer_Phone', 'Recency', 'Frequency', 'Monetary']

    st.dataframe(rfm.head(10))

# --- Cross-Selling ---
elif section == "🔁 Cross-Selling":
    st.subheader("Brand Switching Patterns (Top 3 Alternatives)")
    last_purchase = df.groupby(['Customer_Phone', 'Brand'])['Month'].max().reset_index()
    latest_month = df['Month'].max()
    drop_customers = last_purchase[last_purchase['Month'] < latest_month]
    merged = df.merge(drop_customers, on='Customer_Phone', suffixes=('', '_dropped'))
    switched = merged[(merged['Month'] > merged['Month_dropped']) & (merged['Brand'] != merged['Brand_dropped'])]
    switches = switched.groupby(['Brand_dropped', 'Brand'])['Order_Id'].count().reset_index().sort_values(['Brand_dropped', 'Order_Id'], ascending=[True, False])
    st.dataframe(switches.groupby('Brand_dropped').head(3))

# --- Brand Correlation ---
elif section == "🔗 Brand Correlation":
    st.subheader("Customer Brand Correlation Matrix")
    user_brand = df.groupby(['Customer_Phone', 'Brand'])['Order_Id'].count().unstack().fillna(0)
    corr = user_brand.corr()
    st.dataframe(corr.round(2))

# --- Buyer Analysis ---
elif section == "🥇 Buyer Analysis":
    st.subheader("Top & Bottom Buyers (Latest Month)")
    latest_month = df['Month'].max()
    latest_data = df[df['Month'] == latest_month]
    buyer_summary = latest_data.groupby('Customer_Phone')['Redistribution Value'].sum().reset_index()
    st.write("Top Buyers")
    st.dataframe(buyer_summary.sort_values(by='Redistribution Value', ascending=False).head(10))
    st.write("Bottom Buyers")
    st.dataframe(buyer_summary.sort_values(by='Redistribution Value').head(10))

# --- Retention & Moving Average ---
elif section == "📈 Retention & Moving Average":
    st.subheader("3-Month Moving Average of Orders")
    monthly_orders = df.groupby('Month')['Order_Id'].nunique()
    ma = monthly_orders.rolling(3).mean()
    st.line_chart(ma)

# --- Recommender System ---
elif section == "🤖 Recommender System":
    st.subheader("Hybrid Product Recommendations")

    user_item_matrix = df.pivot_table(index='Customer_Phone', columns='SKU_Code', values='Redistribution Value', aggfunc='sum').fillna(0)
    product_features = df[['SKU_Code', 'Brand']].drop_duplicates().set_index('SKU_Code')
    product_features_encoded = pd.get_dummies(product_features, columns=['Brand'])

    user_sim = cosine_similarity(user_item_matrix)
    user_sim_df = pd.DataFrame(user_sim, index=user_item_matrix.index, columns=user_item_matrix.index)

    item_sim = cosine_similarity(product_features_encoded)
    item_sim_df = pd.DataFrame(item_sim, index=product_features_encoded.index, columns=product_features_encoded.index)

    def hybrid_recommend(customer_id, top_n=5):
        if customer_id not in user_item_matrix.index:
            return pd.DataFrame({'Error': [f"Customer {customer_id} not found."]})
        sim_users = user_sim_df[customer_id]
        weighted_scores = user_item_matrix.T.dot(sim_users).div(sim_users.sum())
        bought = user_item_matrix.loc[customer_id]
        bought_items = bought[bought > 0].index
        content_scores = item_sim_df[bought_items].sum(axis=1)
        final_scores = 0.5 * weighted_scores + 0.5 * content_scores
        final_scores = final_scores.drop(bought_items, errors='ignore')
        return final_scores.sort_values(ascending=False).head(top_n).reset_index().rename(columns={0: 'Score', 'index': 'Recommended SKU'})

    selected_customer = st.selectbox("Select Customer", user_item_matrix.index)
    if st.button("Show Recommendations"):
        st.dataframe(hybrid_recommend(selected_customer))

