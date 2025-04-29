import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from PIL import Image

# --- App Header ---
logo = Image.open("logo.png")  # Ensure logo.png is in the same directory
st.image(logo, width=120)
st.markdown("<h1 style='text-align: center;'>📊 Sales Intelligence & Product Recommendation Dashboard</h1>", unsafe_allow_html=True)

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_data(path="Data Analysis - Sample File.csv"):
    df = pd.read_csv(path)
    df['Redistribution Value'] = df['Redistribution Value'].str.replace(',', '', regex=False).astype(float)
    df['Delivered_date'] = pd.to_datetime(df['Delivered_date'], format='%d/%m/%Y %H:%M', errors='coerce')
    df['Month'] = df['Delivered_date'].dt.to_period('M')
    return df

df = load_data()

# --- Build Matrices & Similarities ---
@st.cache_data
def build_user_item(df):
    ui = df.pivot_table(
        index='Customer_Phone',
        columns='SKU_Code',
        values='Redistribution Value',
        aggfunc='sum'
    ).fillna(0)
    return ui

user_item_matrix = build_user_item(df)

@st.cache_data
def build_item_sim(df):
    prod = df[['SKU_Code','Brand']].drop_duplicates().set_index('SKU_Code')
    prod_enc = pd.get_dummies(prod, columns=['Brand'])
    sim = cosine_similarity(prod_enc)
    return pd.DataFrame(sim, index=prod_enc.index, columns=prod_enc.index)

item_sim_df = build_item_sim(df)

@st.cache_data
def build_user_sim(user_item_matrix):
    sim = cosine_similarity(user_item_matrix)
    return pd.DataFrame(sim, index=user_item_matrix.index, columns=user_item_matrix.index)

user_sim_df = build_user_sim(user_item_matrix)

# --- Hybrid Recommendation Function ---
def hybrid_recommend(customer_id, top_n=5):
    if customer_id not in user_item_matrix.index:
        return pd.DataFrame({'Error': [f"Customer {customer_id} not found."]})
    sim_users = user_sim_df.loc[customer_id]
    weighted_scores = user_item_matrix.T.dot(sim_users) / sim_users.sum()
    bought = user_item_matrix.loc[customer_id]
    bought_items = bought[bought > 0].index
    content_scores = item_sim_df[bought_items].sum(axis=1)
    final_scores = 0.5 * weighted_scores + 0.5 * content_scores
    final_scores = final_scores.drop(bought_items, errors='ignore')
    rec = final_scores.sort_values(ascending=False).head(top_n).reset_index()
    rec.columns = ['Recommended SKU','Score']
    return rec

@st.cache_data
def compute_all_recs():
    return pd.concat([
        hybrid_recommend(c).assign(Customer_Phone=c)
        for c in user_item_matrix.index
    ], ignore_index=True)

all_recommendations = compute_all_recs()

# --- Predict Next Purchase & Likely SKU ---
@st.cache_data
def compute_purchase_summary(df):
    grp = df.sort_values(['Customer_Phone','Delivered_date']).groupby('Customer_Phone')
    last_dates = grp['Delivered_date'].last().rename('last_purchase_date')
    inter_days = grp['Delivered_date'].diff().dt.days
    avg_days = inter_days.groupby(df['Customer_Phone']).mean().fillna(inter_days.mean()).rename('avg_inter_purchase_days')
    summary = pd.concat([last_dates, avg_days], axis=1)
    summary['predicted_next_purchase'] = summary['last_purchase_date'] + pd.to_timedelta(summary['avg_inter_purchase_days'], unit='D')
    freq = df.groupby(['Customer_Phone','SKU_Code']).size().rename('count').reset_index()
    idx = freq.groupby('Customer_Phone')['count'].idxmax()
    likely = freq.loc[idx][['Customer_Phone','SKU_Code']].rename(columns={'SKU_Code':'likely_next_SKU'})
    return summary.reset_index().merge(likely, on='Customer_Phone')

purchase_summary = compute_purchase_summary(df)

# --- RFM Analysis & Discount Suggestions ---
@st.cache_data
def compute_rfm_discounts(df):
    ref_date = df['Delivered_date'].max()
    last_purchase = df.groupby('Customer_Phone')['Delivered_date'].max()
    recency = (ref_date - last_purchase).dt.days
    frequency = df.groupby('Customer_Phone')['Order_Id'].nunique()
    order_vals = df.groupby(['Customer_Phone','Order_Id'])['Redistribution Value'].sum().reset_index()
    monetary = order_vals.groupby('Customer_Phone')['Redistribution Value'].mean()
    rfm = pd.DataFrame({'Recency':recency,'Frequency':frequency,'Monetary':monetary})
    q = {d: rfm[d].quantile([0.25,0.5,0.75]) for d in ['Recency','Frequency','Monetary']}
    def seg(r):
        if r['Recency']<=q['Recency'][0.25] and r['Frequency']>=q['Frequency'][0.75] and r['Monetary']>=q['Monetary'][0.75]:
            return 'Best Customers'
        elif r['Recency']>=q['Recency'][0.75] and r['Frequency']<=q['Frequency'][0.25]:
            return 'At-Risk Customers'
        elif r['Recency']>=q['Recency'][0.75] and r['Monetary']>=q['Monetary'][0.75]:
            return 'Big Spenders Dropping Off'
        elif q['Recency'][0.25]<r['Recency']<=q['Recency'][0.75] and r['Frequency']>=q['Frequency'][0.5]:
            return 'Potential Loyalists'
        else:
            return 'Others'
    rfm['Segment'] = rfm.apply(seg, axis=1)
    med = {
        'Best Customers': rfm[rfm['Segment']=='Best Customers']['Monetary'].median(),
        'Potential Loyalists': rfm[rfm['Segment']=='Potential Loyalists']['Monetary'].median()
    }
    rfm['Recommended_Discount'] = 0
    for segm,pct in med.items():
        mask = (rfm['Segment']==segm)&(rfm['Monetary']<pct)
        rfm.loc[mask,'Recommended_Discount'] = 10 if segm=='Best Customers' else 5
    return rfm.reset_index()

rfm_df = compute_rfm_discounts(df)

# --- Sidebar & Sections ---
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

# Shared monthly summary
monthly_summary = df.groupby('Month')['Redistribution Value'].sum()
monthly_orders = df.groupby(df['Delivered_date'].dt.to_period('M'))['Order_Id'].nunique()

# --- Sections ---
if section == "📊 EDA Overview":
    st.subheader("Sales Trends Over Time")
    st.line_chart(monthly_summary.to_timestamp())
    st.subheader("Top-Selling Products")
    top_products = df.groupby('SKU_Code')['Redistribution Value'].sum().nlargest(10)
    st.bar_chart(top_products)
    st.subheader("Top Brands")
    top_brands = df.groupby('Brand')['Redistribution Value'].sum().nlargest(10)
    st.bar_chart(top_brands)

elif section == "📉 Drop Detection":
    st.subheader("Brand-Level Month-over-Month Drop Flags")
    brand_monthly = df.groupby(['Brand','Month'])['Redistribution Value'].sum().unstack(fill_value=0)
    mom = brand_monthly.pct_change(axis=1)*100
    flags = mom < -30
    display = mom.round(1).astype(str)
    display[flags] += "% 🔻"
    display[~flags] = ""
    st.dataframe(display)

elif section == "👤 Customer Profiling":
    st.subheader("Customer Profile & Forecast")
    customer_id = st.selectbox("Select Customer Phone:", options=user_item_matrix.index)
    cust_rfm = rfm_df[rfm_df['Customer_Phone']==customer_id].iloc[0]
    st.metric("Recency (days)", cust_rfm['Recency'])
    st.metric("Frequency (# orders)", cust_rfm['Frequency'])
    st.metric("Monetary (avg spend)", f"₦{cust_rfm['Monetary']:.2f}")
    st.metric("Segment", cust_rfm['Segment'])
    if cust_rfm['Recommended_Discount']>0:
        st.write(f"💡 Recommended Discount: {cust_rfm['Recommended_Discount']}%")
    cust_sum = purchase_summary[purchase_summary['Customer_Phone']==customer_id].iloc[0]
    st.write("**Predicted Next Purchase:**", cust_sum['predicted_next_purchase'].date())
    st.write("**Likely Next SKU:**", cust_sum['likely_next_SKU'])

elif section == "🔁 Cross-Selling":
    st.subheader("Brand Switching Patterns (Top 3 Alternatives)")
    last_purchase = df.groupby(['Customer_Phone','Brand'])['Month'].max().reset_index()
    latest_month = df['Month'].max()
    dropped = last_purchase[last_purchase['Month']<latest_month]
    merged = df.merge(dropped, on='Customer_Phone', suffixes=('','_dropped'))
    switched = merged[(merged['Month']>merged['Month_dropped'])&(merged['Brand']!=merged['Brand_dropped'])]
    switches = switched.groupby(['Brand_dropped','Brand'])['Order_Id'].count().reset_index()
    switches = switches.sort_values(['Brand_dropped','Order_Id'],ascending=[True,False])
    st.dataframe(switches.groupby('Brand_dropped').head(3))

elif section == "🔗 Brand Correlation":
    st.subheader("Customer Brand Correlation Matrix")
    user_brand = df.groupby(['Customer_Phone','Brand'])['Order_Id'].count().unstack().fillna(0)
    corr = user_brand.corr()
    st.dataframe(corr.round(2))

elif section == "🥇 Buyer Analysis":
    st.subheader("Top & Bottom Buyers (Latest Month)")
    lm = df[df['Month']==df['Month'].max()]
    summary = lm.groupby('Customer_Phone')['Redistribution Value'].sum().reset_index()
    st.write("Top Buyers")
    st.dataframe(summary.nlargest(10,'Redistribution Value'))
    st.write("Bottom Buyers")
    st.dataframe(summary.nsmallest(10,'Redistribution Value'))

elif section == "📈 Retention & Moving Average":
    st.subheader("3-Month Moving Average of Orders")
    ma = monthly_orders.rolling(3).mean()
    st.line_chart(ma.to_timestamp())

elif section == "🤖 Recommender System":
    st.subheader("Hybrid Product Recommendations")
    selected_customer = st.selectbox("Select Customer:", user_item_matrix.index)
    if st.button("Show Recommendations"):
        recs = hybrid_recommend(selected_customer)
        st.dataframe(recs)
