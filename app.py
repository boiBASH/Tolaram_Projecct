import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from PIL import Image

# --- Load and preprocess data ---
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_data_analysis.csv")
    df['Redistribution Value'] = df['Redistribution Value'].str.replace(',', '', regex=False).astype(float)
    df['Delivered_date'] = pd.to_datetime(df['Delivered_date'], errors='coerce')
    df['Month'] = df['Delivered_date'].dt.to_period('M')
    df['Delivered Qty'] = df['Delivered Qty'].fillna(0)
    df['Total_Amount_Spent'] = df['Redistribution Value'] * df['Delivered Qty']
    return df

DF = load_data()

# --- Analysis functions ---
def analyze_customer_purchases(customer_phone):
    customer_df = DF[DF['Customer_Phone'] == customer_phone].copy()
    if customer_df.empty:
        return {}
    customer_df.sort_values('Delivered_date', inplace=True)
    customer_df['Month'] = customer_df['Delivered_date'].dt.to_period('M')

    skus_bought = customer_df['SKU_Code'].unique().tolist()
    last_purchase = (
        customer_df.groupby('SKU_Code')['Delivered_date']
        .max()
        .dt.strftime('%Y-%m-%d')
        .to_dict()
    )
    monthly_qty = (
        customer_df.groupby(['SKU_Code', 'Month'])['Delivered Qty']
        .sum()
        .groupby('SKU_Code')
        .mean()
        .round(2)
        .to_dict()
    )
    avg_interval = {}
    for sku, group in customer_df.groupby('SKU_Code'):
        dates = group['Delivered_date'].drop_duplicates().sort_values()
        if len(dates) > 1:
            intervals = dates.diff().dropna().dt.days / 30.44
            avg_interval[sku] = round(intervals.mean(), 2)
        else:
            avg_interval[sku] = "One"
    monthly_spend = (
        customer_df.groupby(['SKU_Code', 'Month'])['Total_Amount_Spent']
        .sum()
        .groupby('SKU_Code')
        .mean()
        .round(2)
        .to_dict()
    )

    report = {
        'Customer Phone': customer_phone,
        'Total Unique SKUs Bought': len(skus_bought),
        'SKUs Bought': skus_bought,
        'Purchase Summary by SKU': {}
    }
    for sku in skus_bought:
        report['Purchase Summary by SKU'][sku] = {
            'Last Purchase Date': last_purchase.get(sku, 'N/A'),
            'Avg Monthly Quantity': monthly_qty.get(sku, 0),
            'Avg Purchase Interval (Months)': avg_interval.get(sku, 'N/A'),
            'Avg Monthly Spend': monthly_spend.get(sku, 0)
        }
    return report


def predict_next_purchases(customer_phone):
    customer_df = DF[DF['Customer_Phone'] == customer_phone].copy()
    if customer_df.empty:
        return pd.DataFrame()
    customer_df['Month'] = customer_df['Delivered_date'].dt.to_period('M')

    freq = customer_df.groupby('SKU_Code')['Delivered_date'].nunique()
    qty = customer_df.groupby('SKU_Code')['Delivered Qty'].sum()
    spend = customer_df.groupby('SKU_Code')['Total_Amount_Spent'].sum()
    last_purchase = customer_df.groupby('SKU_Code')['Delivered_date'].max()
    recency_days = (pd.Timestamp.today() - last_purchase).dt.days

    score_df = pd.DataFrame({
        'Frequency': freq,
        'Total_Quantity': qty,
        'Total_Spend': spend,
        'Recency': recency_days
    })
    score_df['Frequency_Score'] = score_df['Frequency'] / score_df['Frequency'].sum()
    score_df['Recency_Score'] = 1 - (score_df['Recency'] / score_df['Recency'].max())
    score_df['Combined_Score'] = (
        0.5 * score_df['Frequency_Score'] +
        0.3 * (score_df['Total_Quantity'] / score_df['Total_Quantity'].sum()) +
        0.2 * score_df['Recency_Score']
    )
    score_df['Probability (%)'] = (
        score_df['Combined_Score'] / score_df['Combined_Score'].sum() * 100
    ).round(2)

    avg_monthly_qty = (
        customer_df.groupby(['SKU_Code', 'Month'])['Delivered Qty']
        .sum()
        .groupby('SKU_Code')
        .mean()
    )
    avg_monthly_spend = (
        customer_df.groupby(['SKU_Code', 'Month'])['Total_Amount_Spent']
        .sum()
        .groupby('SKU_Code')
        .mean()
    )
    score_df['Expected Quantity'] = avg_monthly_qty.round(2)
    score_df['Expected Spend'] = avg_monthly_spend.round(2)

    def suggest(row):
        if row['Expected Spend'] > 5000:
            return 'Offer Discount'
        elif row['Expected Quantity'] > 10:
            return 'Free Gift'
        else:
            return 'Thank You Message'
    score_df['Suggestion'] = score_df.apply(suggest, axis=1)

    top3 = score_df.sort_values('Probability (%)', ascending=False).head(3)
    return top3[['Expected Quantity', 'Expected Spend', 'Probability (%)', 'Suggestion']].reset_index()

# --- Streamlit UI ---
logo = Image.open("logo.png")
st.image(logo, width=120)
st.markdown(
    "<h1 style='text-align: center;'>📊 Sales Intelligence & Product Recommendation Dashboard</h1>",
    unsafe_allow_html=True
)

st.sidebar.title("📂 Navigation")
section = st.sidebar.radio(
    "Choose a Section",
    [
        "📊 EDA Overview",
        "📉 Drop Detection",
        "👤 Customer Profiling",
        "🔁 Cross-Selling",
        "🔗 Brand Correlation",
        "🥇 Buyer Analysis",
        "📈 Retention & Moving Average",
        "🤖 Recommender System"
    ]
)

df_monthly = DF.groupby('Month')['Redistribution Value'].sum()

if section == "📊 EDA Overview":
    st.subheader("Sales Trends Over Time")
    data = df_monthly.copy().to_timestamp()
    st.line_chart(data)

    st.subheader("Top-Selling Products")
    top_prods = DF.groupby('SKU_Code')['Redistribution Value'].sum().nlargest(10)
    st.bar_chart(top_prods)

    st.subheader("Top Brands")
    top_brands = DF.groupby('Brand')['Redistribution Value'].sum().nlargest(10)
    st.bar_chart(top_brands)

elif section == "📉 Drop Detection":
    st.subheader("Brand-Level MoM Drop (>30%)")
    bm = DF.groupby(['Brand', 'Month'])['Redistribution Value'].sum().unstack(fill_value=0)
    mom = bm.pct_change(axis=1) * 100
    flags = mom < -30
    disp = mom.round(1).astype(str)
    disp[flags] += "% 🔻"
    disp[~flags] = ""
    st.dataframe(disp)

elif section == "👤 Customer Profiling":
    st.subheader("Customer Purchase Deep-Dive")
    customer_list = sorted(DF['Customer_Phone'].unique())
    sel = st.selectbox("Select Customer Phone:", customer_list)
    if sel:
        report = analyze_customer_purchases(sel)
        st.markdown(f"**Total Unique SKUs Bought:** {report['Total Unique SKUs Bought']}")
        st.markdown(f"**SKUs Bought:** {', '.join(report['SKUs Bought'])}")

        sku_df = pd.DataFrame.from_dict(
            report['Purchase Summary by SKU'], orient='index'
        ).rename_axis('SKU').reset_index()
        st.dataframe(sku_df, use_container_width=True)

        pred_df = predict_next_purchases(sel)
        st.subheader("Next-Purchase Predictions (Top 3 SKUs)")
        st.dataframe(pred_df.set_index('SKU_Code'), use_container_width=True)

elif section == "🔁 Cross-Selling":
    st.subheader("Brand Switching Patterns (Top 3 Alternatives)")
    last_purchase = DF.groupby(['Customer_Phone', 'Brand'])['Month'].max().reset_index()
    latest = DF['Month'].max()
    dropped = last_purchase[last_purchase['Month'] < latest]
    merged = DF.merge(dropped, on='Customer_Phone', suffixes=('', '_dropped'))
    switched = merged[
        (merged['Month'] > merged['Month_dropped']) &
        (merged['Brand'] != merged['Brand_dropped'])
    ]
    switches = switched.groupby(['Brand_dropped', 'Brand'])['Order_Id'].count().reset_index(name='Switch_Count')
    top3 = (
        switches.sort_values(['Brand_dropped', 'Switch_Count'], ascending=[True, False])
        .groupby('Brand_dropped')
        .head(3)
        .reset_index(drop=True)
    )
    st.dataframe(top3)

elif section == "🔗 Brand Correlation":
    st.subheader("Brand Correlation Matrix")
    ub = DF.groupby(['Customer_Phone', 'Brand'])['Order_Id'].count().unstack(fill_value=0)
    st.dataframe(ub.corr().round(2))

elif section == "🥇 Buyer Analysis":
    st.subheader("Top & Bottom Buyers (Latest Month)")
    latest_m = DF['Month'].max()
    bd = (
        DF[DF['Month'] == latest_m]
        .groupby('Customer_Phone')['Redistribution Value']
        .sum()
        .reset_index()
    )
    st.write("Top Buyers")
    st.dataframe(bd.nlargest(10, 'Redistribution Value'))
    st.write("Bottom Buyers")
    st.dataframe(bd.nsmallest(10, 'Redistribution Value'))

elif section == "📈 Retention & Moving Average":
    st.subheader("3-Month MA of Orders")
    ords = DF.groupby('Month')['Order_Id'].nunique()
    st.line_chart(ords.rolling(3).mean())

else:
    st.subheader("Hybrid Recommendations")
    uim = (
        DF.pivot_table(
            index='Customer_Phone',
            columns='SKU_Code',
            values='Redistribution Value',
            aggfunc='sum'
        ).fillna(0)
    )
    pf = (
        DF[['SKU_Code', 'Brand']]
        .drop_duplicates()
        .set_index('SKU_CODE')
    )
    pe = pd.get_dummies(pf, columns=['Brand'])
    us = cosine_similarity(uim)
    isim = cosine_similarity(pe)
    us_df = pd.DataFrame(us, index=uim.index, columns=uim.index)
    is_df = pd.DataFrame(isim, index=pe.index, columns=pe.index)

    def hybrid(c, top=5):
        if c not in uim.index:
            return pd.DataFrame({'Error': [f"Customer {c} not found"]})
        su = us_df[c]
        w = uim.T.dot(su).div(su.sum())
        b = uim.loc[c]
        bi = b[b > 0].index
        cs = is_df[bi].sum(axis=1)
        fs = 0.5 * w + 0.5 * cs
        fs = fs.drop(bi, errors='ignore')
        return fs.nlargest(top).reset_index().rename(columns={0: 'Score', 'index': 'Recommended SKU'})

    sel_c = st.selectbox("Select Customer for Recommendations:", uim.index)
    if st.button("Show Recommendations"):
        st.dataframe(hybrid(sel_c))
