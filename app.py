import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from PIL import Image

# --- Load and preprocess sales data ---
@st.cache_data
def load_sales_data():
    df = pd.read_csv("cleaned_data_analysis.csv")
    df['Redistribution Value'] = (
        df['Redistribution Value']
          .str.replace(',', '', regex=False)
          .astype(float)
    )
    df['Delivered_date'] = pd.to_datetime(
        df['Delivered_date'], errors='coerce', dayfirst=True
    )
    df['Month'] = df['Delivered_date'].dt.to_period('M')
    df['Delivered Qty'] = df['Delivered Qty'].fillna(0)
    df['Total_Amount_Spent'] = df['Redistribution Value'] * df['Delivered Qty']
    return df

# --- Load model predictions with updated formatting & logic ---
@st.cache_data
def load_model_preds():
    preds = pd.read_csv(
        "sku_predictions.csv",
        parse_dates=["last_purchase_date", "pred_next_date"]
    )
    # Drop unnecessary columns
    cols_to_drop = [c for c in ["last_purchase_date","probability","suggestion"] if c in preds.columns]
    preds = preds.drop(columns=cols_to_drop)
    preds = preds.rename(columns={
        "pred_next_date":     "Next Purchase Date",
        "pred_spend":          "Expected Spend",
        "pred_qty":            "Expected Quantity"
    })
    preds["Next Purchase Date"] = preds["Next Purchase Date"].dt.date
    preds["Expected Spend"] = preds["Expected Spend"].round(0).astype(int)
    preds["Expected Quantity"] = preds["Expected Quantity"].round(0).astype(int)
    return preds

DF = load_sales_data()
PRED_DF = load_model_preds()

# --- Analysis functions ---
def analyze_customer_purchases(customer_phone):
    df = DF[DF['Customer_Phone'] == customer_phone].copy()
    if df.empty:
        return {}
    df.sort_values('Delivered_date', inplace=True)
    skus = df['SKU_Code'].unique().tolist()
    last_purchase = df.groupby('SKU_Code')['Delivered_date'].max().dt.strftime('%Y-%m-%d').to_dict()
    monthly_qty = df.groupby(['SKU_Code','Month'])['Delivered Qty'].sum().groupby('SKU_Code').mean().round(2).to_dict()
    avg_interval = {}
    for sku, grp in df.groupby('SKU_Code'):
        dates = grp['Delivered_date'].drop_duplicates().sort_values()
        if len(dates) > 1:
            avg_interval[sku] = round((dates.diff().dt.days.dropna() / 30.44).mean(), 2)
        else:
            avg_interval[sku] = "One"
    monthly_spend = df.groupby(['SKU_Code','Month'])['Total_Amount_Spent'].sum().groupby('SKU_Code').mean().round(2).to_dict()
    report = {'Customer Phone': customer_phone, 'Total Unique SKUs Bought': len(skus), 'SKUs Bought': skus, 'Purchase Summary by SKU': {}}
    for sku in skus:
        report['Purchase Summary by SKU'][sku] = {
            'Last Purchase Date':           last_purchase.get(sku, 'N/A'),
            'Avg Monthly Quantity':         monthly_qty.get(sku, 0),
            'Avg Purchase Interval (Months)': avg_interval.get(sku, 'N/A'),
            'Avg Monthly Spend':            monthly_spend.get(sku, 0)
        }
    return report

def predict_next_purchases(customer_phone):
    df = DF[DF['Customer_Phone'] == customer_phone].copy()
    if df.empty:
        return pd.DataFrame()
    last_purchase = df.groupby('SKU_Code')['Delivered_date'].max()
    avg_interval_days = {}
    for sku, grp in df.groupby('SKU_Code'):
        dates = grp['Delivered_date'].drop_duplicates().sort_values()
        if len(dates) > 1:
            avg_interval_days[sku] = int(dates.diff().dt.days.dropna().mean())
        else:
            avg_interval_days[sku] = np.nan
    avg_qty = df.groupby(['SKU_Code','Month'])['Delivered Qty'].sum().groupby('SKU_Code').mean().round(0)
    avg_spend = df.groupby(['SKU_Code','Month'])['Total_Amount_Spent'].sum().groupby('SKU_Code').mean().round(0)
    score_df = pd.DataFrame({
        'Last Purchase Date':   last_purchase.dt.date,
        'Avg Interval Days':    pd.Series(avg_interval_days),
        'Expected Quantity':    avg_qty,
        'Expected Spend':       avg_spend
    }).dropna(subset=['Avg Interval Days'])
    # ensure SKU_Code is name of index before reset
    result = (
        score_df
        .sort_values('Avg Interval Days')
        .head(3)[['Next Purchase Date','Expected Spend','Expected Quantity']] if 'Next Purchase Date' in score_df.columns else None
    )
    # actually compute Next Purchase Date
    score_df['Next Purchase Date'] = pd.to_datetime(score_df['Last Purchase Date']) + pd.to_timedelta(score_df['Avg Interval Days'], unit='D')
    result = (
        score_df
        .sort_values('Avg Interval Days')
        .head(3)[['Next Purchase Date','Expected Spend','Expected Quantity']]
        .rename_axis('SKU_Code')
        .reset_index()
    )
    return result

# --- Streamlit UI ---
logo = Image.open("logo.png")
st.image(logo, width=120)
st.markdown("<h1 style='text-align: center;'>📊 Sales Intelligence & Product Recommendation Dashboard</h1>", unsafe_allow_html=True)
section = st.sidebar.radio("Choose a Section", [
    "📊 EDA Overview","📉 Drop Detection","👤 Customer Profiling","👤 Customer Profiling (Model Prediction)",
    "🔁 Cross-Selling","🔗 Brand Correlation","🥇 Buyer Analysis","📈 Retention & Moving Average","🤖 Recommender System"
])

df_monthly = DF.groupby('Month')['Redistribution Value'].sum()
if section == "📊 EDA Overview":
    st.subheader("Sales Trends Over Time")
    st.line_chart(df_monthly.to_timestamp())
    st.subheader("Top-Selling Products")
    st.bar_chart(DF.groupby('SKU_Code')['Redistribution Value'].sum().nlargest(10))
    st.subheader("Top Brands")
    st.bar_chart(DF.groupby('Brand')['Redistribution Value'].sum().nlargest(10))
elif section == "📉 Drop Detection":
    st.subheader("Brand-Level MoM Drop (>30%)")
    bm = DF.groupby(['Brand','Month'])['Redistribution Value'].sum().unstack(fill_value=0)
    mom = bm.pct_change(axis=1) * 100
    flags = mom < -30
    disp = mom.round(1).astype(str)
    disp[flags] += "% 🔻"
    disp[~flags] = ""
    st.dataframe(disp)
elif section == "👤 Customer Profiling":
    st.subheader("Customer Purchase Deep-Dive")
    cust = st.selectbox("Select Customer Phone:", sorted(DF['Customer_Phone'].unique()))
    if cust:
        report = analyze_customer_purchases(cust)
        st.markdown(f"**Total Unique SKUs Bought:** {report['Total Unique SKUs Bought']}")
        st.markdown(f"**SKUs Bought:** {', '.join(report['SKUs Bought'])}")
        sku_df = pd.DataFrame.from_dict(report['Purchase Summary by SKU'], orient='index').rename_axis('SKU_Code').reset_index()
        st.dataframe(sku_df, use_container_width=True)
        pred_df = predict_next_purchases(cust)
        st.subheader("Next-Purchase Predictions (Heuristic)")
        st.dataframe(pred_df.set_index('SKU_Code'), use_container_width=True)
elif section == "🔁 Cross-Selling":
    st.subheader("Brand Switching Patterns (Top 3 Alternatives)")
    last_purchase = DF.groupby(['Customer_Phone','Brand'])['Month'].max().reset_index()
    latest = DF['Month'].max()
    dropped = last_purchase[last_purchase['Month'] < latest]
    merged = DF.merge(dropped, on='Customer_Phone', suffixes=('','_dropped'))
    switched = merged[(merged['Month'] > merged['Month_dropped']) & (merged['Brand'] != merged['Brand_dropped'])]
    switches = switched.groupby(['Brand_dropped','Brand'])['Order_Id'].count().reset_index(name='Switch_Count')
    top3 = switches.sort_values(['Brand_dropped','Switch_Count'], ascending=[True,False]).groupby('Brand_dropped').head(3).reset_index(drop=True)
    st.dataframe(top3)
elif section == "🔗 Brand Correlation":
    st.subheader("Brand Correlation Matrix")
    ub = DF.groupby(['Customer_Phone','Brand'])['Order_Id'].count().unstack(fill_value=0)
    st.dataframe(ub.corr().round(2))
elif section == "🥇 Buyer Analysis":
    st.subheader("Top & Bottom Buyers (Latest Month)")
    latest_m = DF['Month'].max()
    bd = DF[DF['Month'] == latest_m].groupby('Customer_Phone')['Redistribution Value'].sum().reset_index()
    st.write("Top Buyers")
    st.dataframe(bd.nlargest(10, 'Redistribution Value'))
    st.write("Bottom Buyers")
    st.dataframe(bd.nsmallest(10, 'Redistribution Value'))
elif section == "📈 Retention & Moving Average":
    st.subheader("3-Month MA of Orders")
    ords = DF.groupby('Month')['Order_Id'].nunique()
    st.line_chart(ords.rolling(3).mean())
elif section == "🤖 Recommender System":
    st.subheader("Hybrid Recommendations")
    uim = DF.pivot_table(index='Customer_Phone', columns='SKU_Code', values='Redistribution Value', aggfunc='sum').fillna(0)
    pf = DF[['SKU_Code','Brand']].drop_duplicates().set_index('SKU_Code')
    pe = pd.get_dummies(pf, columns=['Brand'])
    us_df = pd.DataFrame(cosine_similarity(uim), index=uim.index, columns=uim.index)
    is_df = pd.DataFrame(cosine_similarity(pe), index=pe.index, columns=pe.index)
    sel_c = st.selectbox("Select Customer for Recommendations:", uim.index)
    if st.button("Show Recommendations"):
        b = uim.loc[sel_c]
        w = uim.T.dot(us_df[sel_c]).div(us_df[sel_c].sum())
        bi = b[b > 0].index
        cs = is_df[bi].sum(axis=1)
        fs = 0.5 * w + 0.5 * cs
        fs = fs.drop(bi, errors='ignore')
        st.dataframe(fs.nlargest(5).reset_index().rename(columns={0:'Score','index':'Recommended SKU'}))
elif section == "👤 Customer Profiling (Model Prediction)":
    st.subheader("Next-Purchase Model Predictions")
    cust = st.selectbox("Select Customer Phone:", sorted(PRED_DF['Customer_Phone'].unique()))
    if cust:
        pred_df = PRED_DF[PRED_DF['Customer_Phone'] == cust].drop(columns=["Customer_Phone"]).set_index("SKU_Code")
        st.dataframe(pred_df, use_container_width=True)
