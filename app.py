import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from PIL import Image

logo = Image.open("logo.png")
st.image(logo, width=120)
st.markdown("<h1 style='text-align: center;'>📊 Sales Intelligence & Product Recommendation Dashboard</h1>", unsafe_allow_html=True)

# Load and preprocess data
def load_data():
    df = pd.read_csv("/content/drive/MyDrive/Data Analysis - Sample File.csv")
    df['Redistribution Value'] = df['Redistribution Value'].str.replace(',', '', regex=False).astype(float)
    df['Delivered_date'] = pd.to_datetime(df['Delivered_date'], errors='coerce')
    df['Month'] = df['Delivered_date'].dt.to_period('M')
    return df

DF = load_data()

# Sidebar navigation
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
    bm = DF.groupby(['Brand','Month'])['Redistribution Value'].sum().unstack(fill_value=0)
    mom = bm.pct_change(axis=1) * 100
    flags = mom < -30
    disp = mom.round(1).astype(str)
    disp[flags] += "% 🔻"
    disp[~flags] = ""
    st.dataframe(disp)

elif section == "👤 Customer Profiling":
    st.subheader("Customer RFM, Next Purchase & Discounts")
    # RFM calculation
    max_date = DF['Delivered_date'].max()
    last = DF.groupby('Customer_Phone')['Delivered_date'].max()
    recency = (max_date - last).dt.days
    freq = DF.groupby('Customer_Phone')['Order_Id'].nunique()
    val = DF.groupby(['Customer_Phone','Order_Id'])['Redistribution Value'].sum().reset_index()
    monetary = val.groupby('Customer_Phone')['Redistribution Value'].mean()
    # Assemble RFM DataFrame
    rfm = pd.DataFrame({
        'Customer_Phone': recency.index,
        'Recency': recency.values,
        'Frequency': freq.values,
        'Monetary': monetary.values
    })
    # Quantiles for segmentation and discounts
    q = {col: rfm[col].quantile([0.25,0.5,0.75]).to_dict() for col in ['Recency','Frequency','Monetary']}
    def assign_segment(row):
        if row['Recency'] <= q['Recency'][0.25] and row['Frequency'] >= q['Frequency'][0.75] and row['Monetary'] >= q['Monetary'][0.75]: return 'Best Customers'
        if row['Recency'] >= q['Recency'][0.75] and row['Frequency'] <= q['Frequency'][0.25]: return 'At-Risk Customers'
        if row['Recency'] >= q['Recency'][0.75] and row['Monetary'] >= q['Monetary'][0.75]: return 'Big Spenders Dropping Off'
        if q['Recency'][0.25] < row['Recency'] <= q['Recency'][0.75] and row['Frequency'] >= q['Frequency'][0.5]: return 'Potential Loyalists'
        return 'Others'
    rfm['Segment'] = rfm.apply(assign_segment, axis=1)
    # Discount logic
    med_best = rfm.loc[rfm['Segment']=='Best Customers','Monetary'].median()
    med_loyal = rfm.loc[rfm['Segment']=='Potential Loyalists','Monetary'].median()
    rfm['Recommended_Discount'] = 0
    rfm.loc[(rfm['Segment']=='Best Customers')&(rfm['Monetary']<med_best),'Recommended_Discount']=10
    rfm.loc[(rfm['Segment']=='Potential Loyalists')&(rfm['Monetary']<med_loyal),'Recommended_Discount']=5
    # Next purchase prediction
    grouped = DF.sort_values(['Customer_Phone','Delivered_date']).groupby('Customer_Phone')
    last_dates = grouped['Delivered_date'].last().rename('last_purchase_date')
    inter_days = grouped['Delivered_date'].diff().dt.days
    avg_days = inter_days.groupby(DF['Customer_Phone']).mean().fillna(inter_days.mean()).rename('avg_inter_purchase_days')
    summary = pd.concat([last_dates, avg_days], axis=1)
    summary['predicted_next_purchase'] = summary['last_purchase_date'] + pd.to_timedelta(summary['avg_inter_purchase_days'], unit='D')
    freq_df = DF.groupby(['Customer_Phone','SKU_Code']).size().rename('count').reset_index()
    idx = freq_df.groupby('Customer_Phone')['count'].idxmax()
    likely = freq_df.loc[idx, ['Customer_Phone','SKU_Code']].rename(columns={'SKU_Code':'likely_next_SKU'})
    next_df = summary.reset_index().merge(likely, on='Customer_Phone')
    # Interactive display
    ids = rfm['Customer_Phone'].tolist()
    sel = st.selectbox("Select Customer Phone:", ids)
    cust = rfm[rfm['Customer_Phone']==sel].iloc[0]
    st.metric("Recency (days)",cust['Recency'])
    st.metric("Frequency (orders)",cust['Frequency'])
    st.metric("Avg Spend (NGN)",f"₦{cust['Monetary']:.2f}")
    st.metric("Segment",cust['Segment'])
    if cust['Recommended_Discount']>0:
        st.metric("Recommended Discount",f"{cust['Recommended_Discount']}%")
    # Next purchase metrics
    np_rec = next_df[next_df['Customer_Phone']==sel].iloc[0]
    st.metric("Next Purchase Date", np_rec['predicted_next_purchase'].date())
    st.write(f"Likely Next SKU: {np_rec['likely_next_SKU']}")

elif section == "🔁 Cross-Selling":
    st.subheader("Brand Switching Patterns (Top 3 Alternatives)")
    # Identify last purchase month per customer-brand
    last_purchase = DF.groupby(['Customer_Phone','Brand'])['Month'].max().reset_index()
    # Customers who dropped a brand this period
    latest = DF['Month'].max()
    dropped = last_purchase[last_purchase['Month']<latest]
    # Subsequent different-brand purchases
    merged = DF.merge(dropped,on='Customer_Phone',suffixes=('','_dropped'))
    switched = merged[(merged['Month']>merged['Month_dropped'])&(merged['Brand']!=merged['Brand_dropped'])]
    switches = switched.groupby(['Brand_dropped','Brand'])['Order_Id'].count().reset_index(name='Switch_Count')
    top3 = switches.sort_values(['Brand_dropped','Switch_Count'],ascending=[True,False]).groupby('Brand_dropped').head(3).reset_index(drop=True)
    st.dataframe(top3)

elif section == "🔗 Brand Correlation":
    st.subheader("Brand Correlation Matrix")
    ub = DF.groupby(['Customer_Phone','Brand'])['Order_Id'].count().unstack(fill_value=0)
    st.dataframe(ub.corr().round(2))

elif section == "🥇 Buyer Analysis":
    st.subheader("Top & Bottom Buyers (Latest Month)")
    latest_m = DF['Month'].max()
    bd = DF[DF['Month']==latest_m].groupby('Customer_Phone')['Redistribution Value'].sum().reset_index()
    st.write("Top Buyers")
    st.dataframe(bd.nlargest(10,'Redistribution Value'))
    st.write("Bottom Buyers")
    st.dataframe(bd.nsmallest(10,'Redistribution Value'))

elif section == "📈 Retention & Moving Average":
    st.subheader("3-Month MA of Orders")
    ords = DF.groupby('Month')['Order_Id'].nunique()
    st.line_chart(ords.rolling(3).mean())

else:
    st.subheader("Hybrid Recommendations")
    uim = DF.pivot_table(index='Customer_Phone',columns='SKU_Code',values='Redistribution Value',aggfunc='sum').fillna(0)
    pf = DF[['SKU_Code','Brand']].drop_duplicates().set_index('SKU_Code')
    pe = pd.get_dummies(pf,columns=['Brand'])
    us = cosine_similarity(uim)
    isim = cosine_similarity(pe)
    us_df = pd.DataFrame(us,index=uim.index,columns=uim.index)
    is_df = pd.DataFrame(isim,index=pe.index,columns=pe.index)
    def hybrid(c,top=5):
        if c not in uim.index:
            return pd.DataFrame({'Error':[f"Customer {c} not found"]})
        su = us_df[c]
        w = uim.T.dot(su).div(su.sum())
        b = uim.loc[c]
        bi = b[b>0].index
        cs = is_df[bi].sum(axis=1)
        fs = 0.5*w+0.5*cs
        fs = fs.drop(bi,errors='ignore')
        return fs.nlargest(top).reset_index().rename(columns={0:'Score','index':'Recommended SKU'})
    sel_c = st.selectbox("Select Customer",uim.index)
    if st.button("Show Recommendations"):
        st.dataframe(hybrid(sel_c))
