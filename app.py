import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from PIL import Image

# --- Data Loading & Preprocessing ---
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

@st.cache_data
def load_model_preds():
    preds = pd.read_csv(
        "sku_predictions.csv",
        parse_dates=["last_purchase_date", "pred_next_date"],
    )
    preds = preds.rename(columns={
        "pred_next_date": "Next Purchase Date",
        "pred_spend":      "Expected Spend",
        "pred_qty":        "Expected Quantity",
        "probability":     "Probability"
    })
    preds["Next Purchase Date"] = preds["Next Purchase Date"].dt.date
    preds["Expected Spend"] = preds["Expected Spend"].round(0).astype(int)
    preds["Expected Quantity"] = preds["Expected Quantity"].round(0).astype(int)
    preds["Probability"]   = (preds["Probability"] * 100).round(1)
    def suggest(p):
        if p >= 70:
            return "Follow-up/Alert"
        if p >= 50:
            return "Cross Sell"
        return "Discount"
    preds["Suggestion"] = preds["Probability"].apply(suggest)
    if "last_purchase_date" in preds.columns:
        preds = preds.drop(columns=["last_purchase_date"])
    return preds

# --- Heuristic Profiling Functions ---
def analyze_customer_purchases(customer_phone):
    df = DF[DF['Customer_Phone'] == customer_phone].copy()
    df.sort_values('Delivered_date', inplace=True)
    skus = df['SKU_Code'].unique().tolist()
    last_purchase = df.groupby('SKU_Code')['Delivered_date'].max().dt.strftime('%Y-%m-%d').to_dict()
    monthly_qty = df.groupby(['SKU_Code','Month'])['Delivered Qty'].sum().groupby('SKU_Code').mean().round(2).to_dict()
    avg_interval = {}
    for sku, grp in df.groupby('SKU_Code'):
        dates = grp['Delivered_date'].drop_duplicates().sort_values()
        avg_interval[sku] = round((dates.diff().dt.days.dropna() / 30.44).mean(), 2) if len(dates)>1 else 'One'
    monthly_spend = df.groupby(['SKU_Code','Month'])['Total_Amount_Spent'].sum().groupby('SKU_Code').mean().round(2).to_dict()
    report = {
        'Customer Phone': customer_phone,
        'Total Unique SKUs Bought': len(skus),
        'SKUs Bought': skus,
        'Purchase Summary by SKU': {}
    }
    for sku in skus:
        report['Purchase Summary by SKU'][sku] = {
            'Last Purchase Date': last_purchase.get(sku, 'N/A'),
            'Avg Monthly Quantity': monthly_qty.get(sku, 0),
            'Avg Purchase Interval (Months)': avg_interval.get(sku, 'N/A'),
            'Avg Monthly Spend': monthly_spend.get(sku, 0)
        }
    return report

def predict_next_purchases(customer_phone):
    df = DF[DF['Customer_Phone'] == customer_phone].copy()
    last_purchase = df.groupby('SKU_Code')['Delivered_date'].max()
    avg_interval_days = {}
    for sku, grp in df.groupby('SKU_Code'):
        dates = grp['Delivered_date'].drop_duplicates().sort_values()
        avg_interval_days[sku] = int(dates.diff().dt.days.dropna().mean()) if len(dates)>1 else np.nan
    avg_qty   = df.groupby(['SKU_Code','Month'])['Delivered Qty'].sum().groupby('SKU_Code').mean().round(0)
    avg_spend = df.groupby(['SKU_Code','Month'])['Total_Amount_Spent'].sum().groupby('SKU_Code').mean().round(0)
    score_df = pd.DataFrame({
        'Last Purchase Date': last_purchase.dt.date,
        'Avg Interval Days': pd.Series(avg_interval_days),
        'Expected Quantity': avg_qty,
        'Expected Spend': avg_spend
    }).dropna(subset=['Avg Interval Days'])
    score_df['Next Purchase Date'] = (
        pd.to_datetime(score_df['Last Purchase Date']) + pd.to_timedelta(score_df['Avg Interval Days'], unit='D')
    ).dt.date
    return score_df.sort_values('Avg Interval Days').head(3)[['Next Purchase Date','Expected Spend','Expected Quantity']]

# --- Load Data ---
DF = load_sales_data()
PRED_DF = load_model_preds()

# --- Streamlit App UI ---
st.set_page_config(page_title="Sales Intelligence Dashboard", layout="wide")
logo = Image.open("logo.png")
st.sidebar.image(logo, width=80)
st.sidebar.title("🚀 Sales Insights")
section = st.sidebar.radio(
    "Select Section:",
    [
        "📊 EDA Overview",
        "📉 Drop Detection",
        "👤 Customer Profiling",
        "👤 Model Predictions",
        "🔁 Cross-Selling",
        "🔗 Brand Correlation",
        "🥇 Buyer Analysis",
        "📈 Retention",
        "🤖 Recommender"
    ]
)
st.title("📊 Sales Intelligence Dashboard")

# --- EDA Overview ---
if section == "📊 EDA Overview":
    st.subheader("Exploratory Data Analysis")
    tabs = st.tabs([
        "Top Revenue","Top Quantity","Buyer Types","Buyer Trends",
        "SKU Trends","Qty vs Revenue","Avg Order Value","Lifetime Value",
        "SKU Share %","SKU Pairs"
    ])
    
    with tabs[0]:
        data = DF.groupby("SKU_Code")["Redistribution Value"].sum().nlargest(10)
        st.bar_chart(data)
    with tabs[1]:
        data = DF.groupby("SKU_Code")["Delivered Qty"].sum().nlargest(10)
        st.bar_chart(data)
    with tabs[2]:
        counts = DF.groupby("Customer_Phone")["Delivered_date"].nunique()
        summary = (counts==1).map({True:"One-time",False:"Repeat"}).value_counts()
        st.bar_chart(summary)
    with tabs[3]:
        df_b = DF.copy()
        df_b["MonthTS"] = df_b["Month"].dt.to_timestamp()
        top5= df_b.groupby("Customer_Phone")["Redistribution Value"].sum().nlargest(5).index
        trend=df_b[df_b["Customer_Phone"].isin(top5)]
        trend=trend.groupby(["MonthTS","Customer_Phone"]).sum()["Redistribution Value"].unstack()
        st.line_chart(trend)
    with tabs[4]:
        df_s=DF.copy(); df_s["MonthTS"]=df_s["Month"].dt.to_timestamp()
        top5= df_s.groupby("SKU_Code")["Delivered Qty"].sum().nlargest(5).index
        trend=df_s[df_s["SKU_Code"].isin(top5)]
        trend=trend.groupby(["MonthTS","SKU_Code"]).sum()["Delivered Qty"].unstack()
        st.line_chart(trend)
    with tabs[5]:
        monthly_summary = DF.groupby("Month")[ ["Delivered Qty","Redistribution Value"] ].sum().reset_index()
        monthly_summary["MonthTS"] = monthly_summary["Month"].dt.to_timestamp()
        qty_line = alt.Chart(monthly_summary).mark_line(point=True).encode(
            x=alt.X("MonthTS:T", title="Month"),
            y=alt.Y("Delivered Qty:Q", axis=alt.Axis(title="Total Quantity", titleColor="royalblue")),
            color=alt.value("royalblue")
        )
        rev_line= alt.Chart(monthly_summary).mark_line(point=True).encode(
            x="MonthTS:T",
            y=alt.Y("Redistribution Value:Q", axis=alt.Axis(title="Total Revenue", titleColor="orange")),
            color=alt.value("orange")
        )
        dual_axis = alt.layer(qty_line, rev_line).resolve_scale(y="independent").properties(height=400)
        st.altair_chart(dual_axis, use_container_width=True)
    with tabs[6]:
        data=DF.groupby("Customer_Phone")["Redistribution Value"].mean().nlargest(10)
        st.bar_chart(data)
    with tabs[7]:
        data=DF.groupby("Customer_Phone")["Redistribution Value"].sum().nlargest(10)
        st.bar_chart(data)
    with tabs[8]:
        share=DF.groupby("SKU_Code")["Delivered Qty"].sum()/DF["Delivered Qty"].sum()*100
        st.bar_chart(share.nlargest(10))
    with tabs[9]:
        from itertools import combinations; from collections import Counter
        df_p=DF.copy(); df_p["Order_ID"]=df_p["Customer_Phone"].astype(str)+"_"+df_p["Delivered_date"].astype(str)
        sets=df_p.groupby("Order_ID")["SKU_Code"].apply(set)
        cnt=Counter()
        for s in sets:
            if len(s)>1:
                for pair in combinations(sorted(s),2): cnt[pair]+=1
        top_pairs=pd.Series(cnt).nlargest(10)
        df_pairs=top_pairs.to_frame(name="Count"); df_pairs.index=df_pairs.index.map(lambda t:f"{t[0]} & {t[1]}")
        st.bar_chart(df_pairs)

# --- Drop Detection ---
elif section=="📉 Drop Detection":
    bm=DF.groupby(['Brand','Month'])['Redistribution Value'].sum().unstack(fill_value=0)
    mom=bm.pct_change(axis=1)*100; st.dataframe(mom.round(1).replace({np.nan:""}))

# --- Customer Profiling ---
elif section=="👤 Customer Profiling":
    st.subheader("Customer Purchase Deep-Dive")
    cust=st.selectbox("Select Customer Phone",sorted(DF['Customer_Phone'].unique()))
    if cust:
        report=analyze_customer_purchases(cust)
        st.markdown(f"**Total Unique SKUs Bought:** {report['Total Unique SKUs Bought']}")
        st.markdown(f"**SKUs Bought:** {', '.join(report['SKUs Bought'])}")
        sku_df=pd.DataFrame.from_dict(report['Purchase Summary by SKU'],orient='index')
        sku_df=sku_df.rename_axis('SKU_Code').reset_index()
        st.dataframe(sku_df,use_container_width=True)
        st.subheader("Next-Purchase Predictions (Heuristic)")
        pred_df=predict_next_purchases(cust)
        st.dataframe(pred_df.set_index('SKU_Code'),use_container_width=True)

# --- Model Predictions ---
elif section=="👤 Model Predictions":
    st.subheader("Next-Purchase Model Predictions")
    cust=st.selectbox("Customer",sorted(PRED_DF['Customer_Phone'].unique()))
    if cust:
        p=PRED_DF[PRED_DF['Customer_Phone']==cust].drop(columns=['Customer_Phone']).set_index('SKU_Code')
        p['Probability']=p['Probability'].map(lambda x:f"{x:.1f}%")
        st.dataframe(p,use_container_width=True)

# --- Cross-Selling ---
elif section=="🔁 Cross-Selling":
    st.subheader("Brand Switching Patterns (Top 3)")
    lp=DF.groupby(['Customer_Phone','Brand'])['Month'].max().reset_index()
    dropped=lp[lp['Month']<lp['Month'].max()]
    merged=DF.merge(dropped,on='Customer_Phone',suffixes=('','_dropped'))
    switched=merged[(merged['Month']>merged['Month_dropped'])&(merged['Brand']!=merged['Brand_dropped'])]
    switches=switched.groupby(['Brand_dropped','Brand'])['Order_Id'].count().reset_index(name='Count')
    top3=switches.sort_values(['Brand_dropped','Count'],ascending=[True,False]).groupby('Brand_dropped').head(3)
    st.dataframe(top3,use_container_width=True)

# --- Brand Correlation ---
elif section=="🔗 Brand Correlation":
    st.subheader("Brand Correlation Matrix")
    mat=DF.groupby(['Customer_Phone','Brand'])['Order_Id'].count().unstack(fill_value=0)
    st.dataframe(mat.corr().round(2),use_container_width=True)

# --- Buyer Analysis ---
elif section=="🥇 Buyer Analysis":
    st.subheader("Top & Bottom Buyers (Latest Month)")
    m=DF['Month'].max()
    bd=DF[DF['Month']==m].groupby('Customer_Phone')['Redistribution Value'].sum()
    st.bar_chart(bd.nlargest(10))

# --- Retention ---
elif section=="📈 Retention":
    st.subheader("3-Month MA of Unique Orders")
    orders=DF.groupby('Month')['Order_Id].nunique()
    st.line_chart(orders.rolling(3).mean())

# --- Recommender ---
elif section=="🤖 Recommender":
    st.subheader("Hybrid SKU Recommendations")
    uim=DF.pivot_table(index='Customer_Phone',columns='SKU_Code',values='Redistribution Value',aggfunc='sum').fillna(0)
    pf=pd.get_dummies(DF[['SKU_Code','Brand']].drop_duplicates(),columns=['Brand']).set_index('SKU_Code')
    us=cosine_similarity(uim)
    isim=cosine_similarity(pf)
    sel=st.selectbox("Select Customer",uim.index)
    if st.button("Recommend"):
        w=uim.T.dot(us[uim.index.get_loc(sel)]).drop(sel)
        cs=isim[uim.columns].sum(axis=1)
        scores=(0.5*w+0.5*cs).nlargest(5)
        st.dataframe(scores.reset_index().rename(columns={0:'Score','index':'SKU_Code'}),use_container_width=True)
