import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn as sns
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
        "pred_next_date":     "Next Purchase Date",
        "pred_spend":          "Expected Spend",
        "pred_qty":            "Expected Quantity",
        "probability":         "Probability"
    })
    preds["Next Purchase Date"] = preds["Next Purchase Date"].dt.date
    preds["Expected Spend"] = preds["Expected Spend"].round(0).astype(int)
    preds["Expected Quantity"] = preds["Expected Quantity"].round(0).astype(int)
    preds["Probability"] = (preds["Probability"] * 100).round(1)
    def suggest(row):
        p = row["Probability"]
        if p >= 70:
            return "Follow-up/Alert"
        elif p >= 50:
            return "Cross Sell"
        else:
            return "Discount"
    preds["Suggestion"] = preds.apply(suggest, axis=1)
    if "last_purchase_date" in preds.columns:
        preds = preds.drop(columns=["last_purchase_date"])
    return preds

DF = load_sales_data()
PRED_DF = load_model_preds()

# --- EDA Plotting Functions ---
def plot_top_skus_by_value(df, top_n=10):
    data = df.groupby("SKU_Code")["Redistribution Value"].sum().nlargest(top_n)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=data.values, y=data.index, ax=ax, ci=None)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.set_title(f"Top {top_n} SKUs by Revenue", fontsize=14)
    ax.set_xlabel("Total Spend", fontsize=12)
    ax.set_ylabel("")
    for i, v in enumerate(data.values):
        ax.text(v + v*0.01, i, f'{int(v):,}', va='center', fontsize=10)
    plt.tight_layout()
    return fig

def plot_top_skus_by_qty(df, top_n=10):
    data = df.groupby("SKU_Code")["Delivered Qty"].sum().nlargest(top_n)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=data.values, y=data.index, ax=ax, ci=None)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.set_title(f"Top {top_n} SKUs by Quantity", fontsize=14)
    ax.set_xlabel("Total Units", fontsize=12)
    ax.set_ylabel("")
    for i, v in enumerate(data.values):
        ax.text(v + v*0.01, i, f'{int(v):,}', va='center', fontsize=10)
    plt.tight_layout()
    return fig

def plot_repeat_vs_one_time(df):
    counts = (
        df.groupby("Customer_Phone")["Delivered_date"]
          .nunique()
          .reset_index(name="Purchase Count")
    )
    counts["Type"] = counts["Purchase Count"].apply(lambda x: "One-time" if x==1 else "Repeat")
    summary = counts["Type"].value_counts().reset_index()
    summary.columns = ["Type", "Customer Count"]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=summary, x="Type", y="Customer Count", ax=ax, ci=None)
    ax.set_title("Repeat vs. One-Time Buyers", fontsize=14)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height()):,}',
                    (p.get_x() + p.get_width()/2, p.get_height()),
                    ha='center', va='bottom')
    plt.tight_layout()
    return fig

def plot_monthly_top5_buyers(df):
    top5 = df.groupby("Customer_Phone")["Redistribution Value"].sum().nlargest(5).index
    df5 = df[df["Customer_Phone"].isin(top5)]
    data = (
        df5
        .groupby(['Month','Customer_Phone'])['Redistribution Value']
        .sum()
        .reset_index()
    )
    # ← Add this line:
    data['Month'] = data['Month'].astype(str)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=data,
        x="Month", y="Redistribution Value",
        hue="Customer_Phone",
        marker="o", ax=ax
    )
    ax.set_title("Monthly Value Trend: Top 5 Buyers", fontsize=14)
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Spend", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_monthly_top5_skus(df):
    top5 = df.groupby("SKU_Code")["Delivered Qty"].sum().nlargest(5).index
    df5 = df[df["SKU_Code"].isin(top5)]
    data = (
        df5
        .groupby(['Month','SKU_Code'])['Delivered Qty']
        .sum()
        .reset_index()
    )
    # ← And here:
    data['Month'] = data['Month'].astype(str)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=data,
        x="Month", y="Delivered Qty",
        hue="SKU_Code",
        marker="o", ax=ax
    )
    ax.set_title("Monthly Qty Trend: Top 5 SKUs", fontsize=14)
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Quantity", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_dual_axis_quantity_revenue(df):
    summary = df.groupby("Month")[["Delivered Qty","Redistribution Value"]].sum().reset_index()
    # ← And here too:
    summary['Month'] = summary['Month'].astype(str)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=summary, x="Month", y="Delivered Qty", marker="o", ax=ax1)
    ax2 = ax1.twinx()
    sns.lineplot(data=summary, x="Month", y="Redistribution Value", marker="s", ax=ax2)
    ax1.set_title("Monthly Quantity vs Revenue", fontsize=14)
    ax1.set_xlabel("Month", fontsize=12)
    ax1.set_ylabel("Qty", fontsize=12)
    ax2.set_ylabel("Revenue", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_top_customers_avg_order_value(df):
    data = df.groupby("Customer_Phone")["Redistribution Value"].mean().nlargest(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=data.values, y=data.index, ax=ax, ci=None)
    ax.set_title("Top 10 by Avg Order Value", fontsize=14)
    ax.set_xlabel("Avg Spend", fontsize=12); ax.set_ylabel("")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    for i, v in enumerate(data.values):
        ax.text(v+v*0.01, i, f'{int(v):,}', va='center', fontsize=10)
    plt.tight_layout()
    return fig

def plot_top_customers_lifetime_value(df):
    data = df.groupby("Customer_Phone")["Redistribution Value"].sum().nlargest(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=data.values, y=data.index, ax=ax, ci=None)
    ax.set_title("Top 10 by Lifetime Value", fontsize=14)
    ax.set_xlabel("Total Spend", fontsize=12); ax.set_ylabel("")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    for i, v in enumerate(data.values):
        ax.text(v+v*0.01, i, f'{int(v):,}', va='center', fontsize=10)
    plt.tight_layout()
    return fig

def plot_sku_concentration_pct(df):
    data = (df.groupby("SKU_Code")["Delivered Qty"].sum() /
            df["Delivered Qty"].sum() * 100).nlargest(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=data.values, y=data.index, ax=ax, ci=None)
    ax.set_title("Top 10 SKUs by Qty Share (%)", fontsize=14)
    ax.set_xlabel("Share (%)", fontsize=12); ax.set_ylabel("")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.2f}%'))
    for i, v in enumerate(data.values):
        ax.text(v+0.2, i, f'{v:.2f}%', va='center', fontsize=10)
    plt.tight_layout()
    return fig

def plot_sku_pairs(df):
    from itertools import combinations
    from collections import Counter
    df['Order_ID'] = df['Customer_Phone'].astype(str) + "_" + df['Delivered_date'].astype(str)
    order_skus = df.groupby("Order_ID")["SKU_Code"].apply(set)
    counts = Counter()
    for items in order_skus:
        if len(items)>1:
            for pair in combinations(sorted(items),2):
                counts[pair]+=1
    pair_df = pd.DataFrame(counts.items(), columns=["SKU_Pair","Count"]).nlargest(10,"Count")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x=pair_df['Count'], y=pair_df['SKU_Pair'].astype(str), ax=ax, ci=None)
    ax.set_title("Top 10 SKU Pairs", fontsize=14)
    ax.set_xlabel("Orders Together", fontsize=12); ax.set_ylabel("")
    for i, v in enumerate(pair_df['Count']):
        ax.text(v+1, i, f'{v}', va='center', fontsize=10)
    plt.tight_layout()
    return fig

# --- Streamlit App UI ---
st.set_page_config(page_title="Sales Intelligence Dashboard", layout="wide")
logo = Image.open("logo.png")
st.image(logo, width=120)
st.markdown("<h1 style='text-align:center;'>📊 Sales Intelligence Dashboard</h1>", unsafe_allow_html=True)

section = st.sidebar.radio(
    "Choose a Section",
    [
        "📊 EDA Overview",
        "📉 Drop Detection",
        "👤 Customer Profiling",
        "👤 Customer Profiling (Model Prediction)",
        "🔁 Cross-Selling",
        "🔗 Brand Correlation",
        "🥇 Buyer Analysis",
        "📈 Retention & Moving Average",
        "🤖 Recommender System"
    ]
)

# --- EDA Overview  ---
if section == "📊 EDA Overview":
    st.subheader("Exploratory Data Analysis")
    tabs = st.tabs([
        "Top Revenue", "Top Quantity", "Buyer Types", "Buyers Trend",
        "SKUs Trend", "Qty vs Revenue", "Avg Order Value", "Lifetime Value",
        "SKU Share %", "SKU Pairs"
    ])

    # 1) Top 10 SKUs by Revenue
    with tabs[0]:
        st.markdown("#### Top 10 SKUs by Total Revenue")
        top_revenue = DF.groupby("SKU_Code")["Redistribution Value"] \
                        .sum().nlargest(10)
        st.bar_chart(top_revenue)

    # 2) Top 10 SKUs by Quantity
    with tabs[1]:
        st.markdown("#### Top 10 SKUs by Total Quantity")
        top_qty = DF.groupby("SKU_Code")["Delivered Qty"] \
                    .sum().nlargest(10)
        st.bar_chart(top_qty)

    # 3) Repeat vs One-Time Buyers
    with tabs[2]:
        st.markdown("#### Repeat vs One-Time Buyers")
        buyer_counts = (
            DF.groupby("Customer_Phone")["Delivered_date"]
              .nunique()
              .rename("Purchase Count")
        )
        summary = (buyer_counts == 1).map({True:"One-time",False:"Repeat"}) \
                   .value_counts()
        st.bar_chart(summary)

    # 4) Monthly Trend for Top 5 Buyers
    with tabs[3]:
        st.markdown("#### Monthly Spend Trend: Top 5 Buyers")
        df_b = DF.copy()
        df_b["MonthTS"] = df_b["Month"].dt.to_timestamp()
        top5_buyers = df_b.groupby("Customer_Phone")["Redistribution Value"] \
                          .sum().nlargest(5).index
        trend_b = (df_b[df_b["Customer_Phone"].isin(top5_buyers)]
                   .groupby(["MonthTS","Customer_Phone"])["Redistribution Value"]
                   .sum()
                   .unstack())
        st.line_chart(trend_b)

    # 5) Monthly Trend for Top 5 SKUs
    with tabs[4]:
        st.markdown("#### Monthly Quantity Trend: Top 5 SKUs")
        df_s = DF.copy()
        df_s["MonthTS"] = df_s["Month"].dt.to_timestamp()
        top5_skus = df_s.groupby("SKU_Code")["Delivered Qty"] \
                        .sum().nlargest(5).index
        trend_s = (df_s[df_s["SKU_Code"].isin(top5_skus)]
                   .groupby(["MonthTS","SKU_Code"])["Delivered Qty"]
                   .sum()
                   .unstack())
        st.line_chart(trend_s)

    # 6) Quantity vs Revenue (two separate lines)
    with tabs[5]:
        st.markdown("#### Monthly Quantity & Revenue")
        df_m = DF.copy()
        df_m["MonthTS"] = df_m["Month"].dt.to_timestamp()
        monthly = df_m.groupby("MonthTS")[["Delivered Qty","Redistribution Value"]] \
                      .sum()
        st.line_chart(monthly)

    # 7) Top 10 Customers by Average Order Value
    with tabs[6]:
        st.markdown("#### Top 10 by Avg Order Value")
        avg_order = DF.groupby("Customer_Phone")["Redistribution Value"] \
                      .mean().nlargest(10)
        st.bar_chart(avg_order)

    # 8) Top 10 Customers by Lifetime Value
    with tabs[7]:
        st.markdown("#### Top 10 by Lifetime Value")
        ltv = DF.groupby("Customer_Phone")["Redistribution Value"] \
                .sum().nlargest(10)
        st.bar_chart(ltv)

    # 9) Top 10 SKUs by Quantity Share %
    with tabs[8]:
        st.markdown("#### Top 10 SKUs by Share of Total Qty")
        share = (DF.groupby("SKU_Code")["Delivered Qty"].sum() /
                 DF["Delivered Qty"].sum() * 100).nlargest(10)
        st.bar_chart(share)

        # 10) Top 10 Most Frequently Bought SKU Pairs
    with tabs[9]:
        st.markdown("#### Top 10 SKU Pairs (Bought Together)")

        from itertools import combinations
        from collections import Counter

        df_p = DF.copy()
        df_p["Order_ID"] = (
            df_p["Customer_Phone"].astype(str)
            + "_"
            + df_p["Delivered_date"].astype(str)
        )

        # Build a counter of SKU‐pairs
        pair_sets = df_p.groupby("Order_ID")["SKU_Code"].apply(set)
        cnt = Counter()
        for s in pair_sets:
            if len(s) > 1:
                for pair in combinations(sorted(s), 2):
                    cnt[pair] += 1

        # Turn into DataFrame with proper names in one go
        top_pairs = pd.Series(cnt).nlargest(10)
        df_pairs = (
            top_pairs
            .rename_axis("SKU Pair")
            .reset_index(name="Count")
        )

        # Format the tuple into a string
        df_pairs["SKU Pair"] = df_pairs["SKU Pair"]\
            .apply(lambda t: f"{t[0]} & {t[1]}")

        # Index by that label so bar_chart uses it
        df_pairs = df_pairs.set_index("SKU Pair")

        st.bar_chart(df_pairs)
        
elif section == "📉 Drop Detection":
    st.subheader("Brand-Level MoM Drop (>30%)")
    bm = DF.groupby(['Brand','Month'])['Redistribution Value'].sum().unstack(fill_value=0)
    mom = bm.pct_change(axis=1)*100
    flags = mom < -30
    disp = mom.round(1).astype(str)
    disp[flags] += "% 🔻"; disp[~flags] = ""
    st.dataframe(disp)

elif section == "👤 Customer Profiling":
    st.subheader("Customer Purchase Deep-Dive")
    cust = st.selectbox("Select Customer Phone:", sorted(DF['Customer_Phone'].unique()))
    if cust:
        df_c = DF[DF['Customer_Phone']==cust]
        skus = df_c['SKU_Code'].unique().tolist()
        last = df_c.groupby('SKU_Code')['Delivered_date'].max().dt.date
        qty = df_c.groupby(['SKU_Code','Month'])['Delivered Qty'].sum().groupby('SKU_Code').mean().round(2)
        spend = df_c.groupby(['SKU_Code','Month'])['Total_Amount_Spent'].sum().groupby('SKU_Code').mean().round(2)
        interval = {sku: round((grp['Delivered_date'].drop_duplicates().sort_values().diff().dt.days.dropna().mean()/30.44),2)
                    if len(grp)>1 else "One" for sku, grp in df_c.groupby('SKU_Code')}
        report = pd.DataFrame({
            'SKU_Code': skus,
            'Last Purchase': [last[s] for s in skus],
            'Avg Qty/Mth': [qty.get(s,0) for s in skus],
            'Interval (Mth)': [interval.get(s) for s in skus],
            'Avg Spend/Mth': [spend.get(s,0) for s in skus],
        }).set_index('SKU_Code')
        st.dataframe(report)

elif section == "👤 Customer Profiling (Model Prediction)":
    st.subheader("Next-Purchase Model Predictions")
    cust = st.selectbox("Customer:", sorted(PRED_DF['Customer_Phone'].unique()))
    if cust:
        p = PRED_DF[PRED_DF['Customer_Phone']==cust].drop(columns=['Customer_Phone']).set_index('SKU_Code')
        p['Probability'] = p['Probability'].map(lambda x: f"{x:.1f}%")
        st.dataframe(p)

elif section == "🔁 Cross-Selling":
    st.subheader("Brand Switching Patterns (Top 3)")
    lp = DF.groupby(['Customer_Phone','Brand'])['Month'].max().reset_index()
    latest = DF['Month'].max()
    dropped = lp[lp['Month']<latest]
    merged = DF.merge(dropped, on='Customer_Phone', suffixes=('','_dropped'))
    switched = merged[(merged['Month']>merged['Month_dropped'])&(merged['Brand']!=merged['Brand_dropped'])]
    switches = switched.groupby(['Brand_dropped','Brand'])['Order_Id'].count().reset_index(name='Count')
    top3 = switches.sort_values(['Brand_dropped','Count'],ascending=[True,False]).groupby('Brand_dropped').head(3)
    st.dataframe(top3)

elif section == "🔗 Brand Correlation":
    st.subheader("Brand Correlation Matrix")
    mat = DF.groupby(['Customer_Phone','Brand'])['Order_Id'].count().unstack(fill_value=0)
    st.dataframe(mat.corr().round(2))

elif section == "🥇 Buyer Analysis":
    st.subheader("Top & Bottom Buyers (Latest Month)")
    latest_m = DF['Month'].max()
    bd = DF[DF['Month']==latest_m].groupby('Customer_Phone')['Redistribution Value'].sum().reset_index()
    st.write("Top Buyers"); st.dataframe(bd.nlargest(10,'Redistribution Value'))
    st.write("Bottom Buyers"); st.dataframe(bd.nsmallest(10,'Redistribution Value'))

elif section == "📈 Retention & Moving Average":
    st.subheader("3-Month Moving Avg of Orders")
    ords = DF.groupby('Month')['Order_Id'].nunique()
    st.line_chart(ords.rolling(3).mean())

elif section == "🤖 Recommender System":
    st.subheader("Hybrid Recommendations")
    uim = DF.pivot_table(index='Customer_Phone', columns='SKU_Code', values='Redistribution Value', aggfunc='sum').fillna(0)
    pf = DF[['SKU_Code','Brand']].drop_duplicates().set_index('SKU_Code')
    pe = pd.get_dummies(pf, columns=['Brand'])
    us = pd.DataFrame(cosine_similarity(uim), index=uim.index, columns=uim.index)
    isim = pd.DataFrame(cosine_similarity(pe), index=pe.index, columns=pe.index)
    sel = st.selectbox("Select Customer:", uim.index)
    if st.button("Show Recommendations"):
        w = uim.T.dot(us[sel]).div(us[sel].sum())
        bi = uim.loc[sel][uim.loc[sel]>0].index
        cs = isim[bi].sum(axis=1)
        scores = (0.5*w+0.5*cs).drop(bi, errors='ignore')
        st.dataframe(scores.nlargest(5).reset_index().rename(columns={0:'Score','index':'SKU'}))
