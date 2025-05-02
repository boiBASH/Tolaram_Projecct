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
    df = pd.read_csv("Data Analysis - Sample File.csv")
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
    "🤖 Recommender System",
    "👤 Customer Profiling (In-Depth)"  # New section added here
])

# Shared monthly summary
df_monthly = DF.groupby('Month')['Redistribution Value'].sum()

# New section: Customer Profiling (In-Depth)
if section == "👤 Customer Profiling (In-Depth)":
    st.subheader("Customer Profiling (In-Depth)")

    # Customer phone input
    customer_phone = st.text_input("Enter Customer Phone Number:", "")
    
    if customer_phone:
        # Load customer data and generate the report
        customer_report = analyze_customer_purchases(customer_phone)

        if isinstance(customer_report, str):  # Check if it's an error message
            st.error(customer_report)
        else:
            # Displaying the report in a beautified way
            st.markdown(f"**Customer Phone:** {customer_report['Customer Phone']}")
            st.markdown(f"**Total Unique SKUs Bought:** {customer_report['Total Unique SKUs Bought']}")
            st.markdown(f"**SKUs Bought:** {', '.join(customer_report['SKUs Bought'])}")

            st.subheader("Purchase Summary by SKU")
            for sku, summary in customer_report['Purchase Summary by SKU'].items():
                st.markdown(f"### SKU: {sku}")
                st.write(f"- **Last Purchase Date**: {summary['Last Purchase Date']}")
                st.write(f"- **Avg Monthly Quantity**: {summary['Avg Monthly Quantity']}")
                st.write(f"- **Avg Purchase Interval (Months)**: {summary['Avg Purchase Interval (Months)']}")
                st.write(f"- **Avg Monthly Spend**: {summary['Avg Monthly Spend']}")

    else:
        st.info("Please enter a customer phone number to begin the analysis.")

# Function to analyze customer purchases
def analyze_customer_purchases(customer_phone):
    customer_df = df[df['Customer_Phone'] == customer_phone].copy()

    if customer_df.empty:
        return f"No data found for customer phone: {customer_phone}"

    # Ensure date is sorted
    customer_df.sort_values('Delivered_date', inplace=True)

    # Add Month column
    customer_df['Month'] = customer_df['Delivered_date'].dt.to_period('M')

    # 1. List of SKUs bought
    skus_bought = customer_df['SKU_Code'].unique().tolist()

    # 2. Last purchase date for each SKU
    last_purchase = customer_df.groupby('SKU_Code')['Delivered_date'].max().dt.strftime('%Y-%m-%d').to_dict()

    # 3. Average quantity per month for each SKU
    monthly_qty = (
        customer_df.groupby(['SKU_Code', 'Month'])['Delivered Qty']
        .sum()
        .groupby('SKU_Code')
        .mean()
        .round(2)
        .to_dict()
    )

    # 4. Average interval (in months) between purchases
    avg_interval = {}
    for sku, group in customer_df.groupby('SKU_Code'):
        dates = group['Delivered_date'].drop_duplicates().sort_values()
        if len(dates) > 1:
            intervals = dates.diff().dropna().dt.days / 30.44
            avg_interval[sku] = round(intervals.mean(), 2)
        else:
            avg_interval[sku] = "One"

    # 5. Average monthly spend per SKU
    monthly_spend = (
        customer_df.groupby(['SKU_Code', 'Month'])['Total_Amount_Spent']
        .sum()
        .groupby('SKU_Code')
        .mean()
        .round(2)
        .to_dict()
    )

    # Final formatted output
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
