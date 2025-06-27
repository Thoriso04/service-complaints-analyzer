import streamlit as st
import pandas as pd
import re
import os


def new_func():
    st.set_page_config(page_title="Service Complaints Analyzer", layout="wide")

new_func()
st.title("Service Complaints Analyzer")

uploaded_file = st.file_uploader("Upload your complaints CSV file", type=["csv"])

# Use uploaded CSV or fallback to local CSV
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading the uploaded file: {e}")
        df = None
else:
    fallback_path = "data/complaints.csv"
    if os.path.exists(fallback_path):
        st.info("No file uploaded. Using fallback sample data from 'data/complaints.csv'.")
        df = pd.read_csv(fallback_path)
    else:
        st.warning("No file uploaded and fallback file not found.")
        df = None

# Continue only if data is valid
if df is not None and not df.empty:
    # Clean column names
    df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]

    # Convert 'date' to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Sidebar filters
    st.sidebar.header("Filters")

    # Date filter
    if 'date' in df.columns:
        min_date = df['date'].min().date()
        max_date = df['date'].max().date()
        start_date, end_date = st.sidebar.date_input("Filter by Date Range", [min_date, max_date])
        df = df[(df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)]

    # Complaint Type filter
    if 'complainttype' in df.columns:
        complaint_options = df['complainttype'].unique().tolist()
        selected_types = st.sidebar.multiselect("Filter by Complaint Type", complaint_options, default=complaint_options)
        df = df[df['complainttype'].isin(selected_types)]

    # Keyword Search
    keyword = st.sidebar.text_input("Search Description Keyword").strip().lower()
    if keyword:
        df = df[df['description'].str.lower().str.contains(keyword)]

    st.subheader("Raw Data")
    st.dataframe(df)

    st.subheader("Cleaned Data Preview")
    st.dataframe(df.head())

    st.subheader("Complaint Summary")
    st.write(f"Total complaints: {len(df)}")

    if 'complainttype' in df.columns:
        st.write("Top 5 Complaint Types:")
        st.dataframe(df['complainttype'].value_counts().head())
    else:
        st.warning("Column 'complainttype' not found in the dataset.")

    if 'date' in df.columns:
        st.write(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    else:
        st.warning("Column 'date' not found in the dataset.")

    # Complaint categorization based on description
    if 'description' in df.columns:
        def categorize(text):
            text = text.lower()
            if "bill" in text or "refund" in text:
                return "Billing"
            elif "network" in text or "signal" in text:
                return "Network"
            elif "slow" in text or "speed" in text:
                return "Performance"
            elif "service" in text or "support" in text or "rude" in text:
                return "Customer Service"
            else:
                return "Other"

        df['category'] = df['description'].apply(categorize)

        st.subheader("Complaint Categorization")
        st.dataframe(df[['description', 'category']])

        st.write("Top Complaint Categories")
        st.bar_chart(df['category'].value_counts())
    else:
        st.warning("Column 'description' not found in the dataset.")

    # Trend Analysis
    st.subheader("Complaint Trend Over Time")
    if 'date' in df.columns and df['date'].notna().any():
        trend_df = df[df['date'].notna()]
        trend_daily = trend_df.groupby('date').size().reset_index(name='count')
        st.line_chart(trend_daily.set_index('date'))

        # Optional: Monthly trend
        trend_df['month'] = trend_df['date'].dt.to_period('M').astype(str)
        trend_monthly = trend_df.groupby('month').size().reset_index(name='count')
        st.subheader("Monthly Complaint Trend")
        st.bar_chart(trend_monthly.set_index('month'))
    else:
        st.warning("Cannot display trend. 'date' column missing or not parsable.")

else:
    st.warning("No valid data available.")
