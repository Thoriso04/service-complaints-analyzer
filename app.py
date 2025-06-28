import streamlit as st
import pandas as pd
import os
import plotly.express as px
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER lexicon (only once)
nltk.download('vader_lexicon')

# Configure page
st.set_page_config(page_title="Service Complaints Analyzer", layout="wide")
st.title("Service Complaints Analyzer")

uploaded_file = st.file_uploader("Upload your complaints CSV file", type=["csv"])

# Load data
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

# Proceed if data is valid
if df is not None and not df.empty:
    df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Sidebar filters
    st.sidebar.header("Filters")

    # Date filter
    if 'date' in df.columns and df['date'].notna().any():
        min_date = df['date'].min().date()
        max_date = df['date'].max().date()
        start_date, end_date = st.sidebar.date_input("Filter by Date Range", [min_date, max_date])
        df = df[(df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)]

    # Complaint Type filter
    if 'complainttype' in df.columns:
        complaint_options = df['complainttype'].dropna().unique().tolist()
        selected_types = st.sidebar.multiselect("Filter by Complaint Type", complaint_options, default=complaint_options)
        df = df[df['complainttype'].isin(selected_types)]

    # Keyword search
    keyword = st.sidebar.text_input("Search Description Keyword").strip().lower()
    if keyword and 'description' in df.columns:
        df = df[df['description'].str.lower().str.contains(keyword, na=False)]

    # Complaint categorization
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

    # Sentiment analysis
    if 'description' in df.columns:
        SIA = SentimentIntensityAnalyzer()
        df['sentiment_score'] = df['description'].apply(lambda x: SIA.polarity_scores(x)['compound'])
        df['sentiment'] = df['sentiment_score'].apply(
            lambda x: 'Positive' if x > 0.2 else 'Negative' if x < -0.2 else 'Neutral'
        )

    # Display data
    st.subheader("Raw Data")
    st.dataframe(df)

    st.subheader("Cleaned Data Preview")
    st.dataframe(df.head())

    st.subheader("Complaint Summary")
    st.write(f"Total complaints: {len(df)}")

    if 'complainttype' in df.columns:
        st.subheader("Top Complaint Types (Interactive)")
        complaint_counts = df['complainttype'].value_counts().reset_index()
        complaint_counts.columns = ['Complaint Type', 'Count']
        fig_type = px.bar(complaint_counts, x='Complaint Type', y='Count', title='Complaint Types')
        st.plotly_chart(fig_type)

    if 'category' in df.columns:
        st.subheader("Top Complaint Categories (Interactive)")
        category_counts = df['category'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Count']
        fig_cat = px.pie(category_counts, names='Category', values='Count', title='Complaint Categories')
        st.plotly_chart(fig_cat)

    if 'sentiment' in df.columns:
        st.subheader("Sentiment Distribution")
        fig_sentiment = px.histogram(df, x='sentiment', title='Sentiment Analysis of Complaints')
        st.plotly_chart(fig_sentiment)

    if 'date' in df.columns and df['date'].notna().any():
        st.subheader("Complaint Trend Over Time (Daily)")
        trend_df = df[df['date'].notna()]
        trend_daily = trend_df.groupby('date').size().reset_index(name='count')
        fig_trend = px.line(trend_daily, x='date', y='count', title='Daily Complaint Trend')
        st.plotly_chart(fig_trend)

        trend_df['month'] = trend_df['date'].dt.to_period('M').astype(str)
        trend_monthly = trend_df.groupby('month').size().reset_index(name='count')
        st.subheader("Monthly Complaint Trend")
        fig_month = px.bar(trend_monthly, x='month', y='count', title='Monthly Complaint Volume')
        st.plotly_chart(fig_month)
    else:
        st.warning("Cannot display trend. 'date' column missing or not parsable.")

else:
    st.warning("No valid data available.")
