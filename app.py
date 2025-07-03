import streamlit as st
import pandas as pd
import re
import os
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from prophet import Prophet
from fpdf import FPDF

st.set_page_config(page_title="Service Complaints Analyzer", layout="wide")
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
    # Check if a 'data' directory exists, and if not, create it.
    # This is useful for the fallback mechanism.
    if not os.path.exists("data"):
        os.makedirs("data")
    fallback_path = "data/complaints.csv"
    if os.path.exists(fallback_path):
        st.info("No file uploaded. Using fallback sample data from 'data/complaints.csv'.")
        df = pd.read_csv(fallback_path)
    else:
        st.warning("No file uploaded and fallback file not found.")
        st.info("Please upload a CSV file with 'date', 'complainttype', and 'description' columns.")
        df = None

# Continue only if data is valid
if df is not None and not df.empty:
    # Clean column names
    df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]

    # Convert 'date' to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # --- Sidebar filters ---
    st.sidebar.header("Filters")

    # Date filter
    if 'date' in df.columns and df['date'].notna().any():
        min_date_available = df['date'].min().date()
        max_date_available = df['date'].max().date()
        start_date, end_date = st.sidebar.date_input(
            "Filter by Date Range",
            [min_date_available, max_date_available]
        )
        df = df[(df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)]
    else:
        st.sidebar.warning("Date column not found or contains no valid dates for filtering.")


    # Complaint Type filter
    if 'complainttype' in df.columns:
        # Ensure unique and non-null complaint types
        complaint_options = df['complainttype'].dropna().unique().tolist()
        if complaint_options: # Only show multiselect if there are options
            selected_types = st.sidebar.multiselect(
                "Filter by Complaint Type",
                complaint_options,
                default=complaint_options
            )
            df = df[df['complainttype'].isin(selected_types)]
        else:
            st.sidebar.info("No valid complaint types found.")
    else:
        st.sidebar.warning("Complaint Type column ('complainttype') not found.")


    # Keyword Search
    keyword = st.sidebar.text_input("Search Description Keyword").strip().lower()
    if keyword and 'description' in df.columns:
        df = df[df['description'].astype(str).str.lower().str.contains(keyword, na=False)] # Use .astype(str) to handle non-string types


    # --- Sentiment Analysis ---
    if 'description' in df.columns:
        def get_sentiment(text):
            # Ensure text is a string before passing to TextBlob
            if pd.isna(text):
                return "Neutral" # Or "N/A", depending on how you want to handle missing descriptions
            blob = TextBlob(str(text))
            polarity = blob.sentiment.polarity
            if polarity > 0.1:
                return "Positive"
            elif polarity < -0.1:
                return "Negative"
            else:
                return "Neutral"

        df['sentiment'] = df['description'].apply(get_sentiment)

        # Sentiment Filter
        sentiment_options = ['Positive', 'Neutral', 'Negative']
        selected_sentiments = st.sidebar.multiselect("Filter by Sentiment", sentiment_options, default=sentiment_options)
        df = df[df['sentiment'].isin(selected_sentiments)]
    else:
        st.warning("Cannot perform sentiment analysis. 'description' column missing.")


    # --- Display Section ---
    st.subheader("Raw Data")
    st.dataframe(df)

    st.subheader("Complaint Summary")
    st.write(f"Total complaints: {len(df)}")

    if 'complainttype' in df.columns:
        st.write("Top 5 Complaint Types:")
        st.dataframe(df['complainttype'].value_counts().head())

    if 'date' in df.columns and df['date'].notna().any():
        st.write(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    else:
        st.info("Date range not available (date column missing or empty).")


    # --- Categorize complaints ---
    if 'description' in df.columns:
        def categorize(text):
            # Ensure text is a string
            if pd.isna(text):
                return "Unknown"
            text = str(text).lower()
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
        st.warning("Cannot categorize complaints. 'description' column missing.")


    # --- Sentiment Chart ---
    if 'sentiment' in df.columns:
        st.subheader("Sentiment Distribution")
        st.bar_chart(df['sentiment'].value_counts())
    else:
        st.info("Sentiment distribution not available (sentiment column missing).")

    #add support for visualizing complaint distribution using heatmap if location data is present
    if 'latitude' in df.columns and 'longitude' in df.columns:
        st.subheader("Complaint Location Heatmap")
        location_df = df[['latitude', 'longitude']].dropna()
        st.map(location_df)
    else:
        st.info("No geolocation data (latitude, longitude) found for mapping.")

    if 'description' in df.columns:
        st.subheader("Word Cloud of Complaint Descriptions")

        text = " ".join(df['description'].dropna().astype(str).tolist())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.info("Generating word cloud from descriptions failed")


if 'date' in df.columns:
    st.subheader("Complaint Volume Forecast (Next 30 Days)")

    forecast_df = df['date'].value_counts().reset_index()
    forecast_df.columns = ['ds', 'y']
    forecast_df = forecast_df.sort_values('ds')

    model = Prophet()
    model.fit(forecast_df)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    fig1 = model.plot(forecast)
    st.pyplot(fig1)   

    if st.button("Generate PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Complaint Summary Report", ln=True, align='C')

        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Total Complaints: {len(df)}", ln=True)

    if 'complainttype' in df.columns:
        top_types = df['complainttype'].value_counts().head().to_dict()
        for t, v in top_types.items():
            pdf.cell(200, 10, txt=f"{t}: {v}", ln=True)

    report_path = "summary_report.pdf"
    pdf.output(report_path)
    with open(report_path, "rb") as file:
        st.download_button("Download PDF Report", file, file_name="summary_report.pdf")


    #improve UI
    theme_choice = st.sidebar.radio("Theme", ["Dark", "Light"])
    if theme_choice == "Light":
        st.info("Light theme selected. To activate, change config.toml to base='light'")
    else:
        st.info("Dark theme selected. Already applied if your config.toml has base='dark'")

    # --- Trend Analysis ---
    st.subheader("Complaint Trend Over Time")
    if 'date' in df.columns and df['date'].notna().any():
        trend_df = df[df['date'].notna()]
        trend_daily = trend_df.groupby('date').size().reset_index(name='count')
        st.line_chart(trend_daily.set_index('date'))

        # Ensure 'month' column is created only if 'date' is valid
        trend_df['month'] = trend_df['date'].dt.to_period('M').astype(str)
        trend_monthly = trend_df.groupby('month').size().reset_index(name='count')
        st.subheader("Monthly Complaint Trend")
        st.bar_chart(trend_monthly.set_index('month'))
    else:
        st.warning("Cannot display trend. 'date' column missing or not parsable.")


    # --- Complaint Topic Modeling (LDA) ---
    if 'description' in df.columns and not df['description'].empty:
        st.subheader("Complaint Topic Modeling (LDA)")

        # Preprocessing: Fill NaN with empty string, convert to lower, remove non-alphanumeric
        cleaned_text = df['description'].fillna("").astype(str).str.lower().str.replace(r'[^\w\s]', '', regex=True)

        # Filter out empty strings after cleaning, otherwise CountVectorizer might throw errors
        cleaned_text = cleaned_text[cleaned_text.str.strip() != '']

        if not cleaned_text.empty:
            vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
            try:
                doc_term_matrix = vectorizer.fit_transform(cleaned_text)

                # Adjust n_components if the number of documents is less than 5
                n_components = min(5, doc_term_matrix.shape[0])
                if n_components > 0:
                    lda = LatentDirichletAllocation(n_components=n_components, random_state=42)
                    lda.fit(doc_term_matrix)

                    words = vectorizer.get_feature_names_out()
                    for i, topic in enumerate(lda.components_):
                        st.write(f"**Topic {i+1}**:")
                        # Ensure there are enough words to pick from
                        num_keywords = min(10, len(words))
                        topic_keywords = [words[j] for j in topic.argsort()[-num_keywords:]]
                        st.write(", ".join(topic_keywords))
                else:
                    st.info("Not enough data after preprocessing to perform LDA topic modeling with 5 components.")
            except ValueError as ve:
                st.warning(f"Could not perform LDA topic modeling. Error: {ve}. This often happens if there's not enough unique text data after filtering.")
        else:
            st.info("No valid text data in 'description' column after preprocessing for LDA.")
    else:
        st.warning("Cannot perform Topic Modeling. 'description' column missing or empty.")

else:
    st.warning("No valid data available. Please upload a CSV file to begin analysis.")
