import streamlit as st
import pandas as pd

st.set_page_config(page_title="Service Complaints Analyzer", layout="wide")

st.title("Service Complaints Analyzer")

uploaded_file = st.file_uploader("Upload your complaints CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.warning("Uploaded CSV file is empty.")
        else:
            # Clean column names
            df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]

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
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                st.write(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
            else:
                st.warning("Column 'date' not found in the dataset.")

    except Exception as e:
        st.error(f"Error reading the CSV file: {e}")
