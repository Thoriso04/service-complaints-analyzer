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
            st.subheader("Raw Data")
            st.dataframe(df)
            st.subheader("Summary Statistics")
            st.write(df.describe())
    except Exception as e:
        st.error(f"Error reading the CSV file: {e}")
