import streamlit as st
import pandas as pd
import numpy as np
import io
from retriever import ReviewRetriever
from sentiment import SentimentAgent
from summary import SummaryAgent
from evaluation import evaluate_answer_cosine
import os
from dotenv import load_dotenv
import uuid
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Setup
st.set_page_config(page_title="LLM Param Comparison", layout="wide")
st.title("🔍 Compare LLM Parameters to Baseline")

# Load API key from secrets
api_key = st.secrets.get("OpenAI_API_Key")
if not api_key:
    st.error("❌ OpenAI API key not found in secrets.")
    st.stop()

# Initialize agents
retriever = ReviewRetriever(api_key=api_key)
sentiment_agent = SentimentAgent(api_key=api_key)
summary_agent = SummaryAgent(api_key=api_key)

# Sidebar inputs
st.sidebar.header("Experiment Setup")
query = st.sidebar.text_input("User Query", "i.e. What shampoo is best for dandruff?")
product_list = retriever.get_product_list()
selected_product = st.sidebar.selectbox("Select Product (optional)", [None] + product_list)

# Choose parameter to modify
param_to_change = st.sidebar.selectbox("Parameter to Change", [
    "temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty"
])
new_value = st.sidebar.text_input("New Value for Selected Parameter", "0.7")

run_button = st.sidebar.button("🚀 Compare to Baseline")
run_id = str(uuid.uuid4())[:8]  # Short unique ID


# Define baseline
baseline_params = {
    "temperature": 0.3, 
    "max_tokens": 200, 
    "top_p": 1.0, 
    "frequency_penalty": 0, 
    "presence_penalty": 0
}

# Run evaluation and comparison
if run_button:
    st.subheader("Comparison Results")
    with st.spinner("Running evaluation for baseline and modified settings..."):
        top_reviews = retriever.get_top_k_reviews(query, selected_product)

        if top_reviews.empty:
            st.warning("No relevant reviews found.")
        else:
            top_reviews_with_sentiment = sentiment_agent.analyze_reviews(top_reviews)

            
            baseline_answer = summary_agent.generate_summary(query, top_reviews_with_sentiment, **baseline_params)
            baseline_result = evaluate_answer_cosine(api_key,
                user_query=query,
                retrieved_reviews=top_reviews_with_sentiment,
                generated_answer=baseline_answer,
                export_csv_path="evaluation_logs.csv"
            )
            baseline_result["Setting"] = "Baseline"
            baseline_result["RunID"] = run_id


            # Generate answer with modified param
            test_params = baseline_params.copy()
            test_params[param_to_change] = type(baseline_params[param_to_change])(float(new_value))

            mod_answer = summary_agent.generate_summary(query, top_reviews_with_sentiment, **test_params)
            mod_result = evaluate_answer_cosine(api_key,
                user_query=query,
                retrieved_reviews=top_reviews_with_sentiment,
                generated_answer=mod_answer,
                export_csv_path="evaluation_logs.csv"
            )
            mod_result["Setting"] = f"Modified {param_to_change} = {new_value}"
            mod_result["RunID"] = run_id


            # Show evaluation results
            results_df = pd.DataFrame([baseline_result, mod_result])
            cols = ['RunID', 'Setting'] + [col for col in results_df.columns if col not in ['RunID', 'Setting']]
            results_df = results_df[cols]

            
            # Display evaluation metric scores
            st.markdown("### 📊 Evaluation Metrics")
            st.dataframe(results_df.set_index("Setting"))

            # Show answer comparison
            st.markdown("### 📄 Answer Comparison")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Baseline Answer**")
                st.write(baseline_answer)

            with col2:
                st.markdown(f"**Modified Answer** (Changed {param_to_change} to {new_value})")
                st.write(mod_answer)

            # Download results
            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)
            st.download_button("📥 Download CSV Results", data=csv_buffer.getvalue(), file_name="param_comparison.csv", mime="text/csv")


import json
from google.oauth2 import service_account
from googleapiclient.discovery import build


def overwrite_google_sheet(sheet_id, sheet_range, data):
    SERVICE_ACCOUNT_FILE = "eighth-density-347504-9dd7cfcaf056.json"  # Path to your JSON key
    with open(SERVICE_ACCOUNT_FILE, "r") as f:
        service_account_info = json.load(f)

    creds = service_account.Credentials.from_service_account_info(service_account_info)
    service = build("sheets", "v4", credentials=creds)
    sheet = service.spreadsheets()

    body = {"values": data}

    result = sheet.values().update(
        spreadsheetId=sheet_id,
        range=sheet_range,
        valueInputOption="RAW",
        body=body
    ).execute()

    return result


if os.path.exists("evaluation_logs.csv"):
    try:
        df = pd.read_csv("evaluation_logs.csv")
        values = [df.columns.tolist()] + df.values.tolist()

        SHEET_ID = "1zDGjrexE12zxQpr310mAbc3vQhcgcGS1kq4BphXgMKk"
        SHEET_RANGE = "Sheet1!A1"  # Overwrite from top

        overwrite_google_sheet(SHEET_ID, SHEET_RANGE, values)
        st.success("✅ Evaluation results uploaded (overwritten) to Google Sheet!")
        st.markdown(f"[🔗 Open Sheet](https://docs.google.com/spreadsheets/d/{SHEET_ID})")
    except Exception as e:
        st.error(f"❌ Failed to write to Google Sheet: {e}")
