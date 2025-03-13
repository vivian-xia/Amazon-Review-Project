import os
import streamlit as st
import pandas as pd
from openai import OpenAI
from retriever import ReviewRetriever
from sentiment import SentimentAgent
from summary import SummaryAgent
from evaluation import evaluate_answer_cosine
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Load API key from secrets
api_key = st.secrets.get("OpenAI_API_Key")
if not api_key:
    st.error("❌ OpenAI API key not found in secrets.")
    st.stop()

# Initialize agents
retriever = ReviewRetriever(api_key=api_key)
sentiment_agent = SentimentAgent(api_key=api_key)
summary_agent = SummaryAgent(api_key=api_key)

st.title("Shampoo Review-Based Q&A AI 🔍")
st.write("Ask about a specific shampoo or find the best shampoo for a concern like volume, dandruff, or dry hair.")

query_type = st.radio("What would you like to do?", ["Ask about a specific shampoo", "Find the best shampoo for a concern"])

if query_type == "Ask about a specific shampoo":
    shampoo_list = retriever.get_product_list()
    selected_shampoo = st.selectbox("Select a shampoo product:", shampoo_list)
    user_query = st.text_input(f"Ask a question about '{selected_shampoo}':")

    if user_query:
        with st.spinner("Retrieving reviews..."):
            top_reviews = retriever.get_top_k_reviews(user_query, selected_product=selected_shampoo)

        if top_reviews.empty:
            st.warning(f"No relevant reviews found for {selected_shampoo}.")
        else:
            with st.spinner("Analyzing review sentiment..."):
                top_reviews_with_sentiment = sentiment_agent.analyze_reviews(top_reviews)

            with st.spinner("Generating AI-powered response..."):
                generated_answer = summary_agent.generate_summary(user_query, top_reviews_with_sentiment)

            st.subheader("AI-Generated Answer")
            st.write(generated_answer)

            st.subheader(f"Top Relevant Reviews for {selected_shampoo} with Sentiment")
            for index, row in top_reviews_with_sentiment.iterrows():
                with st.expander(f"Review {index + 1} - {row['product_title']} (Score: {row['similarity_score']:.2f})"):
                    st.write(f"**Sentiment:** {row['sentiment']}")
                    st.write(f"**Review:** {row['combined_context']}")

            evaluate_answer_cosine(
                api_key=api_key,
                user_query=user_query,
                retrieved_reviews=top_reviews_with_sentiment,
                generated_answer=generated_answer,
                export_csv_path="evaluation_logs.csv"
            )

else:
    user_query = st.text_input("Example: What shampoo is best for (e.g., volume, dandruff, dry hair)?")

    if user_query:
        with st.spinner("Retrieving reviews..."):
            top_reviews = retriever.get_top_k_reviews(user_query)

        if top_reviews.empty:
            st.warning("No relevant reviews found.")
        else:
            with st.spinner("Analyzing review sentiment..."):
                top_reviews_with_sentiment = sentiment_agent.analyze_reviews(top_reviews)

            with st.spinner("Generating AI-powered response..."):
                generated_answer = summary_agent.generate_summary(user_query, top_reviews_with_sentiment)

            st.subheader("AI-Generated Answer")
            st.write(generated_answer)

            st.subheader("Top Relevant Reviews with Sentiment")
            for index, row in top_reviews_with_sentiment.iterrows():
                with st.expander(f"Review {index + 1} - {row['product_title']} (Score: {row['similarity_score']:.2f})"):
                    st.write(f"**Sentiment:** {row['sentiment']}")
                    st.write(f"**Review:** {row['combined_context']}")

            evaluate_answer_cosine(
                api_key=api_key,
                user_query=user_query,
                retrieved_reviews=top_reviews_with_sentiment,
                generated_answer=generated_answer,
                export_csv_path="evaluation_logs.csv"
            )

# Upload evaluation logs to Google Sheet
def append_to_google_sheet(sheet_id, sheet_range, data):
    creds = service_account.Credentials.from_service_account_info(st.secrets["google_sheets"])
    service = build("sheets", "v4", credentials=creds)
    sheet = service.spreadsheets()

    body = {
        "values": data
    }

    result = sheet.values().append(
        spreadsheetId=sheet_id,
        range=sheet_range,
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body=body
    ).execute()

    return result

if os.path.exists("evaluation_logs.csv"):
    try:
        df = pd.read_csv("evaluation_logs.csv")
        values = [df.columns.tolist()] + df.values.tolist()

        SHEET_ID = "1YIK6FL1mrSKwnrKK4V1SdipdBJHK-DUh_UvilK9HONo"
        SHEET_RANGE = "Sheet1!A1"  # Adjust as needed

        append_to_google_sheet(SHEET_ID, SHEET_RANGE, values)
        st.success("✅ Evaluation results uploaded to Google Sheet!")
        st.markdown(f"[🔗 Open Sheet](https://docs.google.com/spreadsheets/d/{SHEET_ID})")
    except Exception as e:
        st.error(f"❌ Failed to write to Google Sheet: {e}")
