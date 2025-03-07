import streamlit as st
from retriever import ReviewRetriever
from sentiment import SentimentAgent
from summary import SummaryAgent

# Initialize agents
retriever = ReviewRetriever()
sentiment_agent = SentimentAgent()
summary_agent = SummaryAgent()

st.title("Shampoo Review-Based Q&A AI 🔍")
st.write("Ask about a specific shampoo or find the best shampoo for a concern like volume, dandruff, or dry hair.")

# Let users choose between specific or general questions
query_type = st.radio("What would you like to do?", 
                      ["Ask about a specific shampoo", "Find the best shampoo for a concern"])

if query_type == "Ask about a specific shampoo":
    # Get list of shampoos and create dropdown menu
    shampoo_list = retriever.get_product_list()
    selected_shampoo = st.selectbox("Select a shampoo product:", shampoo_list)

    # User input
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

            # Display results
            st.subheader("AI-Generated Answer")
            st.write(generated_answer)

            st.subheader(f"Top Relevant Reviews for {selected_shampoo} with Sentiment")
            for index, row in top_reviews_with_sentiment.iterrows():
                with st.expander(f"Review {index + 1} - {row['product_title']} (Score: {row['similarity_score']:.2f})"):
                    st.write(f"**Sentiment:** {row['sentiment']}")
                    st.write(f"**Review:** {row['combined_context']}")

else:
    # General concern-based query
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

            # Display results
            st.subheader("AI-Generated Answer")
            st.write(generated_answer)

            st.subheader("Top Relevant Reviews with Sentiment")
            for index, row in top_reviews_with_sentiment.iterrows():
                with st.expander(f"Review {index + 1} - {row['product_title']} (Score: {row['similarity_score']:.2f})"):
                    st.write(f"**Sentiment:** {row['sentiment']}")
                    st.write(f"**Review:** {row['combined_context']}")