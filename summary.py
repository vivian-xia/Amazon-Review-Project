from openai import OpenAI
#import os
#from dotenv import load_dotenv
#import streamlit as st

#load_dotenv()
#api_key = os.getenv("OpenAI_API_Key")

class SummaryAgent:
    def __init__(self, api_key):
        """Initialize OpenAI client."""
        self.client = OpenAI(api_key=api_key)

    def generate_summary(self, user_query, reviews_with_sentiment):
        """Generate a final AI-powered answer based on reviews and their sentiment."""
        review_texts = "\n".join(
            [f"- {row['combined_context']} (Sentiment: {row['sentiment']})" for _, row in reviews_with_sentiment.iterrows()]
        )

        prompt = f"""
        Based on the following shampoo product reviews and their sentiment analysis, answer the user's question: "{user_query}".
        
        Reviews:
        {review_texts}

        Summarize key insights, mentioning trends in positive, neutral, and negative sentiments. 
        The answer should be factual and concise, without making assumptions beyond the reviews.
        Don't make up any information or provide personal opinions.
        
        If the question is unanswerable based on the reviews, communicate that clearly and the response should be in one sentence. 
        If the question is answerable based on the reviews, provide the response in a well-structured paragraph format in ideally 500 tokens.
        """

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are an expert product review analyst."},
                      {"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=500, top_p = 1.0, frequency_penalty=0, presence_penalty=0
        )

        return response.choices[0].message.content.strip()
