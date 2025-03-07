import openai
import os
import json
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OpenAI_API_Key")

class SentimentAgent:
    def __init__(self):
        """Initialize OpenAI client."""
        self.client = openai.OpenAI(api_key=api_key)

    def analyze_reviews(self, reviews):
        """Analyze sentiment of each review and return updated DataFrame with sentiment labels."""
        review_texts = reviews["combined_context"].tolist()

        prompt = f"""
        Analyze the sentiment of each of the following product reviews individually.
        Categorize each review as 'positive', 'neutral', or 'negative'. 

        Reviews:
        {review_texts}

        Return the output in a valid JSON format, with one label per review in the same order:
        {{
          "sentiments": ["positive", "negative", "neutral", ...]
        }}
        DO NOT include any explanations, markdown, or extra text—only the JSON object.
        """

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are an expert sentiment analyzer."},
                      {"role": "user", "content": prompt}],
            temperature=0.3
        )

        raw_output = response.choices[0].message.content.strip()

        # Ensure we extract only the JSON part
        try:
            cleaned_json = raw_output.replace("```json", "").replace("```", "").strip()
            sentiment_data = json.loads(cleaned_json)  # Parse JSON safely
            sentiments = sentiment_data.get("sentiments", [])
        except json.JSONDecodeError:
            sentiments = ["error"] * len(review_texts)

        # Assign sentiment labels to DataFrame
        reviews = reviews.copy()
        reviews["sentiment"] = sentiments

        return reviews
