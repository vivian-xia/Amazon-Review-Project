import numpy as np
import faiss
import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OpenAI_API_Key")

class ReviewRetriever:
    def __init__(self, index_path="faiss_index.idx", data_path="reviews_data.pkl"):
        """Initialize FAISS index and review dataset."""
        self.df = pd.read_pickle(data_path)
        self.index = faiss.read_index(index_path)
        self.client = OpenAI(api_key=api_key)

    def get_product_list(self):
        """Return a sorted list of unique shampoo products from the dataset."""
        return sorted(self.df["product_title"].unique())

    def get_top_k_reviews(self, query_text, selected_product=None, top_k=10):
        """Retrieve top-k most relevant reviews, either for a specific product or across all products."""
        if selected_product:
            # Filter dataset to only include reviews for the selected product
            filtered_df = self.df[self.df["product_title"] == selected_product].copy()
        else:
            # Use all reviews if no specific product is selected
            filtered_df = self.df.copy()
        
        if filtered_df.empty:
            return pd.DataFrame()

        # Convert Query Text to Embedding
        response = self.client.embeddings.create(input=[query_text], model="text-embedding-ada-002")
        query_embedding = response.data[0].embedding  

        # Normalize Embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Search FAISS Index for the most similar reviews
        review_embeddings = self.index.reconstruct_n(0, len(filtered_df))  # Get embeddings for filtered reviews
        distances = np.dot(review_embeddings, query_embedding)  # Compute similarity
        top_indices = np.argsort(distances)[-top_k:][::-1]  # Get top-k indices sorted by highest similarity

        # Retrieve matching reviews
        top_reviews = filtered_df.iloc[top_indices].copy()
        top_reviews["similarity_score"] = distances[top_indices]

        return top_reviews
