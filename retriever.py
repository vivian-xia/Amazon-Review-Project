import gdown
import os
import numpy as np
import faiss
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OpenAI_API_Key")

class ReviewRetriever:
    def __init__(self):
        """Initialize FAISS index and review dataset from Google Drive."""
        self.index_path = "faiss_index.idx"
        self.data_path = "reviews_data.pkl"

        # Google Drive file IDs
        self.drive_files = {
            "faiss_index": "1LckdBqXv0VuUDJ8pFV9kwKEiDROeiWL4",  # Replace with actual file ID for FAISS index
            "reviews_data": "1ppeVQ02us2T5TTFOEhx3XfpqHDM2IOXw"  # Replace with actual file ID for DataFrame
        }

        # Download files if not already present
        self._download_files()

        # Load data
        self.df = pd.read_pickle(self.data_path)
        self.index = faiss.read_index(self.index_path)
        self.client = OpenAI(api_key=api_key)

class ReviewRetriever:
    def __init__(self):
        """Initialize FAISS index and review dataset from Google Drive."""
        self.index_path = "faiss_index.idx"
        self.data_path = "reviews_data.pkl"

        # Google Drive file IDs
        self.drive_files = {
            "faiss_index": "1LckdBqXv0VuUDJ8pFV9kwKEiDROeiWL4",  # Replace with actual file ID for FAISS index
            "reviews_data": "1ppeVQ02us2T5TTFOEhx3XfpqHDM2IOXw"  # Replace with actual file ID for DataFrame
        }

        # Download files if not already present
        self._download_files()

        # Load data
        self.df = pd.read_pickle(self.data_path)
        self.index = faiss.read_index(self.index_path)
        self.client = OpenAI(api_key=api_key)

    def _download_files(self):
        """Download FAISS index and review data from Google Drive if not available."""
        for file_name, file_id in self.drive_files.items():
            # Map "faiss_index" to self.index_path and "reviews_data" to self.data_path
            if file_name == "faiss_index" and not os.path.exists(self.index_path):
                print(f"Downloading {file_name} from Google Drive...")
                gdown.download(f"https://drive.google.com/uc?id={file_id}", self.index_path)
                print("Successfully downloaded!")

            elif file_name == "reviews_data" and not os.path.exists(self.data_path):
                print(f"Downloading {file_name} from Google Drive...")
                gdown.download(f"https://drive.google.com/uc?id={file_id}", self.data_path)
                print("Successfully downloaded!")


    def get_product_list(self):
        """Return a sorted list of unique shampoo products from the dataset."""
        return sorted(self.df["product_title"].unique())

    def get_top_k_reviews(self, query_text, selected_product=None, top_k=10):
        """Retrieve top-k most relevant reviews, either for a specific product or across all products."""
        if selected_product:
            filtered_df = self.df[self.df["product_title"] == selected_product].copy()
        else:
            filtered_df = self.df.copy()
        
        if filtered_df.empty:
            return pd.DataFrame()

        # Convert Query Text to Embedding
        response = self.client.embeddings.create(input=[query_text], model="text-embedding-ada-002")
        query_embedding = response.data[0].embedding  

        # Normalize Embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Search FAISS Index for the most similar reviews
        review_embeddings = self.index.reconstruct_n(0, len(filtered_df))  
        distances = np.dot(review_embeddings, query_embedding)  
        top_indices = np.argsort(distances)[-top_k:][::-1]  

        # Retrieve matching reviews
        top_reviews = filtered_df.iloc[top_indices].copy()
        top_reviews["similarity_score"] = distances[top_indices]

        return top_reviews
