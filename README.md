# Amazon-Review-Project

A chatbot created to answer questions on Amazon products based off review similarity to the user prompt. It takes into account the sentiment of these reviews before providing a summarized answer with various AI agents using OpenAI API. 

The Amazon product reviews are based off of this following [Amazon Review dataset (1995 until 2015)](https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset?resource=download&select=amazon_reviews_us_Beauty_v1_00.tsv).
From this dataset, the beauty products, specifically shampoo products were explored, chunked, embedded. 

The result is an interactive chatbot that outputs the response to the question on products reviewed including which products are recommended for certain concerns or questions on the specific products themselves.

The app can be viewed at the following link: [Amazon Shampoo Review Chatbot](https://vivian-xia-am-amazon-shampoo-retrieval-sentiment-reviews-gsv0rz.streamlit.app/), corresponding to amazon_shampoo_retrieval_sentiment_reviews.py.

The experiment_app.py utilizes the same primary functions and packages with the purpose of comparing the quality of answers using different parameters for this use case.
