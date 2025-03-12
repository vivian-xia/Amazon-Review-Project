import os
import pandas as pd
import nltk
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

nltk.download("wordnet")
nltk.download("omw-1.4")

# -------- TEXT METRICS -------- #
def compute_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return scorer.score(reference, candidate)

def compute_cosine_similarity(reference, candidate):
    ref_response = client.embeddings.create(input=[reference], model="text-embedding-ada-002")
    cand_response = client.embeddings.create(input=[candidate], model="text-embedding-ada-002")

    ref_emb = list(ref_response.data[0].embedding)
    cand_emb = list(cand_response.data[0].embedding)

    score = cosine_similarity([ref_emb], [cand_emb])[0][0]
    return float(score)

def compute_meteor(reference, candidate):
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()
    return meteor_score([reference_tokens], candidate_tokens)

# -------- LLM EVALUATOR -------- #
def call_llm(prompt, model="gpt-4o", temperature=0):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content.strip()

def llm_metric_prompt(metric, question, reviews, answer):
    prompts = {
        "accuracy": f"On a scale of 1 to 5, where 1 is unreliable and 5 is very reliable, rate the factual accuracy of the following answer based only on the information from the reviews. Respond ONLY with a single number (1-5).\n\nAnswer: {answer}\n\nReviews: {reviews}\n\nScore:",
        "relevance": f"On a scale of 1 to 5, where 1 is irrelevant and 5 is highly relevant, rate how well the answer addresses the user's question using only the information from the reviews. Respond ONLY with a single number (1-5).\n\nQuestion: {question}\n\nAnswer: {answer}\n\nReviews: {reviews}\n\nScore:",
        "coherence": f"On a scale of 1 to 5, where 1 is poorly structured and 5 is very well structured, rate the coherence of the answer. Respond ONLY with a single number (1-5).\n\nAnswer: {answer}\n\nScore:",
        "clarity": f"On a scale of 1 to 5, where 1 is unclear and 5 is very clear, rate the clarity of the answer. Respond ONLY with a single number (1-5).\n\nAnswer: {answer}\n\nScore:",
        "consistency": f"On a scale of 1 to 5, where 1 is inconsistent and 5 is very consistent, rate whether the answer avoids contradictions. Respond ONLY with a single number (1-5).\n\nAnswer: {answer}\n\nScore:",
        "sentiment_alignment": f"On a scale of 1 to 5, where 1 is not aligned and 5 is well aligned, rate whether the answer reflects the overall sentiment from the reviews. Respond ONLY with a single number (1-5).\n\nAnswer: {answer}\n\nReviews: {reviews}\n\nScore:"
    }
    return call_llm(prompts[metric])

# -------- MAIN EVALUATION FUNCTION -------- #
def evaluate_answer_cosine(api_key, user_query, retrieved_reviews, generated_answer, export_csv_path="evaluation_logs.csv"):
    global client
    client = OpenAI(api_key=api_key)
    combined_reviews = " ".join(retrieved_reviews['combined_context'].tolist())

    rouge = compute_rouge(combined_reviews, generated_answer)
    cosine_sim = compute_cosine_similarity(combined_reviews, generated_answer)
    meteor = compute_meteor(combined_reviews, generated_answer)

    llm_metrics = {
        metric: llm_metric_prompt(metric, user_query, combined_reviews, generated_answer)
        for metric in ["accuracy", "relevance", "coherence", "clarity", "consistency", "sentiment_alignment"]
    }

    result = {
        "question": user_query,
        "generated_answer": generated_answer,
        "rouge1": rouge['rouge1'].fmeasure,
        "rouge2": rouge['rouge2'].fmeasure,
        "rougeL": rouge['rougeL'].fmeasure,
        "meteor": meteor,
        "cosine_similarity": cosine_sim,
        **llm_metrics
    }

    df = pd.DataFrame([result])
    if os.path.exists(export_csv_path):
        df.to_csv(export_csv_path, mode='a', index=False, header=False)
    else:
        df.to_csv(export_csv_path, index=False)

    return result
