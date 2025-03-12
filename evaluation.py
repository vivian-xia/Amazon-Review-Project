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
    ref_emb = openai.Embedding.create(input=[reference], model="text-embedding-ada-002")['data'][0]['embedding']
    cand_emb = openai.Embedding.create(input=[candidate], model="text-embedding-ada-002")['data'][0]['embedding']
    score = cosine_similarity([ref_emb], [cand_emb])[0][0]
    return float(score)

def compute_meteor(reference, candidate):
    return meteor_score([reference], candidate)

# -------- LLM EVALUATOR -------- #
def call_llm(prompt, model="gpt-4o", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response['choices'][0]['message']['content']

def llm_metric_prompt(metric, question, reviews, answer):
    prompts = {
        "accuracy": f"Is the information factually correct and reliable taken from the reviews with no fabrication? Rate from 1 (unreliable) to 5 (very reliable).\n\nAnswer: {answer}\n\nScore:",
        "relevance": f"Does the answer directly address the user's question using information from the reviews? Rate from 1 (irrelevant) to 5 (highly relevant).\n\nQuestion: {question}\n\nReviews: {reviews}\n\nAnswer: {answer}\n\nScore:",
        "coherence": f"Is the answer logically structured and coherent? Rate from 1 (poor) to 5 (excellent).\n\nAnswer: {answer}\n\nScore:",
        "clarity": f"Is the answer clearly written and easy to understand? Rate from 1 (unclear) to 5 (very clear).\n\nAnswer: {answer}\n\nScore:",
        "consistency": f"Does the answer avoid internal contradictions? Rate from 1 (inconsistent) to 5 (very consistent).\n\nAnswer: {answer}\n\nScore:",
        "sentiment_alignment": f"Does the answer reflect the overall sentiment from the reviews? Rate from 1 (not aligned) to 5 (aligned).\n\nReviews: {reviews}\n\nAnswer: {answer}\n\nScore:"
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
