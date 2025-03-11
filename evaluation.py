import openai
import pandas as pd
import os
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from nltk.translate.meteor_score import meteor_score
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

# Set your OpenAI API key if not using Streamlit secrets
openai.api_key = os.getenv("OPENAI_API_KEY")

# -------- TEXT METRICS -------- #
def compute_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return scorer.score(reference, candidate)

def compute_bertscore(reference, candidate):
    P, R, F1 = bert_score([candidate], [reference], lang="en")
    return {"bert_precision": P.item(), "bert_recall": R.item(), "bert_f1": F1.item()}

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
        "relevance": f"Does the answer directly address the user's question using information from the reviews? Rate from 1 (irrelevant) to 5 (highly relevant).\n\nQuestion: {question}\n\nReviews: {reviews}\n\nAnswer: {answer}\n\nScore:",
        "coherence": f"Is the answer logically structured and coherent? Rate from 1 (poor) to 5 (excellent).\n\nAnswer: {answer}\n\nScore:",
        "clarity": f"Is the answer clearly written and easy to understand? Rate from 1 (unclear) to 5 (very clear).\n\nAnswer: {answer}\n\nScore:",
        "consistency": f"Does the answer avoid internal contradictions? Rate from 1 (inconsistent) to 5 (very consistent).\n\nAnswer: {answer}\n\nScore:",
        "accuracy": f"Is the information factually correct and reliable taken from the reviews with no fabrication? Rate from 1 (unreliable) to 5 (very reliable).\n\nAnswer: {answer}\n\nScore:",
        "sentiment_alignment": f"Does the answer reflect the overall sentiment from the reviews? Rate from 1 (not aligned) to 5 (aligned).\n\nReviews: {reviews}\n\nAnswer: {answer}\n\nScore:"
    }
    return call_llm(prompts[metric])

# -------- MAIN EVALUATION FUNCTION -------- #
def evaluate_answer(user_query, retrieved_reviews, generated_answer, export_csv_path="evaluation_logs.csv"):
    # Prepare reviews as reference
    combined_reviews = " ".join(retrieved_reviews['combined_context'].tolist())

    # Run metrics
    rouge = compute_rouge(combined_reviews, generated_answer)
    bert = compute_bertscore(combined_reviews, generated_answer)
    meteor = compute_meteor(combined_reviews, generated_answer)

    # LLM evaluations
    llm_metrics = {
        metric: llm_metric_prompt(metric, user_query, combined_reviews, generated_answer)
        for metric in ["relevance", "coherence", "clarity", "consistency","accuracy", "sentiment_alignment"]
    }

    # Flatten metrics
    result = {
        "question": user_query,
        "generated_answer": generated_answer,
        "rouge1": rouge['rouge1'].fmeasure,
        "rouge2": rouge['rouge2'].fmeasure,
        "rougeL": rouge['rougeL'].fmeasure,
        "meteor": meteor,
        **bert,
        **llm_metrics
    }

    # Save to CSV
    df = pd.DataFrame([result])
    if os.path.exists(export_csv_path):
        df.to_csv(export_csv_path, mode='a', index=False, header=False)
    else:
        df.to_csv(export_csv_path, index=False)

    return result
