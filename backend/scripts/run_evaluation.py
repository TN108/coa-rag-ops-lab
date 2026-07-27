# backend/scripts/run_evaluation.py
import json
import time
from pprint import pprint
import requests
from tqdm import tqdm

# Config
BASE_URL = "http://127.0.0.1:8000/api/v1"
DATASET_PATH = "C:/Users/Talha Nasir/Desktop/coa-rag-ops-lab/data/evaluation/evaluation_dataset.json"

# Settings to sweep
TOP_K_VALUES = [3, 5, 7]
MIN_RETRIEVAL_SCORES = [0.05, 0.1, 0.2]

def run_rag_evaluation(top_k=5, min_score=0.1):
    payload = {
        "dataset_path": DATASET_PATH,
        "top_k": top_k,
        "min_retrieval_score": min_score,
        "chunking_method": "semantic",
        "adaptive": False
    }
    try:
        resp = requests.post(f"{BASE_URL}/rag/evaluation/run", json=payload)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        print(f"RAG evaluation endpoint error: {e}")
        return {"summary": {}, "results": []}

def run_coa_evaluation(top_k=5, min_score=0.1):
    payload = {
        "dataset_path": DATASET_PATH,
        "top_k": top_k,
        "min_retrieval_score": min_score,
        "chunking_method": "semantic",
        "adaptive": False
    }
    try:
        resp = requests.post(f"{BASE_URL}/coa/evaluation/run", json=payload)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        print(f"COA evaluation endpoint error: {e}")
        return {"results": []}

def main():
    comparison_results = []

    total_runs = len(TOP_K_VALUES) * len(MIN_RETRIEVAL_SCORES)
    with tqdm(total=total_runs, desc="Running evaluation") as pbar:
        for top_k in TOP_K_VALUES:
            for min_score in MIN_RETRIEVAL_SCORES:
                pbar.set_postfix({"top_k": top_k, "min_score": min_score})

                start = time.perf_counter()
                rag_results = run_rag_evaluation(top_k, min_score)
                rag_time = (time.perf_counter() - start) * 1000

                start = time.perf_counter()
                coa_results = run_coa_evaluation(top_k, min_score)
                coa_time = (time.perf_counter() - start) * 1000

                rag_summary = rag_results.get("summary", {})
                coa_summary = {
                    "latency_ms": sum(r.get("latency_ms", 0) for r in coa_results.get("results", [])) / len(coa_results.get("results", [])) if coa_results.get("results") else 0,
                    "total_questions": len(coa_results.get("results", []))
                }

                comparison_results.append({
                    "top_k": top_k,
                    "min_retrieval_score": min_score,
                    "rag_hit_rate": rag_summary.get("retrieval_hit_rate"),
                    "rag_mrr": rag_summary.get("mean_reciprocal_rank"),
                    "rag_semantic_sim": rag_summary.get("mean_semantic_similarity"),
                    "rag_total_latency_ms": rag_time,
                    "coa_total_latency_ms": coa_time,
                    "coa_total_questions": coa_summary["total_questions"]
                })

                pbar.update(1)

    with open("evaluation_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)

    print("\n=== Evaluation Comparison Table ===")
    for row in comparison_results:
        pprint(row)

if __name__ == "__main__":
    main()