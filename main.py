# Agentic RAG Evaluation Framework
# Pipeline: QueryAnalyser -> Retriever -> Evaluator -> Output

import os
import json
from pathlib import Path

from agent.query_analyser import QueryAnalyser
from agent.retriever import Retriever
from evaluator.retrieval_evaluator import RetrievalEvaluator
from agent.query_decomposer import QueryDecomposer

from dotenv import load_dotenv
load_dotenv()

from langsmith import traceable

from datetime import datetime

import random

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'output'

CORPUS_PATH = str(DATA_DIR / "corpus.json")
DATASET_PATH = str(DATA_DIR / "MultiHopRAG.json")

@traceable(name='run_query_for_one')
def run_query_for_one(query, analyser, decomposer, retriever_dict, evaluator, ground_truth):
    # Analyse Query
    analysis = analyser.analyse(query)
    retriever = retriever_dict[analysis['embedder']]

    # Retrieve
    retrieved_nodes = []
    subqueries = None

    if analysis['complexity'] in ('complex', 'very_complex'):
        subqueries = decomposer.decompose(query)
        
        for sq in subqueries:
            retrieved_nodes.extend(retriever.retrieve(sq))
    else:
        retrieved_nodes = retriever.retrieve(query)

    # Evaluate
    result = evaluator.evaluate(query, retrieved_nodes, ground_truth, subqueries)

    return analysis, result

def run_pipeline(queries: list, top_k: int=5):
    """
    full pipeline: analyse -> retrieve -> evaluate
    Input: list of queries to evaluate
    """

    # Step 1: Initialise components
    analyser = QueryAnalyser()
    evaluator = RetrievalEvaluator()
    decomposer = QueryDecomposer()

    # Step 2: Load ground truth
    print("Loading ground truth...")
    ground_truth = evaluator.load_ground_truth(DATASET_PATH)
    print(f"Loaded {len(ground_truth)} ground truth entries")

    # Step 3: Build one retriever per unique embedder
    print("\nBuilding indices for each embedder")
    retrievers = dict()
    unique_embedders = set(analyser.strategies.values())

    for embedder in unique_embedders:
        retriever = Retriever(embedder=embedder, top_k=top_k)
        retriever.build_index(CORPUS_PATH)
        retrievers[embedder] = retriever
    

    # Step 4: Run pipeline for each query - agent picks the embedder based on complexity
    print("Running pipelines...")
    complexity_counts = {complexity: 0 for complexity in analyser.strategies.keys()}

    run_id = f"eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"LangSmith Run ID: {run_id}")

    checkpoint_path = OUTPUT_DIR / f"checkpoint_{run_id}.json"
    
    for count, query in enumerate(queries, start=1):
        
        # Pipeline single run
        analysis, result = run_query_for_one(query, analyser, decomposer, retrievers, evaluator, ground_truth,
            langsmith_extra={'metadata': {'eval_run': run_id, 'dataset_size': len(queries)}})
        complexity_counts[analysis['complexity']] += 1
        
        print(f"\n\n{count}.{query}...\nComplexity=[{analysis['complexity']}]\nHR={result.get('hit_rate', 0):.3f}" \
        f"\nMRR={result.get('mrr', 0):.3f}\nR@5={result.get('recall@5', 0):.3f}\nCR={result.get('context_recall') or 0: .3f}" \
        f"\nDQ={result.get('decomposition_quality') or 0: .3f}")

        if result.get('sub_queries') is not None:
            print("Sub-Queries:")
            for i, sq in enumerate(result['sub_queries'], 1):
                print(f"{count}.{i}:{sq}")

        if count % 250 == 0:
            OUTPUT_DIR.mkdir(exist_ok=True)
            with open(checkpoint_path, 'w') as f:
                json.dump(evaluator.summary(), f, indent=2)
            print(f"Checkpoint saved at query: {count}")


    # Step 5: Summary
    summary = evaluator.summary()
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY ({summary['total_queries']} queries)")
    print(f"{'='*60}")
    print(f"Hit Rate:               {summary['avg_hit_rate']:.4f}")
    print(f"MRR:                    {summary['avg_mrr']:.4f}")
    print(f"Recall@1:               {summary['avg_Recall@1']:.4f}")
    print(f"Recall@5:               {summary['avg_Recall@5']:.4f}")
    print(f"Recall@10:              {summary['avg_Recall@10']:.4f}")
    print(f"Precision@1:            {summary['avg_Precision@1']:.4f}")
    print(f"Precision@5:            {summary['avg_Precision@5']:.4f}")
    print(f"Precision@10:           {summary['avg_Precision@10']:.4f}")
    print(f"Context Recall:         {summary['avg_context_recall']:.4f}")
    print(f"Decomposition Quality:  {summary['avg_decomposition_quality']:.4f}")

    # Step 6: Save the results
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_file = OUTPUT_DIR / f"evaluation_results_{run_id}.json"

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    print("Agentic RAG Eval -- initialized")
    print(f"Data dir: {DATA_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")

    with open(DATASET_PATH, 'r') as f:
        dataset = json.load(f)

    print(f"Total dataset queries: {len(dataset)}")

    random.seed(101)
    sample_indices = random.sample(range(len(dataset)), 500)
    test_queries = [dataset[i]['query'] for i in sorted(sample_indices)]

    # test_queries = [item['query'] for item in dataset]

    print(f"Running pipeline on {len(test_queries)} queries...\n")

    run_pipeline(test_queries, top_k=10)