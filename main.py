# Agentic RAG Evaluation Framework
# Pipeline: QueryAnalyser -> Retriever -> Evaluator -> Output

import os
import json
from pathlib import Path

from agent.query_analyser import QueryAnalyser
from agent.retriever import Retriever
from evaluator.retrieval_evaluator import RetrievalEvaluator
from agent.query_decomposer import QueryDecomposer

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'output'
REFERENCE_REPO = Path("~/agentic-rag-eval/reference-repo").expanduser()

CORPUS_PATH = str(REFERENCE_REPO / "dataset/corpus.json")
DATASET_PATH = str(REFERENCE_REPO / "dataset/MultiHopRAG.json")

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

    # OLD # Step 3: Build index once - done once for all queries

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
    
    for count, query in enumerate(queries, start=1):
        
        # Analyse Query
        analysis = analyser.analyse(query)
        complexity_counts[analysis['complexity']] += 1
        print(f"\nQuery: {query}")
        retriever = retrievers[analysis['embedder']]

        # Retrieve
        retrieved_nodes = []

        if analysis['complexity'] in ('complex', 'very_complex'):
            subqueries = decomposer.decompose(query)
            
            for sq in subqueries:
                retrieved_nodes.extend(retriever.retrieve(sq))
        else:
            retrieved_nodes = retriever.retrieve(query)

        # Evaluate
        result = evaluator.evaluate(query, retrieved_nodes, ground_truth)
        print(f"[{analysis['complexity']}] HR={result.get('hit_rate', 0):.3f} MRR={result.get('mrr', 0):.3f} R@5={result.get('recall@5', 0):.3f} | {count}.{query[:60]}...")

    # Step 5: Summary
    summary = evaluator.summary()
    # print(f"\n{'='*60}")
    # print("\nClassification of queries analysed:")
    # print(f"\tSimple:       {complexity_counts['simple']}")
    # print(f"\tComplex:      {complexity_counts['complex']}")
    # print(f"\tVery Complex: {complexity_counts['very_complex']}")
    # print(f"\n\n{'-*-'*50}")
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY ({summary['total_queries']} queries)")
    print(f"{'='*60}")
    print(f"Hit Rate:     {summary['avg_hit_rate']:.4f}")
    print(f"MRR:          {summary['avg_mrr']:.4f}")
    print(f"Recall@1:     {summary['avg_Recall@1']:.4f}")
    print(f"Recall@5:     {summary['avg_Recall@5']:.4f}")
    print(f"Recall@10:    {summary['avg_Recall@10']:.4f}")
    print(f"Precision@1:  {summary['avg_Precision@1']:.4f}")
    print(f"Precision@5:  {summary['avg_Precision@5']:.4f}")
    print(f"Precision@10: {summary['avg_Precision@10']:.4f}")

    # Step 6: Save the results
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_file = OUTPUT_DIR / "evaluation_results.json"

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

    test_queries = [item['query'] for item in dataset[:50]]

    print(f"Running pipeline on {len(test_queries)} queries...\n")

    run_pipeline(test_queries, top_k=10)