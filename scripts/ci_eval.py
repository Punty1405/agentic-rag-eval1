"""CI Evaluation - Regression Detection"""

import sys
import json
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Your imports here - analyser, decomposer, retriever, evaluator
from agent.query_analyser import QueryAnalyser
from agent.query_decomposer import QueryDecomposer
from agent.retriever import Retriever
from evaluator.retrieval_evaluator import RetrievalEvaluator

# Constants & Paths
BASE_DIR = Path(__file__).parent.parent
CORPUS_PATH = str(BASE_DIR / 'data/corpus.json')
DATASET_PATH = str(BASE_DIR / 'data/MultiHopRAG.json')

BASELINE_PATH = BASE_DIR / "baseline.json"


def load_baseline():
    """Load baseline metrics from baseline.json"""
    with open(BASELINE_PATH, 'r') as f:
        baseline = json.load(f)
    return baseline

def run_evaluation(baseline: dict):
    """Run pipeline on same sample as baseline - but with single embedder to save time on CI Eval Run"""

    EMBEDDER = "BAAI/bge-large-en-v1.5"  # Winner from Day 3 ablation
    
    print(f"CI Mode: Using single embedder ({EMBEDDER}) for speed")

    analyser = QueryAnalyser()
    evaluator = RetrievalEvaluator()
    decomposer = QueryDecomposer()

    print(f"\nBuilding index with {EMBEDDER}...")
    retriever = Retriever(EMBEDDER, top_k=10)
    retriever.build_index(CORPUS_PATH)

    ground_truth = evaluator.load_ground_truth(DATASET_PATH)
    print(f"Loaded {len(ground_truth)} ground truth entries")
    
    with open(DATASET_PATH, 'r') as f:
        dataset = json.load(f)
    
    random.seed(baseline['random_seed'])
    sample_indices = random.sample(range(len(dataset)), baseline['sample_size'])
    target_queries = [dataset[i]['query'] for i in sorted(sample_indices)]

    print(f"\nRunning the pipeline for {len(target_queries)} sample queries")

    for count, query in enumerate(target_queries, 1):

        analysis = analyser.analyse(query)

        retrieved_nodes = []
        sub_queries = None

        if analysis['complexity'] in ['complex', 'very_complex']:
            sub_queries = decomposer.decompose(query)
            nodes = []

            for sq in sub_queries:
                nodes = retriever.retrieve(sq)
                retrieved_nodes.extend(nodes)
        else:
            retrieved_nodes = retriever.retrieve(query)
        
        result = evaluator.evaluate(query, retrieved_nodes, ground_truth, sub_queries)

        print(f"\n\n{count}.{query}...\nComplexity=[{analysis['complexity']}]\nHR={result.get('hit_rate', 0):.3f}" \
        f"\nMRR={result.get('mrr', 0):.3f}\nR@5={result.get('recall@5', 0):.3f}\nCR={result.get('context_recall') or 0: .3f}" \
        f"\nDQ={result.get('decomposition_quality') or 0: .3f}")

        if result.get('sub_queries') is not None:
            print("Sub-Queries:")
            for i, sq in enumerate(result['sub_queries'], 1):
                print(f"{count}.{i}:{sq}")
    
    summary = evaluator.summary()

    return summary



def compare_metrics(baseline, current):
    """
    Compare current metrics to baseline.
    Returns (passed, details_dict)
    passed = True if no regressions detected
    """
    metrics = ['hit_rate', 'context_recall', 'decomposition_quality']
    differences = dict()
    passed = True

    for key in metrics:
        differences[key] = (current[f'avg_{key}'] - baseline['metrics'][key]) / baseline['metrics'][key]

    if any([value<-0.1 for value in differences.values()]):
        passed = False

    return passed, differences

def main():

    """Run CI evaluation and check for regressions"""
    print("="*60)
    print("CI EVALUATION - Regression Detection")
    print("="*60)

    baseline = load_baseline()
    current = run_evaluation(baseline)
    passed, details = compare_metrics(baseline, current)
    
    print(f"\n{'METRIC':<25} {'BASELINE':>10} {'CURRENT':>10} {'CHANGE':>10}")
    print("-"*70)
    
    # Print each metric
    for key in details.keys():
        print(f"{key:<25} {baseline['metrics'][key]:>10.4f} {current[f'avg_{key}']:>10.4f} {details[key]:>+10.2%}")
    
    print("-"*70)
    
    if passed:
        print("PASSED: No regression detected")
        sys.exit(0)
    else:
        print("FAILED: Regression detected (>10% drop)")
        sys.exit(1)
    
if __name__ == "__main__":
    main()