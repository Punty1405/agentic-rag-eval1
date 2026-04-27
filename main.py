# Agentic RAG Evaluation Framework
# Pipeline: QueryAnalyser -> Retriever -> Evaluator -> Output

import os
import json
from pathlib import Path

from agent.query_analyser import QueryAnalyser
from agent.retriever import Retriever
from evaluator.retrieval_evaluator import RetrievalEvaluator

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

    # Step 2: Load ground truth
    print("Loading ground truth...")
    ground_truth = evaluator.load_ground_truth(DATASET_PATH)
    print(f"Loaded {len(ground_truth)} ground truth entries")

    # Step 3: Build index once - done once for all queries
    print("\nBuilding index...")
    first_analysis = analyser.analyse(queries[0])
    retriever = Retriever(embedder=first_analysis['embedder'], top_k=top_k)
    retriever.build_index(CORPUS_PATH)

    # Step 4: Run pipeline for each query
    print("Running pipelines...")
    
    for query in queries:
        
        # Analyse Query
        analysis = analyser.analyse(query)
        print(f"\nQuery: {query}")
        print(f"Complexity: {analysis['complexity']} | Hops: {analysis['hops']}")

        # Retrieve
        retrieved_nodes = retriever.retrieve(query)

        # Evaluate
        result = evaluator.evaluate(query, retrieved_nodes, ground_truth)
        print(f"Hit Rate: {result.get('hit_rate', 'N/A')}")

    # Step 5: Summary
    summary = evaluator.summary()
    print(f"\n\n{'-*-'*50}")
    print(f"EVALUATION SUMMARY")
    print(f"\n\n{'-*-'*50}")
    print(f"Total queries: {summary['total_queries']}")
    print(f"Average hit rate: {summary['average_hit_rate']}")

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

    # test_queries = [
    #     "Who founded Apple?",
        # "Compare the revenue of Apple and Microsoft in 2023",
        # "What is the capital of France?",
        # "What are the differences between Python and JavaScript?"
    # ]

    # analyser = QueryAnalyser()
    
    # for query in test_queries:
    #     result = analyser.analyse(query)

    #     print(f"Query: {query}")
    #     print(f"Complexity: {result['complexity']}")
    #     print(f"Hops: {result['hops']}")
    #     print(f"Embedder: {result['embedder']}")
    # retriever = Retriever(embedder='BAAI/llm-embedder', top_k=5)
    # retriever.build_index(CORPUS_PATH)

    # results = retriever.retrieve(test_queries[0])


    # print(f"Retrived {len(results)} documents for: {test_queries[0]}")

    # for i, node in enumerate(results):
    #     print(f"\n[{i+1}] Score: {node.score:.4f}")
    #     print(f"Text: {node.node.text[:150]}...")


# Start with 5 queries from the dataset — small run to verify pipeline
    test_queries = ["Who is the individual associated with the cryptocurrency industry facing a criminal trial on fraud and conspiracy charges, as reported by both The Verge and TechCrunch, and is accused by prosecutors of committing fraud for personal gain?",
    "Which individual is implicated in both inflating the value of a Manhattan apartment to a figure not yet achieved in New York City's real estate history, according to 'Fortune', and is also accused of adjusting this apartment's valuation to compensate for a loss in another asset's worth, as reported by 'The Age'?",
    "Who is the figure associated with generative AI technology whose departure from OpenAI was considered shocking according to Fortune, and is also the subject of a prevailing theory suggesting a lack of full truthfulness with the board as reported by TechCrunch?",
    "Do the TechCrunch article on software companies and the Hacker News article on The Epoch Times both report an increase in revenue related to payment and subscription models, respectively?",
    "Which online betting platform provides a welcome bonus of up to $1000 in bonus bets for new customers' first losses, runs NBA betting promotions, and is anticipated to extend the same sign-up offer to new users in Vermont, as reported by both CBSSports.com and Sporting News?"
    ]

    run_pipeline(test_queries)