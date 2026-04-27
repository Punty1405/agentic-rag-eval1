import os
from pathlib import Path

from agent.query_analyser import QueryAnalyser

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'output'

REFERENCE_REPO = Path("~/agentic-rag-eval/reference-repo").expanduser()

if __name__ == "__main__":
    print("Agentic RAG Eval -- initialized")
    print(f"Data dir: {DATA_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")

    test_queries = [
        "Who founded Apple?",
        "Compare the revenue of Apple and Microsoft in 2023",
        "What is the capital of France?",
        "What are the differences between Python and JavaScript?"
    ]

    analyser = QueryAnalyser()
    
    for query in test_queries:
        result = analyser.analyse(query)

        print(f"Query: {query}")
        print(f"Complexity: {result['complexity']}")
        print(f"Hops: {result['hops']}")
        print(f"Embedder: {result['embedder']}")
