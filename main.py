import os
from pathlib import Path

from agent.query_analyser import QueryAnalyser
from agent.retriever import Retriever

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'output'

REFERENCE_REPO = Path("~/agentic-rag-eval/reference-repo").expanduser()
CORPUS_PATH = str(REFERENCE_REPO / "dataset/corpus.json")

if __name__ == "__main__":
    print("Agentic RAG Eval -- initialized")
    print(f"Data dir: {DATA_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")

    test_queries = [
        "Who founded Apple?",
        # "Compare the revenue of Apple and Microsoft in 2023",
        # "What is the capital of France?",
        # "What are the differences between Python and JavaScript?"
    ]

    # analyser = QueryAnalyser()
    
    # for query in test_queries:
    #     result = analyser.analyse(query)

    #     print(f"Query: {query}")
    #     print(f"Complexity: {result['complexity']}")
    #     print(f"Hops: {result['hops']}")
    #     print(f"Embedder: {result['embedder']}")
    retriever = Retriever(embedder='BAAI/llm-embedder', top_k=5)
    retriever.build_index(CORPUS_PATH)

    results = retriever.retrieve(test_queries[0])


    print(f"Retrived {len(results)} documents for: {test_queries[0]}")

    for i, node in enumerate(results):
        print(f"\n[{i+1}] Score: {node.score:.4f}")
        print(f"Text: {node.node.text[:150]}...")
