# Retriever - wraps LlamaIndex retrieval pipeline
# Called by the agent with embedder strategy from QueryAnalyser

import sys
from pathlib import Path

REFERENCE_REPO = Path("~/agentic-rag-eval/reference-repo").expanduser()
sys.path.insert(0, str(REFERENCE_REPO))

from llama_index import VectorStoreIndex, ServiceContext
from llama_index.embeddings import HuggingFaceEmbedding
from util import JSONReader

class Retriever:
    def __init__(self, embedder: str, top_k: int=5):
        self.embedder = embedder
        self.top_k = top_k
        self.index = None

    def build_index(self, corpus_path: str):
        """
        Loads corpus and builds vector index.
        Input: corpus_path - path to corpus.json
        """

        reader = JSONReader()
        documents = reader.load_data(corpus_path)

        embed_model = HuggingFaceEmbedding(
            model_name=self.embedder,
            trust_remote_code=True
        )

        service_context = ServiceContext.from_defaults(
            embed_model=embed_model,
            llm=None
        )

        self.index = VectorStoreIndex.from_documents(
            documents,
            service_context=service_context,
            show_progress=True
        )
        
        print(f"Index built with {len(documents)} documents")

    def retrieve(self, query: str) -> list:
        """
        Retrieves top_k relevant documents for a query.
        Input: query - str
        Returns: list of retrieved nodes
        """

        if self.index is None:
            raise ValueError("Index not built. Call build_index() first ")

        retriever = self.index.as_retriever(similarity_top_k=self.top_k)
        results = retriever.retrieve(query)
        return results

