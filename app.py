"""
FastAPI Endpoint for Agentic MultiHop RAG Pipeline
POST /query - Run avluation on a single query
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
from pathlib import Path

# Imports for pipeline
from agent.query_analyser import QueryAnalyser
from agent.query_decomposer import QueryDecomposer
from agent.retriever import Retriever
from evaluator.retrieval_evaluator import RetrievalEvaluator

app = FastAPI(
    title='Agentic Multi-Hop Evaluation API',
    description='Multi-Hop RAG with dual evaluation (structural and semantic)',
    version='1.0.0'
)

# Paths
BASE_DIR = Path(__file__).parent
CORPUS_PATH = str(BASE_DIR / 'data/corpus.json')
DATASET_PATH = str(BASE_DIR / 'data/MultiHopRAG.json')

# Initializing components on startup
analyser = None
decomposer = None
evaluator = None
retrievers = None
ground_truth = None

@app.on_event('startup')
def load_pipeline():
    """
    Initializing pipeline componenets at the start - once
    """

    global analyser, decomposer, evaluator, retrievers, ground_truth

    print("Initializing pipeline componenets...")
    analyser = QueryAnalyser()
    evaluator = RetrievalEvaluator()
    decomposer = QueryDecomposer()

    retrievers = dict()

    for embedder in set(analyser.strategies.values()):
        retriever = Retriever(embedder, top_k=10)
        retriever.build_index(CORPUS_PATH)
        retrievers[embedder] = retriever
    
    ground_truth = evaluator.load_ground_truth(DATASET_PATH)
    print(f"Loaded {len(ground_truth)} ground truth entries")

    print("Pipeline ready!")

class QueryRequest(BaseModel):
    query: str = Field(description='Query input from the user')

class QueryResponse(BaseModel):
    query: str = Field(description='Query input from the user')
    analysis: dict = Field(
        description='Dictionary of result post-query analysis. Includes complexity, estimated_hops, query_intent, entities, relationship')
    sub_queries: List[str] = Field(description='List of subqueries decomposed by decomposer as string', default=None)
    retrieved_context: List[str] = Field(description='List of retrieved contexts from the retriever')
    evaluation: dict = Field(
        description='Dictionary of result post-retrieval analysis. Includes hiit_rate, context_recall and decomposition_quality')

@app.post('/query', response_model=QueryResponse)
def evaluate_query(request: QueryRequest):
    """
    Passing through a single query through RAG pipeline and Evaluating.
    Returns: analysis, retrieval and evaluation metrics
    """

    analysis = analyser.analyse(request.query)

    retriever = retrievers[analysis['embedder']]
    retrieved_nodes = []
    sub_queries = None

    if analysis['complexity'] in ['complex', 'very_complex']:
        sub_queries = decomposer.decompose(request.query)

        for sq in sub_queries:
            nodes = retriever.retrieve(sq)
            retrieved_nodes.extend(nodes)
    else:
        retrieved_nodes = retriever.retrieve(request.query)

    result = evaluator.evaluate(request.query, retrieved_nodes, ground_truth, sub_queries)

    return QueryResponse(
        query=request.query,
        analysis=analysis,
        sub_queries=sub_queries,
        retrieved_context=[node.node.text for node in retrieved_nodes],
        evaluation={
            'hit_rate': result.get('hit_rate'),
            'context_recall': result.get('context_recall'),
            'decomposition_quality': result.get('decomposition_quality')
        }
    )

@app.get("/")
def root():
    """
    Health check endpoint
    """

    return {
        'status': 'online',
        'message': 'Agentic Multi-Hop RAG Evaluation API',
        'endpoints': {
            'POST /query': 'Evaluate a query',
            'GET /docs': 'API documentation'
        }
    }
