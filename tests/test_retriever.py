import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, BASE_DIR)

from agent.retriever import Retriever


DATA_DIR = BASE_DIR / 'data'
CORPUS_PATH = str(DATA_DIR / "corpus.json")

def test_retriever_returns_list():
    """
    Evaluate if the retriever returns a list
    """
    embedder = 'BAAI/llm-embedder'

    retriever = Retriever(embedder)
    retriever.build_index(CORPUS_PATH)

    query = "Who, according to articles in Sporting News, stand to make a profit "\
    "by predicting outcomes such as a team's lead at the end of a quarter or the total "\
    "points scored, and can also capitalize on event hype, like putting $130 on the Cowboys "\
    "to potentially gain $100?"

    retrieved_nodes = retriever.retrieve(query)

    assert isinstance(retrieved_nodes, list), f"Expected a list return, got {type(retrieved_nodes)}"


def test_retriever_nodes_have_text():
    """
    Evaluate if the retrieved nodes have text strings
    """

    embedder = 'BAAI/llm-embedder'

    retriever = Retriever(embedder)
    retriever.build_index(CORPUS_PATH)

    query = "Who, according to articles in Sporting News, stand to make a profit "\
    "by predicting outcomes such as a team's lead at the end of a quarter or the total "\
    "points scored, and can also capitalize on event hype, like putting $130 on the Cowboys "\
    "to potentially gain $100?"

    retrieved_nodes = retriever.retrieve(query)

    for node in retrieved_nodes:
        assert isinstance(node.node.text, str), f"Expected string in node.node.text, got {type(node.node.text)}"


def test_retriever_respects_topk_limit():
    """
    Evaluate if the number of retrieved nodes are in accordance to top_k
    """

    embedder = 'BAAI/llm-embedder'

    query = "Who, according to articles in Sporting News, stand to make a profit "\
    "by predicting outcomes such as a team's lead at the end of a quarter or the total "\
    "points scored, and can also capitalize on event hype, like putting $130 on the Cowboys "\
    "to potentially gain $100?"

    for top_k in [5, 10]:
        retriever = Retriever(embedder, top_k)
        retriever.build_index(CORPUS_PATH)
        retrieved_nodes = retriever.retrieve(query)
        
        assert len(retrieved_nodes) == top_k, f"Expected {top_k} nodes, got {len(retrieved_nodes)} nodes"


if __name__=='__main__':
    test_retriever_returns_list()
    test_retriever_nodes_have_text()
    test_retriever_respects_topk_limit()