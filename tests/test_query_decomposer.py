import os
import sys
from pathlib import Path

sys.path.insert(0, Path(__file__).parent.parent)

from agent.query_decomposer import QueryDecomposer

def test_decomposer_returns_list():
    """
    Evaluate if the query decomposer always returns a list
    """

    decomposer = QueryDecomposer()

    query = "Does the article from The Verge suggest that Google's influence on the internet's appearance is the same aspect of its impact as the financial influence on platforms described in the TechCrunch article about Google's spending, and is the anticompetitive behavior towards news publishers mentioned in another TechCrunch article a separate issue from these influences?"

    sub_queries = decomposer.decompose(query)

    assert isinstance(sub_queries, list), f"Expected decomposer output to be a list, got {type(sub_queries)}"


def test_decomposer_elements_are_strings():
    """
    Evaluate if the query decomposer always returns a list of strings
    """

    decomposer = QueryDecomposer()

    query = "Does the article from The Verge suggest that Google's influence on the internet's appearance is the same aspect of its impact as the financial influence on platforms described in the TechCrunch article about Google's spending, and is the anticompetitive behavior towards news publishers mentioned in another TechCrunch article a separate issue from these influences?"

    sub_queries = decomposer.decompose(query)

    for sq in sub_queries:
        assert isinstance(sq, str), f"Expected sub-queries to be a str, got {type(sq)}"


def test_decomposer_generates_multiple_subqueries():
    """For multi-hop queries, decomposer should generate 2+ sub-questions"""
    
    decomposer = QueryDecomposer()
    
    query = "Does the article from The Verge suggest that Google's influence on the internet's appearance is the same aspect of its impact as the financial influence on platforms described in the TechCrunch article about Google's spending, and is the anticompetitive behavior towards news publishers mentioned in another TechCrunch article a separate issue from these influences?"
    
    sub_queries = decomposer.decompose(query)
    
    assert len(sub_queries) >= 2, f"Expected 2+ sub-queries for multi-hop query, got {len(sub_queries)}"


if __name__ == '__main__':
    test_decomposer_returns_list()
    test_decomposer_elements_are_strings()
    test_decomposer_generates_multiple_subqueries()

