import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))
from retrieval_evaluator import RetrievalEvaluator

def make_mock_nodes(texts: list) -> list:
    """
    Helper - creates mock retrieved nodes from text list
    """

    nodes = []

    for text in texts:
        node = MagicMock()
        node.node.text = text
        nodes.append(node)

    return nodes

GROUND_TRUTH = {
    'Who founded Apple?': {
        'answer': 'Steve Jobs',
        'evidence': ['Apple was founded in 1976 by Steve Jobs']
    }
}

def test_happy_path():
    """
    Retrieved chunk contains exact evidence - hit_rate should be 1.0
    """

    evaluator = RetrievalEvaluator()

    nodes = make_mock_nodes([
        "Apple was founded in 1976 by Steve Jobs",
        "Whatever microsoft is - it is"
    ])

    result = evaluator.evaluate("Who founded Apple?", nodes, GROUND_TRUTH)
    # print(result)
    assert result['hit_rate'] == 1.0, f"Expected 1.0 got {result['hit_rate']}"
    print("Happy Path PASS")

def test_negative_path():
    """
    Retrieved chunk contain no relevant evidence - hit_rate should be 0.0
    """

    evaluator = RetrievalEvaluator()
    nodes = make_mock_nodes([
        "Google was founded by Larry Page",
        "Microsoft released Windows 11 last year"
    ])

    result = evaluator.evaluate("Who founded Apple?", nodes, GROUND_TRUTH)
    assert result['hit_rate'] == 0.0, f"Expected 0.0, got {result['hit_rate']}"
    print("Negative Path PASS")

def test_semantic_edge_case():
    """
    Retrieved chunk semantically correct, but worded differently.
    'in' check fails - hit_rate = 0 even though answer is present
    Known limitation of the substring matching
    """

    evaluator = RetrievalEvaluator()
    nodes = make_mock_nodes([
        "Jobs co-founded Apple Computer Company in April 1976",
    ])

    result = evaluator.evaluate("Who founded Apple?", nodes, GROUND_TRUTH)
    assert result['hit_rate'] == 0.0, f"Expected 0.0 (known limitation), got {result['hit_rate']}"
    print("Semantic Negative Path PASS - known limitation confirmed")


if __name__ == '__main__':
    test_happy_path()
    test_negative_path()
    test_semantic_edge_case()

    print("All evaluator tests passed")