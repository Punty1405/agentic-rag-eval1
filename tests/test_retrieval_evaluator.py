import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluator.retrieval_evaluator import RetrievalEvaluator

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
    },
    "Does the article from The Verge suggest that Google's influence on the internet's appearance is the same aspect of its impact as the financial influence on platforms described in the TechCrunch article about Google's spending, and is the anticompetitive behavior towards news publishers mentioned in another TechCrunch article a separate issue from these influences?" : {
        'answer': 'Whatever',
        'evidence': [
            'The company did accomplish that goal, but in doing so, it inadvertently and profoundly changed how the internet looked.',
            'In our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how Google tried to have Chrome preinstalled on iPhones.',
            'The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google siphons off news publishers content, their readers and ad revenue through anticompetitive means.'
        ]
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


def test_context_recall_valid():
    """
    Evaluating context_recall - should be float (0.0, 1.0)
    """

    evaluator = RetrievalEvaluator()

    nodes = make_mock_nodes([
        "Apple was founded in 1976 by Steve Jobs",
        "Whatever microsoft is - it is"
    ])

    result = evaluator.evaluate("Who founded Apple?", nodes, GROUND_TRUTH)
    
    assert isinstance(result['context_recall'], float), f"Expected float value between 0.0 and 1.0, got {result['context_recall']}"
    assert 0.0 <= result['context_recall'] <= 1.0, f"Expected float value between 0.0 and 1.0, got {result['context_recall']}"


def test_context_recall_none_when_no_reference():
    """
    Context Recall is None when no ground truth exists
    """

    evaluator = RetrievalEvaluator()

    nodes = make_mock_nodes([
        "Boney Kapoor is a Boney man with bones for tongue and ears",
        "Boney Kapoor's wife si Sreedevi"
    ])

    result = evaluator.evaluate("Who is Boney Kapoor?", nodes, GROUND_TRUTH)
    assert result['error'] == 'Query not in ground truth', f"Expected None, got {result}"



def test_decomposition_quality_without_subqueries():
    """
    Decomposition quality score is expected only for complex / very complex queries
    """

    evaluator = RetrievalEvaluator()

    nodes = make_mock_nodes([
        "Apple was founded in 1976 by Steve Jobs",
        "Whatever microsoft is - it is"
    ])

    result = evaluator.evaluate("Who founded Apple?", nodes, GROUND_TRUTH)
    
    assert result['decomposition_quality'] is None, f"Expected DQ to be None, got {result['decomposition_quality']}"


def test_decomposition_quality_with_subqueries():
    """
    Complex queries should have decomposition_quality score
    """

    evaluator = RetrievalEvaluator()

    nodes = make_mock_nodes([
            'The company did accomplish that goal, but in doing so, it inadvertently and profoundly changed how the internet looked.',
            'In our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how Google tried to have Chrome preinstalled on iPhones.',
            'The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google siphons off news publishers content, their readers and ad revenue through anticompetitive means.'
        ])

    sub_queries = [
        "What does the article from The Verge say about Google's influence on the internet's appearance?",
        "What does the TechCrunch article about Google's spending describe regarding its financial influence on platforms?",
        "What anticompetitive behavior towards news publishers is mentioned in the TechCrunch article?",
        "Are the influences described in the Verge and TechCrunch articles considered the same aspect of Google's impact?",
        "Is the anticompetitive behavior towards news publishers a separate issue from the influences discussed in the other articles?"
        ]
    
    result = evaluator.evaluate(
        "Does the article from The Verge suggest that Google's influence on the internet's appearance is the same aspect of its impact as the financial influence on platforms described in the TechCrunch article about Google's spending, and is the anticompetitive behavior towards news publishers mentioned in another TechCrunch article a separate issue from these influences?",
        nodes, GROUND_TRUTH, sub_queries)

    assert isinstance(result['decomposition_quality'], float), f"Expected float value between 0.0 and 1.0, got {result['decomposition_quality']}"
    assert 0.0 <= result['decomposition_quality'] <= 1.0, f"Expected float value between 0.0 and 1.0, got {result['decomposition_quality']}"
    
    
if __name__ == '__main__':
    test_happy_path()
    test_negative_path()
    test_semantic_edge_case()
    test_context_recall_valid()
    test_context_recall_none_when_no_reference()
    test_decomposition_quality_without_subqueries()
    test_decomposition_quality_with_subqueries()


    print("All evaluator tests passed")