import os
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.query_analyser import QueryAnalyser

def test_query_analyser_output_structure():
    """
    Evaluate the output from query analyser - expecting a dict return with
    following keys: "query", "embedder", "complexity", "estimated_hops", "query_intent", "entities", "relationships"
    """

    analyser = QueryAnalyser()

    query = "Who founded Apple?"

    analysis = analyser.analyse(query)

    required_keys = ['complexity', 'estimated_hops', 'query_intent', 'entities', 'relationships', 'query', 'embedder']

    for key in required_keys:
        assert key in analysis, f"Missing key: {key}"
    

def test_query_analyser_output_bounds():
    """
    Evaluate that the analyser outputs are bound the following values:
    "complexity": "simple" | "complex" | "very_complex",
    "estimated_hops": 1 | 2 | 3,
    "query_intent": "factual" | "comparative" | "temporal" | "causal" | "inferential"
    """

    queries = [
        "Who, according to articles in Sporting News, stand to make a profit by predicting outcomes such as a team's lead at the end of a quarter or the total points scored, and can also capitalize on event hype, like putting $130 on the Cowboys to potentially gain $100?",
        "Who is the individual that was once likened to a prominent investor, admitted to challenges in overseeing a rapidly expanding crypto company, faced allegations of fraud in a legal setting, and discussed corporate governance intentions with a venture firm, as reported by TechCrunch, The Verge, Cnbc, and TechCrunch respectively?",
        "Who is the individual who, after Judge Lewis Kaplan's intervention, admitted to being informed about a financial discrepancy and is also alleged by the prosecution to have knowingly committed fraud, as reported by The Verge and TechCrunch?",
        "After the report by Fortune on October 4, 2023, regarding Sam Bankman-Fried's alleged use of Caroline Ellison as a front at Alameda Research, and the subsequent report by TechCrunch involving Sam Bankman-Fried's alleged motives for committing fraud, is the portrayal of Sam Bankman-Fried's actions by both news sources consistent?",
        'Who founded Apple?'
    ]

    analyser = QueryAnalyser()

    for query in queries:
        analysis = analyser.analyse(query)

        assert analysis['complexity'] in ["simple", "complex", "very_complex"], f"Query 'complexity' out of bounds, got {analysis['complexity']}"
        assert 1 <= analysis['estimated_hops'] <= 3, f"Estimated Hops is out of bounds, got {analysis['estimated_hops']}"
        assert analysis['query_intent'] in ["factual", "comparative", "temporal", "causal", "inferential"], f"Query 'query_intent' out of bounds, got {analysis['query_intent']}"
    
def test_query_analyser_output_type():
    """
    Evaluate that the analyser outputs are of the following types:
    "complexity": str,
    "estimated_hops": int,
    "query_intent": str,
    "entities": str,
    "relationships": str
    """

    queries = [
        "Who, according to articles in Sporting News, stand to make a profit by predicting outcomes such as a team's lead at the end of a quarter or the total points scored, and can also capitalize on event hype, like putting $130 on the Cowboys to potentially gain $100?",
        "Who is the individual that was once likened to a prominent investor, admitted to challenges in overseeing a rapidly expanding crypto company, faced allegations of fraud in a legal setting, and discussed corporate governance intentions with a venture firm, as reported by TechCrunch, The Verge, Cnbc, and TechCrunch respectively?",
        "Who is the individual who, after Judge Lewis Kaplan's intervention, admitted to being informed about a financial discrepancy and is also alleged by the prosecution to have knowingly committed fraud, as reported by The Verge and TechCrunch?",
        "After the report by Fortune on October 4, 2023, regarding Sam Bankman-Fried's alleged use of Caroline Ellison as a front at Alameda Research, and the subsequent report by TechCrunch involving Sam Bankman-Fried's alleged motives for committing fraud, is the portrayal of Sam Bankman-Fried's actions by both news sources consistent?",
        'Who founded Apple?'
    ]

    analyser = QueryAnalyser()

    for query in queries:
        analysis = analyser.analyse(query)

        assert isinstance(analysis['complexity'], str), f"Complexity is not a str, got {type(analysis['complexity'])}"
        assert isinstance(analysis['estimated_hops'], int), f"Estimated Hops is not an int, got {type(analysis['estimated_hops'])}"
        assert isinstance(analysis['query_intent'], str), f"Query intent is not a str, got {type(analysis['query_intent'])}"
        assert isinstance(analysis['entities'], list), f"Entities is not a str, got {type(analysis['entities'])}"
        assert isinstance(analysis['relationships'], list), f"Relationships is not a str, got {type(analysis['relationships'])}"

if __name__ == '__main__':
    test_query_analyser_output_structure()
    test_query_analyser_output_bounds()
    test_query_analyser_output_type()
