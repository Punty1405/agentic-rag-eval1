# Retrieval Evaluator - measures quality of retrieved documents
# Compares retrieved results against ground truth from dataset

import json
from pathlib import Path

class RetrievalEvaluator:
    def __init__(self):
        self.results = []

    def load_ground_truth(self, dataset_path: str) -> dict:
        """
        Loads ground truth answers and supporting evidence.
        Input:
        dataset_path: string - path to MultiHopRAG.json
        Returns: dict of query -> supporting evidence        
        """

        with open(dataset_path, 'r') as f:
            data = json.load(f)

        ground_truth = {}

        for item in data:
            ground_truth[item['query']] = {
                'answer': item['answer'],
                'evidence': [e['fact'] for e in item['evidence_list']]
            }

        return ground_truth

    
    def evaluate(self, query: str, retrieved_nodes: list, ground_truth: dict) -> dict:
        """
        Evaluates retrieval quality for a single query.
        Measures: Hit Rate - did we retrieve any relevant document?
        """

        if query not in ground_truth:
            return {"query": query, "error": "Query not in ground truth"}
        
        expected_evidence = ground_truth[query]['evidence']
        retrieved_evidence = [node.node.text for node in retrieved_nodes]

        hits = 0

        # print(expected_evidence)
        # print(retrieved_evidence)

        for evidence in expected_evidence:
            for retrieved in retrieved_evidence:
                # print()
                if evidence.lower() in retrieved.lower():
                    hits += 1
                    break

        hit_rate = hits / len(expected_evidence) if expected_evidence else 0

        result = {
            'query': query,
            'expected_evidence_count': len(expected_evidence),
            'hits': hits,
            'hit_rate': round(hit_rate, 4),
            'answer': ground_truth[query]['answer']
        }

        self.results.append(result)

        return result

    def summary(self) -> dict:
        """
        Summarizes evaluation across all queries
        """

        if not self.results:
            return {"error": "No results to summarise"}

        avg_hit_rate = sum(r['hit_rate'] for r in self.results) / len(self.results)

        return {
            'total_queries': len(self.results),
            'average_hit_rate': round(avg_hit_rate, 4),
            'results': self.results
        }