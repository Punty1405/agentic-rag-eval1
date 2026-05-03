# Retrieval Evaluator - measures quality of retrieved documents
# Compares retrieved results against ground truth from dataset

import json
from pathlib import Path
from rapidfuzz import fuzz

from langsmith import traceable

from ragas.metrics.collections import ContextRecall
from ragas.llms import llm_factory
from openai import AsyncOpenAI
from evaluator.decomposition_quality import DecompositionQuality
import asyncio

class RetrievalEvaluator:
    def __init__(self):
        self.results = []

        # Initializing RAGAS metrics
        self.llm = llm_factory('gpt-4o-mini', client=AsyncOpenAI())
        self.context_recall = ContextRecall(llm=self.llm)
        self.decomposition_quality = DecompositionQuality(llm=self.llm)


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

    
    @traceable(name="RetrievalEvaluator.evaluate")
    def evaluate(
        self, query: str, retrieved_nodes: list, ground_truth: dict, sub_queries: list=None, k_values : list=[1, 5, 10]
        ) -> dict:
        """
        Evaluates retrieval quality for a single query.
        Measures: dict with hit_rate, MRR, Recall@5, Recall@10, Precision@5 and Precision@10
        """

        if query not in ground_truth:
            return {"query": query, "error": "Query not in ground truth"}
        
        expected_evidence = ground_truth.get(query, {}).get('evidence', [])
        retrieved_evidence = [node.node.text for node in retrieved_nodes]
        total_expected = len(expected_evidence)
        total_retrieved = len(retrieved_evidence)

        # Capture relevant ranks of the retrieved_nodes
        relevant_ranks = set()
        hits = 0

        for expected in expected_evidence:
            for rank, retrieved in enumerate(retrieved_evidence, start=1):
                if fuzz.partial_ratio(expected.lower(), retrieved.lower()) >= 75:
                    relevant_ranks.add(rank)
                    hits += 1
                    break

        # Hit Rate — fraction of expected evidence found anywhere in retrieved
        hit_rate = hits / len(expected_evidence) if expected_evidence else 0

        # MRR - Mean Reciprocal Rank
        mrr = 1.0 / min(relevant_ranks) if relevant_ranks else 0

        # Recall@K and Precision@K
        recall_at_k = dict()
        precision_at_k = dict()
        for k in k_values:
            relevant_in_top_k = sum(1 for r in relevant_ranks if r <= k)
            recall_at_k[f'Recall@{k}'] = relevant_in_top_k / total_expected if total_expected else 0
            precision_at_k[f'Precision@{k}'] = relevant_in_top_k / k if k <= total_retrieved else relevant_in_top_k / total_retrieved

        # Preparing countext and reference for RAGAS
        retrieved_contexts = retrieved_evidence
        reference = " ".join(expected_evidence)

        # Context Recall calculation
        cr_score_value = None
        if reference.strip():
            cr_score = asyncio.run(
                self.context_recall.ascore(
                    user_input=query,
                    retrieved_contexts=retrieved_contexts,
                    reference=reference
                )
            )
            cr_score_value = cr_score.value

        # Decomposition Quality calculation
        decomposition_score = None
        if sub_queries:
            decomp_result = self.decomposition_quality.score(query, sub_queries)
            decomposition_score = decomp_result['score']


        result = {
            'query': query,
            'hit_rate': round(hit_rate, 4),
            'mrr': round(mrr, 4),
            **{k: round(v, 4) for k,v in recall_at_k.items()},
            **{k: round(v, 4) for k,v in precision_at_k.items()},
            'total_expected': total_expected,
            'total_retrieved': total_retrieved,

            # RAGAS integration
            'context_recall': cr_score_value,
            'decomposition_quality': decomposition_score,
            'sub_queries': sub_queries
        }

        self.results.append(result)

        return result

    def summary(self) -> dict:
        """
        Summarizes evaluation across all queries
        """

        if not self.results:
            return {"error": "No results to summarise"}

        metrics_keys = ['hit_rate', 'mrr', 'Recall@1', 'Recall@5', 'Recall@10', 'Precision@1', 'Precision@5', 'Precision@10', 'context_recall',
         'decomposition_quality']

        averages = dict()

        for key in metrics_keys:
            values = [each_result.get(key) for each_result in self.results
            if 'error' not in each_result and each_result.get(key) is not None]

        if values:            
            averages[f'avg_{key}'] = round(sum(values)/len(values), 4) if values else 0
        else:
            averages[f'avg_{key}'] = 0

        return {
            'total_queries': len(self.results),
            **averages,
            'results': self.results
        }