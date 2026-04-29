# Query Analyzer - Agent's decision-making component
# Analyzes incoming query and selects retrieval strategy

import os
from openai import OpenAI
from dotenv import load_dotenv
import json

SYSTEM_PROMPT = """
You are a query analysis assistant for a multi-hop retrieval system. Analyze the given query and return a structured JSON response.

Return ONLY valid JSON in this exact format:
{
  "complexity": "simple" | "complex" | "very_complex",
  "estimated_hops": 1 | 2 | 3,
  "query_intent": "factual" | "comparative" | "temporal" | "causal" | "inferential",
  "entities": ["entity1", "entity2"],
  "relationships": ["what to find about whom"]
}

Definitions:
- simple: single fact lookup
- complex: requires combining 2 facts
- very_complex: requires combining 3+ facts or chained reasoning
"""

class QueryAnalyser:
    def __init__(self, model: str='gpt-4o-mini'):
        
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')

        if not api_key:
            raise ValueError("OPENAI_API_KEY Missing")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        
        self.strategies = {
            'simple': 'BAAI/llm-embedder',
            'complex': 'BAAI/bge-large-en-v1.5',
            'very_complex': 'BAAI/bge-large-en-v1.5',
        }

    def analyse(self, query: str) -> dict:
        """
        Analyses the incoming query complexity and returns retrieval strategy.
        Input:
        query: str - incoming query
        Returns: dict - with complexity, estimated hops and embedder
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {'role':'system', 'content':SYSTEM_PROMPT},
                {'role':'user', 'content': query}
            ],
            temperature=0.0,
            response_format={'type':'json_object'}
        )

        analysis = json.loads(response.choices[0].message.content)
        analysis['query'] = query
        analysis['embedder'] = self.strategies[analysis['complexity']]

        # complexity = self._assess_complexity(query)
        # hops = self._estimate_hops(query)
        # embedder = self.strategies[complexity]

        return analysis
    
    # def _assess_complexity(self, query: str) -> str:
    #     """
    #     Simple heuristic - we'll make this smarter later
    #     """

    #     complex_indicators = "and both compare difference between also additionally while".split()

    #     query_lower = query.lower()

    #     if any(indicator in query_lower for indicator in complex_indicators):
    #         return 'complex'
    #     else:
    #         return 'simple'
        
    # def _estimate_hops(self, query: str) -> int:
    #     """
    #     Estimate number of retrieval hops needed
    #     """

    #     if self._assess_complexity(query) == 'simple':
    #         return 1
    #     return 2    