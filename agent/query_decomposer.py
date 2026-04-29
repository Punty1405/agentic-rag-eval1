# Query Decomposer - breaks complex multi-hop queries into sub-queries
# User OpenAI API to reason about query structure

import os
from openai import OpenAI
from dotenv import load_dotenv

SYSTEM_PROMPT = """
You are a query decomposition assistant. Break complex multi-hop queries into independent sub-queries that can each be answered separately.
Each sub-query should retrieve a distinct piece of evidence needed to answer the original query.

Return ONLY the sub-queries, one per line. No numbering, no explanations, no preamble.
"""

class QueryDecomposer:
    def __init__(self, model: str="gpt-4o-mini"):

        if load_dotenv() is None:
            raise ValueError("API Key Missing")
        
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = model

    def decompose(self, query: str) -> list:
        response = self.client.chat.completions.create(
            model=self.model,
            messages = [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': query}
            ],
            temperature=0.0
        )
        
        text = response.choices[0].message.content.strip()
        
        subqueries = [line.strip() for line in text.split('\n') if line.strip()]
        
        return subqueries