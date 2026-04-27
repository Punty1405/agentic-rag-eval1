# Query Analyzer - Agent's decision-making component
# Analyzes incoming query and selects retrieval strategy

class QueryAnalyser:
    def __init__(self):
        self.strategies = {
            'simple': 'BAAI/llm-embedder',
            'complex': 'BAAI/llm-embedder', # we'll add more embedding models later
        }

    def analyse(self, query: str) -> dict:
        """
        Analyses the incoming query complexity and returns retrieval strategy.
        Input:
        query: str - incoming query
        Returns: dict - with complexity, hops and embedder
        """

        complexity = self._assess_complexity(query)
        hops = self._estimate_hops(query)
        embedder = self.strategies[complexity]

        return {
            'query': query,
            'complexity': complexity,
            'hops': hops,
            'embedder': embedder
        }
    
    def _assess_complexity(self, query: str) -> str:
        """
        Simple heuristic - we'll make this smarter later
        """

        complex_indicators = "and both compare difference between also additionally while".split()

        query_lower = query.lower()

        if any(indicator in query_lower for indicator in complex_indicators):
            return 'complex'
        else:
            return 'simple'
        
    def _estimate_hops(self, query: str) -> int:
        """
        Estimate number of retrieval hps needed
        """

        if self._assess_complexity(query) == 'simple':
            return 1
        return 2

    