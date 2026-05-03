"""Utility functions - copied from reference repo for CI compatibility"""

import json
from typing import List
from llama_index.schema import Document


class JSONReader:
    """JSON reader for corpus data."""
    
    def __init__(self, is_jsonl: bool = False) -> None:
        self.is_jsonl = is_jsonl
    
    def load_data(self, input_file: str) -> List[Document]:
        """Load data from the input file."""
        documents = []
        with open(input_file, 'r') as file:
            load_data = json.load(file)
        
        for data in load_data:
            metadata = {
                "title": data['title'],
                "published_at": data['published_at'],
                "source": data['source']
            }
            documents.append(Document(text=data['body'], metadata=metadata))
        
        return documents