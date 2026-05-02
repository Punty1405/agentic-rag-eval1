import os
from langsmith import Client
from dotenv import load_dotenv
from collections import Counter

load_dotenv()

client = Client()
RUN_ID = 'eval-20260430-170415'
PROJECT_NAME = 'agentic-rag-eval'

project = client.read_project(project_name=PROJECT_NAME)

eval_runs = list(client.list_runs(
    project_id=project.id,
    filter=f'and(eq(name, "RetrievalEvaluator.evaluate"), eq(metadata_key, "eval_run"), eq(metadata_value, "{RUN_ID}"))',
    limit=100
))

print(f"Pulled {len(eval_runs)} evaluator runs")

zero_hits = []
perfect_hits = []
partial_hits = []

for run in eval_runs:
    output = run.outputs or {}
    hit_rate = output.get('hit_rate', None)

    if hit_rate is None:
        continue
    
    record = {
        'query': output.get('query', ""),
        'hit_rate': hit_rate,
        "trace_id": str(run.trace_id) if run.trace_id else None,
    }

    if hit_rate == 0:
        zero_hits.append(record)
    elif 0.3 <= hit_rate <= 0.7:
        partial_hits.append(record)
    elif hit_rate >= 0.9:
        perfect_hits.append(record)

# print(f"Total runs: {len(zero_hits) + len(perfect_hits) + len(partial_hits)}")
print(f"  Zero hits: {len(zero_hits)}")
print(f"  Perfect hits: {len(perfect_hits)}")
print(f"  Partial hits: {len(partial_hits)}")

print("\n=== 5 REASONABLE PERFORMERS (0.3 <=hit_rate <= 0.7) ===")

for r in partial_hits:
    print(f"\nHR: {r['hit_rate']}")
    print(f"Query: {r['query']}")
    print(f"Trace: https://smith.langchain.com/runs/{r['trace_id']}")