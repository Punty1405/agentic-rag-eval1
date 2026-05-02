import os
from langsmith import Client
from dotenv import load_dotenv
from pathlib import Path
import json

load_dotenv()

client = Client()
RUN_ID = 'eval-20260430-170415'
PROJECT_NAME = 'agentic-rag-eval'

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / 'output'

# Get project
project = client.read_project(project_name=PROJECT_NAME)

eval_runs = list(client.list_runs(
    project_id = project.id,
    filter=f'and(eq(name, "RetrievalEvaluator.evaluate"), eq(metadata_key, "eval_run"), eq(metadata_value, "{RUN_ID}"))',
    limit=100
))

print(f"Pulled {len(eval_runs)} evaluator runs")

# Grouping by scores
low_cr = [] # context_recall < 0.5
high_gap = [] # HR << CR

for run in eval_runs:
    output = run.outputs or {}
    hit_rate = output.get('hit_rate', 0)

    # Considering low hit_rate as proxy for failures
    if hit_rate < 0.5:
        low_cr.append({
            'query': output.get('query', ""),
            'hit_rate': hit_rate,
            'trace_id': str(run.trace_id),
            'run_id': str(run.id)
        })

print(f"Found {len(low_cr)} low-scoring queries (hit_rate < 0.5)")

# Fetch full trace tree to get analyser + decomposer outputs
print("\n" + "="*70)
print("INSPECTING 10 FAILURE TRACES")
print("\n" + "="*70)

cases_to_evaluate = low_cr[:10]
for i, case in enumerate(cases_to_evaluate, 1):
    print(f"Trace {i}: Query: {case['query']}")
    print(f"Hit Rate: {case['hit_rate']}")

    parent_runs = list(client.list_runs(
        project_id = project.id,
        trace_filter = f'eq(trace_id, "{case["trace_id"]}")',
        is_root=True
    ))

    if not parent_runs:
        print(f"    [No parent run found]")
        continue

    parent = parent_runs[0]

    # Fetching child runs (analyser, decomposer, retriever, evaluator)
    child_runs = list(client.list_runs(
        project_id = project.id,
        trace_filter = f'eq(trace_id, "{case["trace_id"]}")',
        is_root=False
    ))

    # Extract analyser and decomposer outputs
    for child in child_runs:
        if child.name == 'QueryAnalyser.analyse':
            print(f"\n Analyser Output:")
            case['Analyser_Output'] = child.outputs
            
        elif child.name == 'QueryDecomposer.decompose':
            print(f"\n Decomposer Output:")
            case['Decomposer_Output'] = child.outputs

trace_inpsector_dump = OUTPUT_DIR / f"trace_inpsector_dump.json"

OUTPUT_DIR.mkdir(exist_ok=True)
with open(trace_inpsector_dump, 'w') as f:
    json.dump(cases_to_evaluate, f, indent=2)