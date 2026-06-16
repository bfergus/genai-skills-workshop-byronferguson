# %% [markdown]
# # Lab 5 — ADS Agent: Vertex AI Evaluation
#
# Runs the ADS agent over a held-out Q/A dataset and scores it on coherence,
# fluency, instruction following, groundedness, and QA quality using the
# Vertex AI evaluation harness.

# %% Cell 1 — imports
import os
import time
import pandas as pd

import vertexai
from vertexai.evaluation import EvalTask

import Lab5_agent as agent

# %% Cell 2 — config
# Reuse the resolver from Lab5_agent so all files behave consistently
PROJECT_ID = agent.PROJECT_ID
LOCATION = os.getenv("LOCATION", "us-east4")
EVAL_DATASET_PATH = "Lab5_eval_dataset.csv"
EVAL_RESULTS_PATH = "Lab5_eval_results.csv"
EXPERIMENT_NAME = "lab5-ads-eval"

vertexai.init(project=PROJECT_ID, location=LOCATION)

print(f"Project    : {PROJECT_ID}")
print(f"Location   : {LOCATION}")
print(f"Dataset    : {EVAL_DATASET_PATH}")
print(f"Experiment : {EXPERIMENT_NAME}")

# %% Cell 3 — load eval dataset
df = pd.read_csv(EVAL_DATASET_PATH)
print(f"Loaded {len(df)} eval examples.")
print(df.head())

# %% Cell 4 — GCS bucket for eval artifacts
EVAL_GCS_BUCKET = f"gs://{PROJECT_ID}-lab5-eval"
print(f"Eval output bucket: {EVAL_GCS_BUCKET}")
print("(Create with: gsutil mb -l us-east4 " + EVAL_GCS_BUCKET + " — if it doesn't already exist.)")

# %% Cell 5 — run the agent over every question
responses = []
for i, row in df.iterrows():
    q = row["question"]
    print(f"  [{i+1}/{len(df)}] {q[:70]}")
    t0 = time.time()
    result = agent.answer(q)
    dt = int((time.time() - t0) * 1000)
    responses.append({
        "question": q,
        "reference_answer": row["reference_answer"],
        "agent_answer": result.get("answer", ""),
        "blocked": result.get("blocked", False),
        "sources": ",".join(result.get("sources", [])),
        "latency_ms": dt,
    })

resp_df = pd.DataFrame(responses)
print(f"Collected {len(resp_df)} agent responses.")

# %% Cell 6 — build evaluation dataframe in the schema EvalTask expects
eval_df = pd.DataFrame({
    "prompt": resp_df["question"],
    "response": resp_df["agent_answer"],
    "reference": resp_df["reference_answer"],
})
print(eval_df.head())

# %% Cell 7 — run EvalTask
metrics = [
    "coherence",
    "fluency",
    "instruction_following",
    "groundedness",
    "question_answering_quality",
]
eval_task = EvalTask(
    dataset=eval_df,
    metrics=metrics,
    experiment=EXPERIMENT_NAME,
    output_uri_prefix=EVAL_GCS_BUCKET,
)
eval_result = eval_task.evaluate()
print("EvalTask completed.")

# %% Cell 8 — summary
print("=" * 60)
print("SUMMARY METRICS")
print("=" * 60)
print(eval_result.summary_metrics)
print()
print("PER-EXAMPLE METRICS")
print("=" * 60)
print(eval_result.metrics_table)

# %% Cell 9 — save results to CSV
combined = resp_df.copy()
try:
    mt = eval_result.metrics_table.reset_index(drop=True)
    combined = pd.concat([combined.reset_index(drop=True), mt], axis=1)
except Exception as e:
    print(f"Could not merge metrics_table: {e}")

combined.to_csv(EVAL_RESULTS_PATH, index=False)
print(f"Saved {EVAL_RESULTS_PATH}")
