# %% [markdown]
# # Lab 5 — Alaska Department of Snow: Ingest notebook
#
# Downloads source documents from `gs://labs.roitraining.com/alaska-dept-of-snow`,
# extracts text (PDF / HTML / TXT), chunks it, embeds with
# `text-embedding-004`, and loads BigQuery `ads.ads_kb`. Also creates the
# `ads.ads_audit` table used by the agent.
#
# Idempotent: re-running truncates and reloads `ads_kb`.

# %% Cell 1 — pip install
# !pip install --quiet google-cloud-storage google-cloud-bigquery google-cloud-aiplatform pypdf beautifulsoup4 pandas

# %% Cell 2 — imports
import os
import io
import re
import json
import time
import warnings
from pathlib import Path
from typing import List, Dict

from google.cloud import storage
from google.cloud import bigquery
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput

from pypdf import PdfReader
from bs4 import BeautifulSoup

# %% Cell 3 — config
def _resolve_project_id() -> str:
    explicit = os.getenv("PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
    if explicit:
        return explicit
    import google.auth
    _, project = google.auth.default()
    if not project:
        raise RuntimeError("Could not determine GCP project ID — set PROJECT_ID or run gcloud auth application-default login.")
    return project


PROJECT_ID = _resolve_project_id()
LOCATION = os.getenv("LOCATION", "us-east4")
BQ_DATASET = "ads"
BQ_KB_TABLE = "ads_kb"
BQ_AUDIT_TABLE = "ads_audit"
EMBED_MODEL = "text-embedding-004"

SOURCE_BUCKET = "labs.roitraining.com"
SOURCE_PREFIX = "alaska-dept-of-snow/"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBED_BATCH = 100

LOCAL_DOWNLOAD_DIR = Path("/tmp/ads_source")
LOCAL_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

print(f"Project:  {PROJECT_ID}")
print(f"Location: {LOCATION}")
print(f"Source:   gs://{SOURCE_BUCKET}/{SOURCE_PREFIX}")
print(f"Target:   {PROJECT_ID}.{BQ_DATASET}.{BQ_KB_TABLE}")

# %% Cell 4 — init clients
vertexai.init(project=PROJECT_ID, location=LOCATION)
storage_client = storage.Client(project=PROJECT_ID)
bq_client = bigquery.Client(project=PROJECT_ID, location=LOCATION)
embed_model = TextEmbeddingModel.from_pretrained(EMBED_MODEL)
print("Clients initialized.")

# %% Cell 5 — list and download all source objects
print(f"Listing gs://{SOURCE_BUCKET}/{SOURCE_PREFIX} ...")
bucket = storage_client.bucket(SOURCE_BUCKET)
blobs = list(storage_client.list_blobs(bucket, prefix=SOURCE_PREFIX))
# Filter out "directory" placeholder blobs
blobs = [b for b in blobs if not b.name.endswith("/")]
print(f"Found {len(blobs)} objects.")

local_files: List[Path] = []
for blob in blobs:
    safe_name = blob.name.replace("/", "_")
    local_path = LOCAL_DOWNLOAD_DIR / safe_name
    blob.download_to_filename(str(local_path))
    local_files.append(local_path)
    print(f"  downloaded {blob.name} -> {local_path}")
print(f"Downloaded {len(local_files)} files.")

# %% Cell 6 — extract text per file type
def extract_text(path: Path) -> str:
    """Extract plain text from PDF, HTML, or TXT. Returns "" for unsupported."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            reader = PdfReader(str(path))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        elif suffix in (".html", ".htm"):
            html = path.read_text(encoding="utf-8", errors="ignore")
            soup = BeautifulSoup(html, "html.parser")
            # Drop script/style noise
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            return soup.get_text(separator="\n")
        elif suffix == ".txt":
            return path.read_text(encoding="utf-8", errors="ignore")
        else:
            warnings.warn(f"Skipping unsupported file type: {path.name}")
            return ""
    except Exception as e:
        warnings.warn(f"Failed to extract {path.name}: {e}")
        return ""


extracted: Dict[str, str] = {}
for f in local_files:
    text = extract_text(f)
    if text.strip():
        # Use the original GCS name (recover by stripping our prefix replacement)
        original_name = f.name.replace("_", "/", 1) if "_" in f.name else f.name
        # The "_" -> "/" replace is only for the leading prefix; simpler: use file name as-is.
        extracted[f.name] = text
        print(f"  extracted {f.name}: {len(text)} chars")
    else:
        print(f"  skipped {f.name} (no text)")

print(f"Extracted text from {len(extracted)} files.")

# %% Cell 7 — chunk into 800-char overlapping chunks
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Sliding-window character chunker. Collapses excessive whitespace first."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    chunks = []
    start = 0
    step = size - overlap
    while start < len(cleaned):
        chunks.append(cleaned[start:start + size])
        start += step
    return chunks


all_chunks: List[Dict] = []
for source_file, text in extracted.items():
    pieces = chunk_text(text)
    for i, piece in enumerate(pieces):
        all_chunks.append({
            "source_file": source_file,
            "chunk_index": i,
            "content": piece,
        })
print(f"Produced {len(all_chunks)} chunks from {len(extracted)} files.")

# %% Cell 8 — create dataset + tables
dataset_ref = bigquery.Dataset(f"{PROJECT_ID}.{BQ_DATASET}")
dataset_ref.location = LOCATION
bq_client.create_dataset(dataset_ref, exists_ok=True)
print(f"Dataset {BQ_DATASET} ready.")

kb_schema = [
    bigquery.SchemaField("source_file", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("chunk_index", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("content", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
]
kb_table = bigquery.Table(f"{PROJECT_ID}.{BQ_DATASET}.{BQ_KB_TABLE}", schema=kb_schema)
bq_client.create_table(kb_table, exists_ok=True)
print(f"Table {BQ_KB_TABLE} ready.")

audit_schema = [
    bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("session_id", "STRING"),
    bigquery.SchemaField("user_prompt", "STRING"),
    bigquery.SchemaField("prompt_blocked", "BOOL"),
    bigquery.SchemaField("prompt_filters", "STRING"),
    bigquery.SchemaField("retrieved_chunks", "STRING"),
    bigquery.SchemaField("llm_response", "STRING"),
    bigquery.SchemaField("response_blocked", "BOOL"),
    bigquery.SchemaField("response_filters", "STRING"),
    bigquery.SchemaField("latency_ms", "INT64"),
]
audit_table = bigquery.Table(f"{PROJECT_ID}.{BQ_DATASET}.{BQ_AUDIT_TABLE}", schema=audit_schema)
bq_client.create_table(audit_table, exists_ok=True)
print(f"Table {BQ_AUDIT_TABLE} ready.")

# %% Cell 9 — embed in batches and load with WRITE_TRUNCATE
print(f"Embedding {len(all_chunks)} chunks in batches of {EMBED_BATCH} ...")
rows_to_load = []
for i in range(0, len(all_chunks), EMBED_BATCH):
    batch = all_chunks[i:i + EMBED_BATCH]
    inputs = [TextEmbeddingInput(text=c["content"], task_type="RETRIEVAL_DOCUMENT") for c in batch]
    embeddings = embed_model.get_embeddings(inputs)
    for chunk, emb in zip(batch, embeddings):
        rows_to_load.append({
            "source_file": chunk["source_file"],
            "chunk_index": chunk["chunk_index"],
            "content": chunk["content"],
            "embedding": list(emb.values),
        })
    print(f"  embedded {min(i + EMBED_BATCH, len(all_chunks))}/{len(all_chunks)}")

job_config = bigquery.LoadJobConfig(
    schema=kb_schema,
    write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
)
load_job = bq_client.load_table_from_json(
    rows_to_load,
    f"{PROJECT_ID}.{BQ_DATASET}.{BQ_KB_TABLE}",
    job_config=job_config,
)
load_job.result()
print(f"Loaded {len(rows_to_load)} rows into {BQ_KB_TABLE}.")

# %% Cell 10 — summary
table = bq_client.get_table(f"{PROJECT_ID}.{BQ_DATASET}.{BQ_KB_TABLE}")
print("=" * 60)
print("INGEST SUMMARY")
print("=" * 60)
print(f"Files processed   : {len(extracted)}")
print(f"Chunks produced   : {len(all_chunks)}")
print(f"Rows in {BQ_KB_TABLE}: {table.num_rows}")
print()
sample_q = f"""
SELECT source_file, chunk_index, SUBSTR(content, 1, 120) AS preview,
       ARRAY_LENGTH(embedding) AS dims
FROM `{PROJECT_ID}.{BQ_DATASET}.{BQ_KB_TABLE}`
LIMIT 1
"""
for row in bq_client.query(sample_q).result():
    print("Sample row:")
    print(f"  source_file : {row.source_file}")
    print(f"  chunk_index : {row.chunk_index}")
    print(f"  preview     : {row.preview}")
    print(f"  embed dims  : {row.dims}")
print("=" * 60)
print("Ingest complete.")
