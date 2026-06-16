# Lab 5 — Alaska Department of Snow (ADS) Generative AI Agent

A production-quality RAG (Retrieval Augmented Generation) chat agent for the
fictional Alaska Department of Snow case study. Built on Google Cloud using
Vertex AI (Gemini 2.5 Flash + text-embedding-004), BigQuery vector search,
Cloud Run, and Google Cloud Model Armor for prompt/response safety.

---

## Case study summary

The Alaska Department of Snow (ADS) provides citizens, schools, and businesses
with information about snow plowing schedules, road conditions, school closures,
snow emergency procedures, and related services. The department fields tens of
thousands of phone calls and emails each winter season.

Leadership wants to deploy a generative AI assistant on the public website to
deflect routine inquiries, freeing dispatchers to focus on emergencies. The
solution must:

1. Answer only from approved, authoritative source content (no hallucination).
2. Block adversarial prompts and unsafe responses.
3. Be auditable — every interaction logged.
4. Run inside Google Cloud (existing department contract).
5. Stay cost-controlled (CFO objection).

---

## Architecture (ASCII)

```
                        +-------------------------+
                        |  Citizen Browser        |
                        |  (static/index.html JS) |
                        +-----------+-------------+
                                    | HTTPS POST /chat
                                    v
                  +-----------------+----------------+
                  |   Cloud Run (us-east4)           |
                  |   FastAPI (Lab5_app.py)          |
                  |   + static UI                    |
                  +-----------------+----------------+
                                    |
                                    v
                  +-----------------+----------------+
                  |   Lab5_agent.answer() pipeline   |
                  +---+----------+----------+--------+
                      |          |          |
                      v          v          v
              +-------+--+  +----+-----+  +-+--------+
              |  Model   |  | Vertex   |  | BigQuery |
              |  Armor   |  | AI       |  | ads_kb   |
              |  prompt+ |  | embed +  |  | VECTOR_  |
              |  response|  | Gemini   |  | SEARCH   |
              +----------+  +----------+  +----------+
                      |
                      v
              +-------+----------+   +------------------+
              | BigQuery         |   | Cloud Logging    |
              | ads_audit table  |   | structured JSON  |
              +------------------+   +------------------+
```

Source content lives in `gs://labs.roitraining.com/alaska-dept-of-snow` and is
ingested by `Lab5_ingest.py` into the BigQuery `ads.ads_kb` table.

**External tools (live data merged into the LLM prompt):**
- **Alaska 511** — road conditions, closures, airport status (api key required)
- **National Weather Service** — current weather + 7-day forecasts (no key, free)

See **Tools** below.

---

## Setup steps (in order)

1. **Enable required APIs** in your GCP project:

   ```bash
   gcloud services enable \
     aiplatform.googleapis.com \
     bigquery.googleapis.com \
     storage.googleapis.com \
     modelarmor.googleapis.com \
     run.googleapis.com \
     cloudbuild.googleapis.com \
     logging.googleapis.com
   ```

2. **Create Model Armor templates** (in `us-east4`):

   - `ads-prompt-template` — enable prompt injection / jailbreak detection,
     PII detection, malicious URLs, RAI filters (HARASSMENT, HATE_SPEECH,
     SEXUALLY_EXPLICIT, DANGEROUS at MEDIUM_AND_ABOVE).
   - `ads-response-template` — enable RAI filters, PII redaction, and
     malicious URL detection.

3. **Set environment variables** — see [Environment variables](#environment-variables) below for the full reference. Minimal:

   ```bash
   export PROJECT_ID="your-gcp-project-id"
   ```

   Recommended (enables both live-data tools):

   ```bash
   export PROJECT_ID="your-gcp-project-id"
   export LOCATION="us-east4"
   export ALASKA_511_API_KEY="your-511-key"
   export NWS_USER_AGENT="ads-agent (contact: youremail@example.gov)"
   ```

4. **Install dependencies:**

   ```bash
   pip install -r Lab5_requirements.txt
   ```

5. **Run ingest** (one-time, re-runnable):

   ```bash
   jupyter nbconvert --to notebook --execute Lab5_ingest.py
   # or simply open Lab5_ingest.py in VS Code / Jupyter and run all cells
   ```

   This downloads source files, chunks them, embeds them, and loads
   `ads.ads_kb` and creates `ads.ads_audit`.

6. **Run evaluation:**

   ```bash
   python Lab5_eval.py
   ```

   Produces `Lab5_eval_results.csv` and uploads detailed metrics to the
   Vertex AI Experiments console.

7. **Deploy to Cloud Run:**

   ```bash
   bash Lab5_deploy.sh
   ```

---

## Running locally

```bash
export PROJECT_ID="your-gcp-project-id"
export LOCATION="us-east4"
python Lab5_app.py
# open http://localhost:8080
```

---

## Deploying

`Lab5_setup.sh` is the all-in-one orchestrator. It runs three phases in order — any of which can be skipped — plus you can run them individually via `Lab5_deploy.sh`. All phases are idempotent, so re-running is safe.

### The three phases

| Phase | Flag to skip       | What it does                                                       | When to skip it |
|-------|--------------------|--------------------------------------------------------------------|------------------|
| 1. Prep   | `--skip-prep`   | Enables APIs, grants IAM roles to the runtime SA, creates Model Armor templates, creates BQ dataset | After the first successful prep, or when re-deploying agent-only code changes |
| 2. Load   | `--skip-load`   | Installs ingest deps, runs `Lab5_ingest.py` (GCS → chunks → embeddings → BQ) | The KB hasn't changed |
| 3. Deploy | `--skip-deploy` | Calls `Lab5_deploy.sh` — builds container, pushes to Artifact Registry, deploys to Cloud Run | You only want to refresh data without redeploying |

There is also a convenience alias:

| Flag | Equivalent to |
|------|---------------|
| `--prep-only` | `--skip-load --skip-deploy` |

### Common command patterns

**First-time full setup** (prep + ingest + deploy):

```bash
export PROJECT_ID="qwiklabs-gcp-00-16d0362ac1ac"
export LOCATION="us-east4"
export ALASKA_511_API_KEY="your-511-key"
export NWS_USER_AGENT="ads-agent (contact: you@example.gov)"
bash Lab5_setup.sh
```

**Code change only** — agent logic, prompts, or UI tweaks — skip prep + load:

```bash
bash Lab5_setup.sh --skip-prep --skip-load
```

This is the fastest iteration loop, ~3-5 min per redeploy.

**Refresh the knowledge base** without redeploying:

```bash
bash Lab5_setup.sh --skip-prep --skip-deploy
```

Re-runs `Lab5_ingest.py` which uses `WRITE_TRUNCATE`, so the BQ table is replaced atomically. New content shows up in answers immediately — Cloud Run reads BQ live, no restart needed.

**Just verify prep without doing anything else:**

```bash
bash Lab5_setup.sh --prep-only
```

Useful for confirming APIs/IAM/templates without touching data or services.

**Deploy a specific code change to Cloud Run only** (bypass the orchestrator):

```bash
bash Lab5_deploy.sh
```

Same as `Lab5_setup.sh --skip-prep --skip-load`, but skips the orchestrator's setup checks. Slightly faster, useful when you know the environment is already configured.

### Watching the deploy

```bash
# See expanded commands as they run
bash -x Lab5_setup.sh --skip-prep --skip-load

# Tail Cloud Run logs while the new revision rolls out
gcloud run services logs tail ads-agent --region us-east4
```

### After a successful deploy

The deploy step prints the service URL like this:

```
==========================================
 Deployed: https://ads-agent-305418526864.us-east4.run.app
==========================================
```

You can also retrieve it on demand:

```bash
gcloud run services describe ads-agent \
  --region us-east4 \
  --format='value(status.url)'
```

### Rolling back

Cloud Run keeps every revision. If the new deploy misbehaves:

```bash
# List recent revisions
gcloud run revisions list --service ads-agent --region us-east4

# Roll traffic back to a previous revision
gcloud run services update-traffic ads-agent \
  --region us-east4 \
  --to-revisions ads-agent-00007-abc=100
```

---

## Running tests

```bash
pytest Lab5_tests.py -v                  # unit tests (all mocked)
pytest Lab5_tests.py -v -m live          # also hit real GCP APIs
```

---

## Environment variables

### Required to run the agent

| Variable | Default if unset | Purpose |
|----------|------------------|---------|
| `PROJECT_ID` | auto-resolved via `google.auth.default()` / `gcloud config` / `GOOGLE_CLOUD_PROJECT` | GCP project for BQ, Vertex AI, Model Armor |

`PROJECT_ID` is the only must-set variable. Everything else has a sensible default.

### Recommended for full feature set

| Variable | Default | Purpose |
|----------|---------|---------|
| `LOCATION` | `us-east4` | GCP region for all services |
| `ALASKA_511_API_KEY` | _(empty — tool disabled)_ | Enables the Alaska 511 live-data tool. Register at https://511.alaska.gov/developers/account |
| `NWS_USER_AGENT` | `ads-agent (contact: ads@alaska.example.gov)` | NWS requires a descriptive User-Agent with contact info |

### Tunable (rarely changed)

| Variable | Default | Purpose |
|----------|---------|---------|
| `BQ_DATASET` | `ads` | BigQuery dataset name |
| `BQ_KB_TABLE` | `ads_kb` | Knowledge base table |
| `BQ_AUDIT_TABLE` | `ads_audit` | Audit log table |
| `EMBED_MODEL` | `text-embedding-004` | Embedding model |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Generation model |
| `PROMPT_ARMOR_TEMPLATE` | `ads-prompt-template` | Model Armor prompt template ID |
| `RESPONSE_ARMOR_TEMPLATE` | `ads-response-template` | Model Armor response template ID |
| `SERVICE_NAME` | `ads-agent` | Cloud Run service name (deploy only) |
| `REGION` | `us-east4` (mirrors `LOCATION`) | Cloud Run region (deploy.sh) |
| `PROMPT_TEMPLATE_ID` | `ads-prompt-template` | Setup script — Model Armor template creation |
| `RESPONSE_TEMPLATE_ID` | `ads-response-template` | Setup script — Model Armor template creation |

### Copy-paste blocks

**Minimal (just project):**

```bash
export PROJECT_ID="qwiklabs-gcp-00-16d0362ac1ac"
```

**Full feature set:**

```bash
export PROJECT_ID="qwiklabs-gcp-00-16d0362ac1ac"
export LOCATION="us-east4"
export ALASKA_511_API_KEY="your-511-key-here"
export NWS_USER_AGENT="ads-agent (contact: your.email@example.gov)"
```

**Full + tunable overrides** (only if you want custom names):

```bash
export PROJECT_ID="qwiklabs-gcp-00-16d0362ac1ac"
export LOCATION="us-east4"
export REGION="us-east4"
export SERVICE_NAME="ads-agent"
export BQ_DATASET="ads"
export BQ_KB_TABLE="ads_kb"
export BQ_AUDIT_TABLE="ads_audit"
export EMBED_MODEL="text-embedding-004"
export GEMINI_MODEL="gemini-2.5-flash"
export PROMPT_ARMOR_TEMPLATE="ads-prompt-template"
export RESPONSE_ARMOR_TEMPLATE="ads-response-template"
export ALASKA_511_API_KEY="your-511-key-here"
export NWS_USER_AGENT="ads-agent (contact: your.email@example.gov)"
```

### Verify everything is propagated

Variables set in your shell aren't seen by `bash` subprocesses unless they are **exported**. Verify before running the setup script:

```bash
bash -c 'env | grep -E "^(PROJECT_ID|LOCATION|REGION|ALASKA_511_API_KEY|NWS_USER_AGENT|GEMINI_MODEL|BQ_)"'
```

If any expected variable is missing from the output, re-`export` it in the current shell.

---

## Debugging

### See every command the setup/deploy script runs

`Lab5_setup.sh` and `Lab5_deploy.sh` execute as bash subprocesses, so the variables and substituted values aren't visible by default. Use `bash -x` to print every command after variable expansion, before it executes:

```bash
bash -x Lab5_setup.sh
bash -x Lab5_setup.sh --skip-prep --skip-load     # also works with flags
bash -x Lab5_deploy.sh
```

Output looks like:

```
+ PROJECT_ID=qwiklabs-gcp-00-16d0362ac1ac
+ LOCATION=us-east4
+ gcloud services enable aiplatform.googleapis.com bigquery.googleapis.com ...
+ '[' false = false ']'
+ banner 'PHASE 4: Deploying to Cloud Run'
```

Every line prefixed with `+` is a command about to run, with all variables already substituted. This is the easiest way to see exactly what the script is doing, what env vars resolved to, and where it's failing.

### Run only a specific phase

```bash
bash Lab5_setup.sh --skip-load --skip-deploy     # prep only
bash Lab5_setup.sh --skip-prep --skip-deploy     # load only
bash Lab5_setup.sh --skip-prep --skip-load       # deploy only
bash Lab5_setup.sh --prep-only                   # alias for the first
```

### Inspect runtime behavior in Cloud Run

```bash
# Last 50 log entries
gcloud run services logs read ads-agent --region us-east4 --limit 50

# Tail logs live while you click in the chat UI
gcloud run services logs tail ads-agent --region us-east4

# Confirm which live-data tools are active (Alaska 511, NWS)
gcloud run services logs read ads-agent --region us-east4 \
  | grep agent_startup_tools

# See the audit trail of every interaction
bq query --use_legacy_sql=false \
  "SELECT timestamp, user_prompt, prompt_blocked, prompt_filters,
          response_blocked, response_filters
   FROM \`${PROJECT_ID}.ads.ads_audit\`
   ORDER BY timestamp DESC LIMIT 20"
```

### Test Model Armor templates directly (bypass the agent)

If the agent is blocking everything or letting everything through, query Model Armor directly to see what it's actually returning. Useful for diagnosing parser bugs vs. over-aggressive filter settings:

```bash
python3 <<'PYEOF'
from google.cloud import modelarmor_v1
from google.cloud.modelarmor_v1 import ModelArmorClient, SanitizeUserPromptRequest
import os
PROJECT_ID = os.environ["PROJECT_ID"]
client = ModelArmorClient(client_options={"api_endpoint": "modelarmor.us-east4.rep.googleapis.com"})
req = SanitizeUserPromptRequest(
    name=f"projects/{PROJECT_ID}/locations/us-east4/templates/ads-prompt-template",
    user_prompt_data=modelarmor_v1.DataItem(text="what are road conditions"),
)
print(client.sanitize_user_prompt(request=req))
PYEOF
```

### Run unit tests with verbose output

```bash
pytest Lab5_tests.py -v -s          # -s shows print() output
pytest Lab5_tests.py -v -m live     # also hit real GCP APIs
pytest Lab5_tests.py -v -k armor    # run only tests matching "armor"
```

---

## Tools

The agent ships with two external tools, each keyword-gated and short-circuited
on failure. Multiple tools can fire on the same question — e.g. *"What's the
weather and are the roads open in Fairbanks?"* triggers both.

### Alaska 511

| Endpoint                   | When triggered                                                 |
|----------------------------|----------------------------------------------------------------|
| `/api/v2/get/event`        | "road closure", "highway closure", "traffic", "accident", "roadwork", "511" |
| `/api/v2/get/winterroads`  | "winter driving", "driving conditions", "road conditions", "highway conditions" |
| `/api/v2/get/airports`     | "airport"                                                      |

Auth: API key (query string). Rate limit: 10 calls / 60 s. Timeout: 8 s.
Source: https://511.alaska.gov/developers/doc

### National Weather Service (NWS)

| Endpoint chain                                                | Mode      |
|--------------------------------------------------------------|-----------|
| `/points/{lat},{lon}` → station → `/observations/latest`     | `current` |
| `/points/{lat},{lon}` → `forecast` URL                        | `forecast`|

Intent detection:
- **Current** triggers: "current weather", "right now", "current conditions"
- **Forecast** triggers: "forecast", "tomorrow", "tonight", "this week", "weekend"
- Bare "weather" / "temperature" / "snowfall" defaults to `forecast`

Location detection: 16 Alaska cities are matched against the question
(Anchorage, Fairbanks, Juneau, Wasilla, Palmer, Kenai, Soldotna, Homer, Seward,
Valdez, Nome, Kodiak, Sitka, Ketchikan, Utqiagvik/Barrow, Bethel). Defaults to
Anchorage when no city is named.

Auth: none — public API. Requires `User-Agent` header (set via
`NWS_USER_AGENT` env var). Timeout: 8 s.
Source: https://www.weather.gov/documentation/services-web-api

---

## Presentation talking points

### Security & privacy
- **Two-sided Model Armor**: every prompt is scanned BEFORE it hits Gemini,
  and every response is scanned BEFORE it reaches the citizen. Prompt-injection,
  PII, jailbreaks, and unsafe content are all blocked at the edge.
- **Fail-closed design**: if Model Armor returns an error, the request is
  treated as blocked. Safety degradation never opens the door.
- **Grounded answers only**: the system prompt explicitly forbids answering
  outside the retrieved context. No "world knowledge" leakage.
- **Audit trail**: every interaction (prompt, retrieved chunks, response,
  filter verdicts, latency) is written to both BigQuery (`ads_audit`) and
  Cloud Logging.

### Privacy
- No personal data is persisted beyond the audit log, which is access-controlled
  via IAM. Session IDs are random UUIDs — no user accounts required.
- PII detection in Model Armor redacts sensitive content from logs.

### Cost
- Gemini 2.5 Flash is roughly 10× cheaper than Pro, sufficient for grounded QA.
- BigQuery vector search avoids a separate vector database (no idle cost).
- Cloud Run scales to zero — pay only for active citizen traffic.
- See `Lab5_architecture.md` for itemized monthly estimate.

### Accuracy
- Retrieval is over an authoritative ADS-curated corpus, not the open web.
- **Time-sensitive answers use live data**: road conditions, closures, and
  airport status come from Alaska 511 (the same source the public website uses)
  at query time — no staleness risk.
- Vertex AI evaluation harness (`Lab5_eval.py`) measures coherence, fluency,
  instruction following, **groundedness**, and QA quality on a held-out test
  set. Numbers are reproducible and tracked across releases.
- Temperature 0.2 keeps generation conservative.

---

## File index

| File                       | Purpose                                            |
|----------------------------|----------------------------------------------------|
| Lab5_README.md             | This document                                      |
| Lab5_architecture.md       | Detailed architecture + cost + objection rebuttals |
| Lab5_ingest.py             | One-time ingest notebook (GCS -> BQ embeddings)    |
| Lab5_agent.py              | Core agent module (importable)                     |
| Lab5_app.py                | FastAPI Cloud Run service                          |
| Lab5_static/index.html     | Chat UI                                            |
| Lab5_tests.py              | Pytest suite                                       |
| Lab5_eval.py               | Vertex AI EvalTask harness                         |
| Lab5_eval_dataset.csv      | Eval Q/A pairs                                     |
| Lab5_Dockerfile            | Container build                                    |
| Lab5_requirements.txt      | Python deps                                        |
| Lab5_deploy.sh             | Cloud Run deploy script                            |
