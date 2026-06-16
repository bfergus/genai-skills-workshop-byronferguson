# Lab 5 — Alaska Department of Snow: Architecture

## 1. Overview

The ADS Agent is a Retrieval Augmented Generation (RAG) service that answers
citizen questions about snow operations using a curated, authoritative
knowledge base. The system runs entirely on Google Cloud in the `us-east4`
region, behind two-sided Google Cloud Model Armor safety filtering.

## 2. ASCII diagram

```
                                Citizens / Public Web
                                         |
                                         | HTTPS
                                         v
                            +------------+------------+
                            |   Cloud Run (us-east4)  |
                            |   ads-agent service     |
                            |                         |
                            |   FastAPI (Lab5_app.py) |
                            |   - GET  /              |
                            |   - GET  /static/*      |
                            |   - POST /chat          |
                            |   - GET  /health        |
                            +-----------+-------------+
                                        |
                                        | answer(question)
                                        v
                            +-----------+-------------+
                            |    Lab5_agent.py        |
                            |                         |
                            |   1. sanitize_prompt --------> Model Armor
                            |   2. embed query ------------> text-embedding-004
                            |   3. search_kb --------------> BigQuery VECTOR_SEARCH
                            |   3b. needs_511_data? -------> Alaska 511 API (HTTPS)
                            |                                (events / winterroads /
                            |                                 airports / cameras / alerts)
                            |   3c. needs_weather_data? ---> NWS api.weather.gov (HTTPS)
                            |                                (current obs / 7-day forecast)
                            |   4. build_prompt + gemini --> gemini-2.5-flash
                            |   5. sanitize_response ------> Model Armor
                            |   6. log_interaction --------> BQ + Cloud Logging
                            +-------------------------+

       Source data (one-time, Lab5_ingest.py):
       gs://labs.roitraining.com/alaska-dept-of-snow
              -> pypdf / BeautifulSoup / raw text
              -> 800-char chunks (100 overlap)
              -> text-embedding-004 (768-dim, RETRIEVAL_DOCUMENT)
              -> BigQuery ads.ads_kb

       External tools (per-request, keyword-gated):
       [1] Alaska 511:  https://511.alaska.gov/api/v2/get/{endpoint}
              -> events / winterroads / airports / cameras / alerts
              -> intent-routed by question keywords
       [2] NWS:         https://api.weather.gov/{points|stations|forecast}
              -> current observation OR multi-day forecast
              -> location-routed to nearest Alaska city
       Both merge into the Gemini prompt as LIVE DATA block(s).
```

## 3. Components

| Component            | Service                | Purpose                                       |
|----------------------|------------------------|-----------------------------------------------|
| UI                   | Cloud Run (static)     | Single-page chat widget                       |
| API                  | Cloud Run (FastAPI)    | HTTP entry point                              |
| Safety               | Model Armor            | Prompt + response sanitization                |
| Embeddings           | Vertex AI              | text-embedding-004 (768 dims)                 |
| Retrieval            | BigQuery VECTOR_SEARCH | Top-K cosine similarity over ads_kb           |
| Generation           | Vertex AI Gemini       | gemini-2.5-flash, temperature 0.2             |
| Audit (structured)   | BigQuery ads_audit     | Long-term queryable log                       |
| Audit (operational)  | Cloud Logging          | Real-time JSON logs                           |
| Source content       | Cloud Storage          | Authoritative ADS documents                   |
| Live data tool       | Alaska 511 API         | Real-time road / winter / airport / camera data |
| Live data tool       | National Weather Service | Current weather observation + 7-day forecast |

## 4. Data flow

1. Citizen submits a question through the chat widget.
2. FastAPI receives `POST /chat`, invokes `Lab5_agent.answer()`.
3. **Prompt sanitization** — Model Armor evaluates the question. If any filter
   trips, the pipeline aborts with a friendly blocked message; the event is
   logged.
4. **Embedding** — the question is embedded with `task_type=RETRIEVAL_QUERY`.
5. **Vector search** — BigQuery `VECTOR_SEARCH` returns the top-K (default 5)
   chunks, ordered by distance.
6. **Live data tool (conditional)** — `needs_511_data(question)` keyword-routes
   the question to one of the Alaska 511 endpoints (`event`, `winterroads`,
   `airports`, `cameras`, `alerts`). When triggered, the agent fetches up to 10
   live items over HTTPS with an 8-second timeout. Failures and missing API key
   are logged and the agent falls back to KB-only context.
7. **Prompt assembly** — system instructions + KB context + optional LIVE DATA
   block + question. The system instruction directs Gemini to prefer live 511
   data over the static KB for relevant questions and to cite when it does.
8. **Generation** — Gemini 2.5 Flash produces the answer.
9. **Response sanitization** — Model Armor scans the model output; blocked
   responses are replaced with a generic safe message.
10. **Audit** — interaction is written to `ads.ads_audit` (BigQuery) and
    emitted as structured JSON to Cloud Logging.
11. Result is returned to the browser, which renders message + source files.

## 4a. Tools / external integrations

### Alaska 511 live-data tool

| Aspect | Detail |
|--------|--------|
| **Provider** | Alaska 511 — operated by Alaska DOT&PF |
| **Documentation** | https://511.alaska.gov/developers/doc |
| **Base URL** | `https://511.alaska.gov/api/v2/get/{endpoint}` |
| **Authentication** | API key (query string `?key=...`) — set via `ALASKA_511_API_KEY` env var on Cloud Run |
| **Rate limit** | 10 calls / 60 seconds (per the provider) |
| **Timeout** | 8 seconds (agent-side) |
| **Format** | JSON |

**Intent routing** — `needs_511_data()` picks the right endpoint based on keywords:

| Question contains | Endpoint called |
|-------------------|-----------------|
| `airport` | `airports` |
| `winter driving`, `driving conditions`, `road conditions`, `highway conditions` | `winterroads` |
| `road closure`, `highway closure`, `road closed`, `traffic`, `accident`, `roadwork`, `511` | `event` |

**Fail-safe behavior:**
- Missing API key → tool short-circuits with `alaska_511_skipped_no_key` log; agent falls back to KB-only.
- Endpoint unreachable / 4xx / 5xx → `alaska_511_fetch_failed` log; agent falls back to KB-only.
- API key never appears in logs (only `safe_url` without the `key` parameter is recorded).

**Why a tool, not another KB ingest:** road/airport status changes minute-to-minute. Re-ingesting on a schedule would always be stale. A live API call gives the citizen the most current information at the moment they ask.

### National Weather Service (NWS) live-data tool

| Aspect | Detail |
|--------|--------|
| **Provider** | National Weather Service (NOAA) |
| **Documentation** | https://www.weather.gov/documentation/services-web-api |
| **Base URL** | `https://api.weather.gov` |
| **Authentication** | None — public API. `User-Agent` header is **required** with contact info. Configured via `NWS_USER_AGENT` env var. |
| **Rate limit** | NWS asks for "reasonable" use; no published hard limit |
| **Timeout** | 8 seconds (agent-side) |
| **Format** | GeoJSON |

**Endpoints used:**

| Use case | Endpoint chain |
|----------|----------------|
| **Current weather** | `/points/{lat},{lon}` → `observationStations` → `/stations/{id}/observations/latest` |
| **Forecast (7-day)** | `/points/{lat},{lon}` → `forecast` URL (resolves to `/gridpoints/{office}/{x},{y}/forecast`) |

**Intent routing** — `needs_weather_data()` picks current vs forecast:

| Question contains | Mode |
|-------------------|------|
| "current weather", "weather right now", "current conditions", "right now" | `current` |
| "forecast", "tomorrow", "tonight", "this week", "weekend" | `forecast` |
| bare "weather", "temperature", "snowfall" | `forecast` (default — broader coverage) |

**Location routing** — `detect_alaska_location()` matches the question against a built-in table of 16 Alaska cities (Anchorage, Fairbanks, Juneau, Wasilla, Palmer, Kenai, Soldotna, Homer, Seward, Valdez, Nome, Kodiak, Sitka, Ketchikan, Utqiagvik/Barrow, Bethel). Defaults to Anchorage when no city is named.

**Fail-safe behavior:** any HTTP error, timeout, or parse failure is logged as `nws_fetch_failed` and the agent falls back to the static KB and (if applicable) Alaska 511 data only.

## 5. Security controls

- **Model Armor (two-sided)**: prompt-injection, jailbreak, PII, malicious URL,
  and RAI filters on both ingress and egress.
- **Fail-closed**: any error from Model Armor is treated as a block.
- **Grounded-only system prompt**: the model is instructed to answer only from
  the provided context and say "I don't know" otherwise.
- **No PII persistence**: session IDs are random UUIDs; Model Armor PII filters
  redact sensitive data before logging.
- **IAM isolation**: the Cloud Run service account has narrow scopes —
  BigQuery read on `ads_kb`, write on `ads_audit`, Vertex AI user, Model Armor
  user. Nothing else.
- **Private container**: image stored in Artifact Registry, scanned for CVEs.
- **Two audited external calls** outside the GCP boundary, both to
  authoritative US/Alaska government APIs:
  1. **Alaska 511** (state DOT&PF) — API key injected via Cloud Run env var,
     redacted from logs.
  2. **National Weather Service** (NOAA) — public, no API key; requires only
     a `User-Agent` header for identification.
  Both have an 8-second timeout and gracefully fall back to KB-only context on
  any failure.

## 6. Cost analysis (rough monthly, US East)

Assumptions: 50,000 citizen questions/month, average 5 retrieved chunks at ~800
chars each, average response ~400 tokens.

| Service                     | Monthly usage              | Approx. cost (USD) |
|-----------------------------|----------------------------|--------------------|
| Cloud Run (1 vCPU, 1Gi)     | ~150 hr active             | $6                 |
| Vertex AI embeddings        | 50k * 1 query embed        | <$1                |
| Vertex AI Gemini 2.5 Flash  | 50k * ~5k in + 400 out     | ~$25               |
| BigQuery storage            | <1 GB ads_kb + ads_audit   | <$1                |
| BigQuery vector search      | 50k queries on <1GB table  | ~$5                |
| Model Armor (prompt+resp)   | 100k sanitize calls        | ~$10               |
| Cloud Logging               | <50 GB structured logs     | ~$25               |
| Cloud Storage (source)      | <1 GB                      | <$1                |
| Artifact Registry           | <1 GB image                | <$1                |
| Alaska 511 API              | ~5k calls (10% of traffic) | $0 (free public API) |
| NWS API                     | ~10k calls (20% of traffic, ~3 per request) | $0 (free public API) |
| **Total**                   |                            | **~$75/month**     |

Re-ingest (one-time per content refresh): ~$1 for document embeddings.

## 7. Stakeholder objection rebuttals

### Objection 1 (Admin): "We have a cloud reservation we must use."
**Rebuttal**: Every component — Cloud Run, Vertex AI, BigQuery, Model Armor,
Cloud Storage, Cloud Logging — runs inside Google Cloud and counts against the
existing committed-use / reservation spend. No third-party SaaS, no separate
vector database, no external LLM API. The entire data plane lives in
`us-east4`, satisfying any regional reservation constraint.

### Objection 2 (CFO): "Generative AI is too expensive."
**Rebuttal**: Itemized estimate above is ~$75/month at 50k queries. That is
0.15¢ per answered question. Even at 10× traffic the cost is well under the
fully-loaded hourly cost of a single dispatcher. We chose Gemini 2.5 **Flash**
(not Pro) which is roughly 10× cheaper while sufficient for grounded QA. We
use BigQuery VECTOR_SEARCH instead of a managed vector DB (no idle cost). Cloud
Run scales to zero between traffic bursts. Re-evaluation runs monthly, not
continuously.

### Objection 3 (Operations): "AI hallucinates — citizens will get wrong info."
**Rebuttal**:
1. **Grounded RAG**: the model is constrained to answer only from the
   ADS-curated corpus. The system prompt explicitly says "If you don't know,
   say so."
2. **Authoritative live data for time-sensitive answers**: road conditions,
   closures, and airport status are fetched in real time from Alaska 511 — the
   same source the public sees on the 511 website — and passed to the LLM as
   ground truth. No risk of stale or invented information for those topics.
3. **Sources surfaced**: the UI shows which source files the answer came from,
   so citizens (and reviewers) can verify.
4. **Measured accuracy**: `Lab5_eval.py` runs the Vertex AI evaluation harness
   on a held-out test set, scoring **groundedness**, instruction following,
   coherence, fluency, and QA quality. We publish those numbers per release.
5. **Model Armor on the response**: catches any unsafe output the model
   might produce.
6. **Full audit**: every Q/A pair is logged; we can review and improve
   continuously.

## 8. Why these choices

| Choice                            | Why                                                                 |
|-----------------------------------|---------------------------------------------------------------------|
| **Vertex AI (not OpenAI/Claude)** | Stays inside GCP boundary; counts against existing reservation; data residency in us-east4. |
| **Gemini 2.5 Flash (not Pro)**    | ~10× cheaper, sub-second latency, sufficient for grounded QA.       |
| **text-embedding-004**            | Current Google production embedding model, 768 dims, well-supported by BQ VECTOR_SEARCH. |
| **BigQuery vector search**        | No separate vector DB to operate. Joins naturally with audit data. Scales to billions of rows. Avoids idle infrastructure cost. |
| **Model Armor (not bespoke filters)** | Managed, regularly updated, two-sided, supports prompt-injection + RAI + PII + malicious URLs in one service. |
| **Cloud Run (not GKE/VM)**        | Scales to zero, no infra to manage, integrates with Cloud Build and IAM. |
| **FastAPI**                       | Lightweight, async, OpenAPI for free, easy testing.                 |
| **Vanilla JS UI**                 | No build pipeline, no framework upgrade tax, served from same Cloud Run service. |
| **us-east4**                      | Matches existing department region; all services available there.   |
| **Dual audit (BQ + Cloud Logging)** | BQ for long-term queryable analytics, Cloud Logging for real-time ops dashboards / alerts. |
| **Chunk 800 / overlap 100**       | Empirically good for English procedural docs; keeps context windows tight; overlap prevents boundary loss. |
| **Top-K = 5**                     | Enough recall for procedural questions without ballooning prompt cost. |
| **Alaska 511 live tool**          | Time-sensitive data (road status, closures, airport status) changes minute-to-minute; a tool call is more accurate than periodic re-ingest. Free public API, authoritative source, keyword-gated to keep latency low. |
| **NWS live tool**                 | Weather changes by the hour. NWS is the authoritative US weather source, free, no API key, and provides both current observations and 7-day forecasts in one client. Pairs naturally with the 511 tool for snow-related citizen questions. |
