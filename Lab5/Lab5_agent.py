"""
Lab5_agent — Alaska Department of Snow RAG agent.

This module is the core of the ADS chat assistant. It exposes a single public
function, ``answer(question, session_id=None)``, that runs the full pipeline:

    1. Sanitize the user prompt with Google Cloud Model Armor.
    2. Embed the prompt and search the BigQuery knowledge base via
       ``VECTOR_SEARCH`` for the most relevant chunks.
    3. Build a grounded prompt and call Gemini 2.5 Flash.
    4. Sanitize the model response with Model Armor.
    5. Write a structured audit record to BigQuery ``ads_audit`` and to
       Cloud Logging.

Safety posture is **fail-closed**: any Model Armor failure is treated as a
block. Audit logging failures are tolerated (logged + swallowed) so a
transient BigQuery hiccup never breaks user traffic.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _resolve_project_id() -> str:
    """
    Resolve the GCP project ID at runtime, in this order:
      1. PROJECT_ID env var (explicit override)
      2. GOOGLE_CLOUD_PROJECT env var (set automatically on Cloud Run / GCE)
      3. google.auth.default() — ADC / metadata server lookup
    """
    explicit = os.getenv("PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
    if explicit:
        return explicit
    import google.auth
    _, project = google.auth.default()
    if not project:
        raise RuntimeError(
            "Could not determine GCP project ID. "
            "Set PROJECT_ID or GOOGLE_CLOUD_PROJECT, or run "
            "`gcloud auth application-default login` with a project set."
        )
    return project


PROJECT_ID = _resolve_project_id()
LOCATION = os.getenv("LOCATION", "us-east4")
BQ_DATASET = os.getenv("BQ_DATASET", "ads")
BQ_KB_TABLE = os.getenv("BQ_KB_TABLE", "ads_kb")
BQ_AUDIT_TABLE = os.getenv("BQ_AUDIT_TABLE", "ads_audit")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-004")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
PROMPT_ARMOR_TEMPLATE = os.getenv("PROMPT_ARMOR_TEMPLATE", "ads-prompt-template")
RESPONSE_ARMOR_TEMPLATE = os.getenv("RESPONSE_ARMOR_TEMPLATE", "ads-response-template")

_BASE_SYSTEM_INSTRUCTION = (
    "You are the Alaska Department of Snow information assistant. "
    "Answer the citizen's question using the provided context. "
    "If a 'LIVE DATA' block is present and contains information relevant to "
    "the question, you MUST use it to answer — do not say 'I don't know' "
    "when live data is available. "
)

_NWS_TOOL_DESCRIPTION = (
    "  - National Weather Service (api.weather.gov): current weather & forecasts\n"
)

_511_TOOL_DESCRIPTION = (
    "  - Alaska 511 (511.alaska.gov): road conditions, closures, airport status\n"
)

_CLOSING_INSTRUCTION = (
    "When summarizing live road or weather data, present each item as a "
    "separate bullet on its own line. For each road, include the condition, "
    "any restrictions, and the last-updated time when present. "
    "If neither the context nor the live data answers the question, say you "
    "don't know and advise the citizen to contact the Alaska Department of "
    "Snow directly. Be concise, accurate, and friendly. Do not invent information."
)


def build_system_instruction() -> str:
    """Compose a SYSTEM_INSTRUCTION that advertises only the tools currently
    enabled. This keeps the LLM honest — it can't 'expect' 511 data the agent
    will never provide, and won't cite a source it didn't actually use."""
    tool_lines = []
    # NWS is always available (no API key required)
    tool_lines.append(_NWS_TOOL_DESCRIPTION)
    if is_511_enabled():
        tool_lines.insert(0, _511_TOOL_DESCRIPTION)   # list 511 first when present

    live_section = (
        "If a 'LIVE DATA' block is present it contains real-time data fetched "
        "at request time — prefer it over the static knowledge base for the "
        "topics it covers. Live data sources:\n"
        + "".join(tool_lines)
        + "Cite which live source you used when you reference it. "
    )
    return _BASE_SYSTEM_INSTRUCTION + live_section + _CLOSING_INSTRUCTION


# NOTE: build_system_instruction() depends on is_511_enabled() which is
# defined further down the file alongside the 511 tool. The function is only
# called per-request from build_prompt(), so we do NOT materialize a
# module-level SYSTEM_INSTRUCTION constant here — it would forward-reference
# is_511_enabled() and fail at import time.

# ---------------------------------------------------------------------------
# Structured JSON logger
# ---------------------------------------------------------------------------

class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        # Merge any structured "extra" fields
        if hasattr(record, "structured"):
            payload.update(record.structured)  # type: ignore[arg-type]
        return json.dumps(payload, default=str)


logger = logging.getLogger("lab5_agent")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(_JsonFormatter())
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

# ---------------------------------------------------------------------------
# Client initialization (cached)
# ---------------------------------------------------------------------------

_clients: Optional[Dict[str, Any]] = None


def init_clients() -> Dict[str, Any]:
    """Initialize and cache GCP clients. Safe to call repeatedly."""
    global _clients
    if _clients is not None:
        return _clients

    import vertexai
    from google.cloud import bigquery
    from vertexai.language_models import TextEmbeddingModel
    from vertexai.generative_models import GenerativeModel

    # Model Armor — regional endpoint
    from google.cloud import modelarmor_v1
    from google.api_core.client_options import ClientOptions

    vertexai.init(project=PROJECT_ID, location=LOCATION)

    bq_client = bigquery.Client(project=PROJECT_ID, location=LOCATION)
    embed_model = TextEmbeddingModel.from_pretrained(EMBED_MODEL)
    gen_model = GenerativeModel(GEMINI_MODEL)

    # Model Armor requires a regional endpoint, not the default global one.
    armor_endpoint = f"modelarmor.{LOCATION}.rep.googleapis.com"
    armor_client = modelarmor_v1.ModelArmorClient(
        client_options=ClientOptions(api_endpoint=armor_endpoint)
    )

    _clients = {
        "bq": bq_client,
        "embed": embed_model,
        "gen": gen_model,
        "armor": armor_client,
        "armor_module": modelarmor_v1,
    }

    # One-time startup log of which live-data tools are active
    logger.info(
        "agent_startup_tools",
        extra={"structured": {
            "alaska_511_enabled": is_511_enabled(),
            "nws_enabled": True,   # NWS has no key requirement
            "project": PROJECT_ID,
            "location": LOCATION,
        }},
    )

    return _clients


# ---------------------------------------------------------------------------
# Model Armor helpers
# ---------------------------------------------------------------------------

def _armor_template_path(template_id: str) -> str:
    return f"projects/{PROJECT_ID}/locations/{LOCATION}/templates/{template_id}"


def _flagged_filters_from_result(result: Any) -> List[str]:
    """
    Walk a Model Armor SanitizationResult and return the names of filters
    whose match state indicates a positive detection. Model Armor enum values
    can be inconsistent across versions (string vs int), so we compare on
    name as well as on the canonical MATCH_FOUND value.
    """
    flagged: List[str] = []
    try:
        results_map = getattr(result, "filter_results", None) or getattr(result, "sanitization_filter_results", None)
        if results_map is None:
            return flagged
        # filter_results is typically a MapField filter_name -> FilterResult
        items = results_map.items() if hasattr(results_map, "items") else results_map
        for name, fr in items:
            match_state = getattr(fr, "match_state", None)
            # match_state may be enum, int, or have .name attribute
            state_name = getattr(match_state, "name", str(match_state))
            # Exact match — substring check would catch NO_MATCH_FOUND too
            if str(state_name) == "MATCH_FOUND" or state_name == 1 or state_name == "1":
                flagged.append(str(name))
    except Exception as e:  # never let parsing kill the request
        logger.warning(f"Could not parse Model Armor result: {e}")
    return flagged


def sanitize_prompt(text: str) -> Tuple[bool, List[str]]:
    """Returns (passed, flagged_filters). Fail-closed: errors => blocked."""
    try:
        clients = init_clients()
        armor = clients["armor"]
        ma = clients["armor_module"]
        request = ma.SanitizeUserPromptRequest(
            name=_armor_template_path(PROMPT_ARMOR_TEMPLATE),
            user_prompt_data=ma.DataItem(text=text),
        )
        response = armor.sanitize_user_prompt(request=request)
        result = getattr(response, "sanitization_result", response)
        # Top-level invocation result
        invoc = getattr(result, "filter_match_state", None)
        invoc_name = getattr(invoc, "name", str(invoc))
        flagged = _flagged_filters_from_result(result)
        # Exact-string compare — "MATCH_FOUND" is a substring of "NO_MATCH_FOUND"!
        passed = (str(invoc_name) != "MATCH_FOUND") and (not flagged)
        return passed, flagged
    except Exception as e:
        logger.error(
            "Model Armor prompt sanitize FAILED (fail-closed)",
            extra={"structured": {"error": str(e)}},
        )
        return False, ["armor_error"]


def sanitize_response(text: str) -> Tuple[bool, List[str]]:
    """Returns (passed, flagged_filters). Fail-closed: errors => blocked."""
    try:
        clients = init_clients()
        armor = clients["armor"]
        ma = clients["armor_module"]
        request = ma.SanitizeModelResponseRequest(
            name=_armor_template_path(RESPONSE_ARMOR_TEMPLATE),
            model_response_data=ma.DataItem(text=text),
        )
        response = armor.sanitize_model_response(request=request)
        result = getattr(response, "sanitization_result", response)
        invoc = getattr(result, "filter_match_state", None)
        invoc_name = getattr(invoc, "name", str(invoc))
        flagged = _flagged_filters_from_result(result)
        # Exact-string compare — "MATCH_FOUND" is a substring of "NO_MATCH_FOUND"!
        passed = (str(invoc_name) != "MATCH_FOUND") and (not flagged)
        return passed, flagged
    except Exception as e:
        logger.error(
            "Model Armor response sanitize FAILED (fail-closed)",
            extra={"structured": {"error": str(e)}},
        )
        return False, ["armor_error"]


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def search_kb(question: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Embed the question with task_type=RETRIEVAL_QUERY and run BigQuery
    VECTOR_SEARCH against ads_kb. Returns a list of dicts:
    {source_file, content, distance}.
    """
    from vertexai.language_models import TextEmbeddingInput

    clients = init_clients()
    embed_model = clients["embed"]
    bq = clients["bq"]

    q_input = TextEmbeddingInput(text=question, task_type="RETRIEVAL_QUERY")
    q_emb = embed_model.get_embeddings([q_input])[0].values

    # BigQuery VECTOR_SEARCH syntax: searches a base table column ('embedding')
    # against a query vector, returning top_k neighbors by distance.
    sql = f"""
    SELECT
      base.source_file  AS source_file,
      base.chunk_index  AS chunk_index,
      base.content      AS content,
      distance
    FROM VECTOR_SEARCH(
      TABLE `{PROJECT_ID}.{BQ_DATASET}.{BQ_KB_TABLE}`,
      'embedding',
      (SELECT @qvec AS embedding),
      top_k => @top_k,
      distance_type => 'COSINE'
    )
    ORDER BY distance ASC
    """
    from google.cloud import bigquery
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("qvec", "FLOAT64", list(q_emb)),
            bigquery.ScalarQueryParameter("top_k", "INT64", int(top_k)),
        ]
    )
    rows = bq.query(sql, job_config=job_config).result()
    return [
        {
            "source_file": r["source_file"],
            "chunk_index": r["chunk_index"],
            "content": r["content"],
            "distance": float(r["distance"]),
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Tool: Alaska 511 live data
# ---------------------------------------------------------------------------
# When a citizen asks about road conditions, closures, or airport status the
# agent fetches live data from 511.alaska.gov and includes it in the Gemini
# context alongside the static knowledge base.
#
# API documentation: https://511.alaska.gov/developers/doc
# - Base URL: https://511.alaska.gov/api/v2/get/{endpoint}
# - Auth: API key required as ?key=... query parameter
# - Rate limit: 10 calls / 60 seconds
# - Format: JSON (default) or XML
#
# Set the API key via the ALASKA_511_API_KEY env var. If unset, the tool is
# disabled (logged + returns empty); answers fall back to KB-only context.

import urllib.request
import urllib.error
from urllib.parse import urlencode

_ALASKA_511_BASE    = "https://511.alaska.gov/api/v2/get"
_ALASKA_511_KEY     = os.getenv("ALASKA_511_API_KEY", "")
_ALASKA_511_TIMEOUT = 8   # seconds — keep low so the chat stays responsive

# Endpoints verified from the official Alaska 511 developer docs
_511_ENDPOINTS = {
    "events":    "event",         # roadwork, closures, accidents
    "winter":    "winterroads",   # current winter road conditions
    "airports":  "airports",      # airport info / status
    "cameras":   "cameras",       # traffic cameras
    "alerts":    "alerts",        # alert notifications
}

# Map question keywords to the best endpoint to call
_511_INTENT_MAP = (
    # (keyword, endpoint)
    ("airport",          "airports"),
    ("winter driving",   "winter"),
    ("driving condition","winter"),
    ("road condition",   "winter"),
    ("road closure",     "events"),
    ("road closed",      "events"),
    ("highway closure",  "events"),
    ("highway closed",   "events"),
    ("highway condition","winter"),
    ("traffic",          "events"),
    ("accident",         "events"),
    ("roadwork",         "events"),
    ("511",              "events"),
)


def is_511_enabled() -> bool:
    """The 511 tool is enabled only when an API key is configured."""
    return bool(_ALASKA_511_KEY)


def needs_511_data(question: str) -> Optional[str]:
    """Return the matching 511 endpoint key, or None if not applicable
    OR the 511 tool is disabled (no API key)."""
    if not is_511_enabled():
        return None
    q = (question or "").lower()
    for keyword, endpoint in _511_INTENT_MAP:
        if keyword in q:
            return endpoint
    return None


def fetch_511_data(category: str = "events", limit: int = 10) -> List[Dict[str, Any]]:
    """Fetch live data from Alaska 511. Returns [] on any failure or missing key."""
    path = _511_ENDPOINTS.get(category)
    if not path:
        return []
    if not _ALASKA_511_KEY:
        logger.info(
            "alaska_511_skipped_no_key",
            extra={"structured": {"category": category}},
        )
        return []

    query = urlencode({"key": _ALASKA_511_KEY, "format": "json"})
    url = f"{_ALASKA_511_BASE}/{path}?{query}"
    safe_url = f"{_ALASKA_511_BASE}/{path}?format=json"   # logged without the key
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "ads-agent/1.0", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=_ALASKA_511_TIMEOUT) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        items = payload if isinstance(payload, list) else [payload]
        logger.info(
            "alaska_511_fetch_ok",
            extra={"structured": {"category": category, "url": safe_url, "count": len(items)}},
        )
        return items[:limit]
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
        logger.warning(
            "alaska_511_fetch_failed",
            extra={"structured": {"category": category, "url": safe_url, "error": str(e)}},
        )
        return []


def _has_value(v: Any) -> bool:
    """True if v is meaningfully present (not None / empty / whitespace)."""
    if v is None:
        return False
    if isinstance(v, str) and not v.strip():
        return False
    if isinstance(v, (list, dict)) and not v:
        return False
    return True


def _format_unix_ts(ts: Any) -> Optional[str]:
    """Convert a Unix timestamp (int/float/str) to a readable UTC string."""
    try:
        n = int(ts)
        if n <= 0:
            return None
        return datetime.fromtimestamp(n, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except (TypeError, ValueError):
        return None


def _format_winter_roads(header: str, items: List[Dict[str, Any]]) -> str:
    """Format /winterroads items: one road per block with all relevant detail.

    Field names are taken from the live Alaska 511 winterroads response (note
    that several keys contain spaces, e.g. 'Overall Status', 'Surface
    Conditions'). Null/empty fields are skipped so the LLM doesn't waste
    context on them.
    """
    lines = [f"[{header} — fetched at request time]"]

    for i, e in enumerate(items, 1):
        road     = e.get("RoadwayName") or f"Road #{i}"
        area     = e.get("AreaName")
        section  = e.get("LocationDescription")
        status   = e.get("Overall Status")
        surface  = e.get("Surface Conditions")
        atmos    = e.get("Atmospheric Conditions")
        warnings = e.get("Warnings")
        comments = e.get("Comments")

        # Temperature range (only if at least one bound present)
        t_min = e.get("Temperature (F) Min")
        t_max = e.get("Temperature (F) Max")
        if _has_value(t_min) and _has_value(t_max):
            temp = f"{t_min}°F to {t_max}°F"
        elif _has_value(t_min) or _has_value(t_max):
            temp = f"{t_min or t_max}°F"
        else:
            temp = None

        # Snow accumulation range
        s_min = e.get("Snow (inches) Min")
        s_max = e.get("Snow (inches) Max")
        if _has_value(s_min) and _has_value(s_max):
            snow = f"{s_min}\" to {s_max}\""
        elif _has_value(s_min) or _has_value(s_max):
            snow = f"{s_min or s_max}\""
        else:
            snow = None

        # Wind summary
        w_speed_min = e.get("Wind Speed (MPH) Min")
        w_speed_max = e.get("Wind Speed (MPH) Max")
        w_dir       = e.get("Wind Direction")
        w_type      = e.get("Wind Type")
        wind_bits = []
        if _has_value(w_speed_min) and _has_value(w_speed_max):
            wind_bits.append(f"{w_speed_min}-{w_speed_max} mph")
        elif _has_value(w_speed_min) or _has_value(w_speed_max):
            wind_bits.append(f"{w_speed_min or w_speed_max} mph")
        if _has_value(w_dir):  wind_bits.append(str(w_dir))
        if _has_value(w_type): wind_bits.append(f"({w_type})")
        wind = " ".join(wind_bits) if wind_bits else None

        updated_ts = _format_unix_ts(e.get("LastUpdated"))

        block = [f"  {i}. {road}" + (f" — {area}" if _has_value(area) else "")]
        if _has_value(section):  block.append(f"     Section     : {section}")
        if _has_value(status):   block.append(f"     Status      : {status}")
        if _has_value(surface):  block.append(f"     Surface     : {surface}")
        if _has_value(atmos):    block.append(f"     Atmosphere  : {atmos}")
        if _has_value(warnings): block.append(f"     Warnings    : {warnings}")
        if temp:                 block.append(f"     Temperature : {temp}")
        if snow:                 block.append(f"     Snow        : {snow}")
        if wind:                 block.append(f"     Wind        : {wind}")
        if _has_value(comments): block.append(f"     Comments    : {comments}")
        if updated_ts:           block.append(f"     Last update : {updated_ts}")

        lines.append("\n".join(block))
    return "\n".join(lines)


def format_511_context(category: str, events: List[Dict[str, Any]]) -> str:
    """Pretty-print 511 results for inclusion in the Gemini prompt."""
    if not events:
        return ""
    header = {
        "events":   "Live traffic events from Alaska 511",
        "winter":   "Live winter road conditions from Alaska 511",
        "airports": "Live airport status from Alaska 511",
        "cameras":  "Live traffic camera feeds from Alaska 511",
        "alerts":   "Live alerts from Alaska 511",
    }.get(category, "Live data from Alaska 511")

    # winterroads needs a different output shape — one road per block with
    # full condition detail, not the one-line summary used for traffic events.
    if category == "winter":
        return _format_winter_roads(header, events)

    lines = [f"[{header} — fetched at request time]"]
    for i, e in enumerate(events, 1):
        # Field names vary by endpoint — try common variants
        title    = e.get("Description") or e.get("Name") or e.get("EventType") or "Item"
        roadway  = e.get("RoadwayName") or e.get("LocationDescription") or ""
        severity = e.get("Severity") or ""
        closure  = " (full closure)" if e.get("IsFullClosure") else ""
        evt_type = e.get("EventType") or ""
        bits = []
        if evt_type: bits.append(f"[{evt_type}]")
        bits.append(str(title))
        if roadway:  bits.append(f"on {roadway}")
        if severity: bits.append(f"severity={severity}")
        lines.append(f"  {i}. " + " ".join(bits) + closure)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool: National Weather Service (NWS) live data
# ---------------------------------------------------------------------------
# When a citizen asks about weather, current temperature, snowfall, or
# forecast, the agent calls the public NWS API (api.weather.gov).
#
# API documentation: https://www.weather.gov/documentation/services-web-api
# - No API key required.
# - User-Agent header is REQUIRED — should identify the caller with contact info.
# - Pipeline: /points/{lat},{lon}  ->  forecast URL or observation stations URL.
# - Endpoints used:
#   * Current weather  -> /stations/{stationId}/observations/latest
#   * Forecast (7-day) -> /gridpoints/{office}/{x},{y}/forecast
#                         (returned via points lookup as `forecast` URL)

_NWS_BASE       = "https://api.weather.gov"
_NWS_USER_AGENT = os.getenv("NWS_USER_AGENT", "ads-agent (contact: ads@alaska.example.gov)")
_NWS_TIMEOUT    = 8   # seconds

# Common Alaska cities — lat / lon. Used to anchor weather lookups.
_ALASKA_CITIES: Dict[str, Tuple[float, float]] = {
    "anchorage":   (61.2181, -149.9003),
    "fairbanks":   (64.8378, -147.7164),
    "juneau":      (58.3019, -134.4197),
    "wasilla":     (61.5814, -149.4394),
    "palmer":      (61.5994, -149.1128),
    "kenai":       (60.5544, -151.2583),
    "soldotna":    (60.4878, -151.0583),
    "homer":       (59.6425, -151.5483),
    "seward":      (60.1042, -149.4422),
    "valdez":      (61.1308, -146.3483),
    "nome":        (64.5011, -165.4064),
    "kodiak":      (57.7900, -152.4072),
    "sitka":       (57.0531, -135.3300),
    "ketchikan":   (55.3422, -131.6461),
    "barrow":      (71.2906, -156.7886),
    "utqiagvik":   (71.2906, -156.7886),   # same as Barrow
    "bethel":      (60.7922, -161.7558),
}

_WEATHER_KEYWORDS = (
    "weather", "temperature", "snowfall", "snow forecast",
    "blizzard", "winter storm", "wind chill", "precipitation",
    "rain", "snow", "fog", "icy",
)

_WEATHER_CURRENT_KEYWORDS = (
    "current weather", "weather right now", "weather now",
    "current temperature", "current conditions", "right now",
    "outside now", "today right now",
)

_WEATHER_FORECAST_KEYWORDS = (
    "forecast", "tomorrow", "tonight", "this week",
    "weekend", "next few days", "later today", "extended",
)


def needs_weather_data(question: str) -> Optional[str]:
    """Return 'current' or 'forecast' for a weather question, else None."""
    q = (question or "").lower()
    if not any(kw in q for kw in _WEATHER_KEYWORDS):
        return None
    if any(kw in q for kw in _WEATHER_CURRENT_KEYWORDS):
        return "current"
    if any(kw in q for kw in _WEATHER_FORECAST_KEYWORDS):
        return "forecast"
    # Bare "weather" defaults to forecast — broader coverage is more useful.
    return "forecast"


def detect_alaska_location(question: str) -> Tuple[str, float, float]:
    """Find an Alaska city mentioned in the question. Defaults to Anchorage."""
    q = (question or "").lower()
    for city, (lat, lon) in _ALASKA_CITIES.items():
        if city in q:
            return city.title(), lat, lon
    lat, lon = _ALASKA_CITIES["anchorage"]
    return "Anchorage", lat, lon


def _nws_get(url: str) -> Optional[Dict[str, Any]]:
    """GET an NWS endpoint with the required User-Agent header."""
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": _NWS_USER_AGENT,
                "Accept": "application/geo+json",
            },
        )
        with urllib.request.urlopen(req, timeout=_NWS_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
        logger.warning(
            "nws_fetch_failed",
            extra={"structured": {"url": url, "error": str(e)}},
        )
        return None


def fetch_weather_data(intent: str, lat: float, lon: float) -> Dict[str, Any]:
    """Fetch current observation OR multi-day forecast for a lat/lon.

    NWS API flow:
      1. GET /points/{lat},{lon}  -> returns forecast & observationStations URLs
      2. For 'current':  GET observationStations -> first station -> latest obs
         For 'forecast': GET forecast URL directly
    """
    points = _nws_get(f"{_NWS_BASE}/points/{lat:.4f},{lon:.4f}")
    if not points:
        return {}
    props = points.get("properties", {}) or {}

    if intent == "current":
        stations_url = props.get("observationStations")
        if not stations_url:
            return {}
        stations = _nws_get(stations_url)
        if not stations:
            return {}
        features = stations.get("features") or []
        if not features:
            return {}
        station_id = features[0].get("properties", {}).get("stationIdentifier")
        if not station_id:
            return {}
        obs = _nws_get(f"{_NWS_BASE}/stations/{station_id}/observations/latest")
        if not obs:
            return {}
        return {"type": "current", "station": station_id, "obs": obs.get("properties", {}) or {}}

    # forecast path
    forecast_url = props.get("forecast")
    if not forecast_url:
        return {}
    forecast = _nws_get(forecast_url)
    if not forecast:
        return {}
    periods = (forecast.get("properties") or {}).get("periods") or []
    return {"type": "forecast", "periods": periods}


def format_weather_context(city: str, data: Dict[str, Any]) -> str:
    """Pretty-print NWS data for inclusion in the Gemini prompt."""
    if not data:
        return ""
    if data.get("type") == "current":
        obs = data.get("obs", {})
        station = data.get("station", "")
        # Temperatures from NWS are SI (celsius); convert to F for clarity
        temp_c = (obs.get("temperature") or {}).get("value")
        temp_f = f"{(temp_c * 9 / 5) + 32:.1f}°F" if isinstance(temp_c, (int, float)) else "n/a"
        desc = obs.get("textDescription", "") or "n/a"
        wind_kph = (obs.get("windSpeed") or {}).get("value")
        wind_mph = f"{wind_kph * 0.621371:.1f} mph" if isinstance(wind_kph, (int, float)) else "n/a"
        wind_dir = (obs.get("windDirection") or {}).get("value")
        wind_dir_str = f"{wind_dir:.0f}°" if isinstance(wind_dir, (int, float)) else ""
        ts = obs.get("timestamp", "")
        return (
            f"[Current weather for {city} (NWS station {station}, observed {ts})]\n"
            f"  Conditions : {desc}\n"
            f"  Temperature: {temp_f}\n"
            f"  Wind       : {wind_mph} {wind_dir_str}".strip()
        )
    if data.get("type") == "forecast":
        periods = data.get("periods", [])
        if not periods:
            return ""
        lines = [f"[NWS 7-day forecast for {city}]"]
        for p in periods[:8]:   # next ~4 days (each day has 2 periods: day + night)
            name = p.get("name", "")
            short = p.get("shortForecast", "")
            temp = p.get("temperature", "")
            unit = p.get("temperatureUnit", "")
            wind = p.get("windSpeed", "")
            wind_dir = p.get("windDirection", "")
            lines.append(f"  {name}: {short} | {temp}°{unit} | wind {wind} {wind_dir}".rstrip())
        return "\n".join(lines)
    return ""


# ---------------------------------------------------------------------------
# Prompt assembly and generation
# ---------------------------------------------------------------------------

def build_prompt(
    question: str,
    chunks: List[Dict[str, Any]],
    live_data: str = "",
) -> str:
    """Assemble the Gemini prompt with system instructions, context, question.
    `live_data` is optional pre-formatted text appended after the KB context
    (used for Alaska 511 tool output)."""
    if chunks:
        context_block = "\n\n".join(
            f"[Source: {c['source_file']} chunk {c.get('chunk_index', '?')}]\n{c['content']}"
            for c in chunks
        )
    else:
        context_block = "(no relevant context found)"

    # Put LIVE DATA *before* the static KB context — the model treats earlier
    # content as the primary source, and live data is always more authoritative
    # for the topics it covers.
    live_block = f"--- LIVE DATA (real-time, authoritative for the topics it covers) ---\n{live_data}\n--- END LIVE DATA ---\n\n" if live_data else ""

    return (
        f"{build_system_instruction()}\n\n"
        f"{live_block}"
        f"--- CONTEXT ---\n{context_block}\n--- END CONTEXT ---\n\n"
        f"Citizen question: {question}\n\n"
        f"Answer:"
    )


def call_gemini(prompt: str) -> str:
    """Invoke Gemini 2.5 Flash. Temperature 0 for deterministic, grounded output."""
    from vertexai.generative_models import GenerationConfig

    clients = init_clients()
    gen = clients["gen"]
    # Temperature 0: same prompt + context => same answer. Critical for the
    # tool-use path so the model can't randomly ignore the LIVE DATA block.
    config = GenerationConfig(temperature=0.0, max_output_tokens=1024)
    response = gen.generate_content(prompt, generation_config=config)
    text = getattr(response, "text", None)
    if text is None:
        # Some SDK versions require walking candidates -> content -> parts
        try:
            text = response.candidates[0].content.parts[0].text
        except Exception:
            text = ""
    return (text or "").strip()


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------

def log_interaction(
    session_id: str,
    user_prompt: str,
    prompt_blocked: bool,
    prompt_filters: List[str],
    retrieved_chunks: List[Dict[str, Any]],
    llm_response: str,
    response_blocked: bool,
    response_filters: List[str],
    latency_ms: int,
) -> None:
    """Write to BQ ads_audit AND emit a structured Cloud Logging record."""
    now = datetime.now(timezone.utc)
    row = {
        "timestamp": now.isoformat(),
        "session_id": session_id,
        "user_prompt": user_prompt,
        "prompt_blocked": prompt_blocked,
        "prompt_filters": json.dumps(prompt_filters),
        "retrieved_chunks": json.dumps(
            [
                {
                    "source_file": c.get("source_file"),
                    "chunk_index": c.get("chunk_index"),
                    "distance": c.get("distance"),
                }
                for c in retrieved_chunks
            ]
        ),
        "llm_response": llm_response,
        "response_blocked": response_blocked,
        "response_filters": json.dumps(response_filters),
        "latency_ms": int(latency_ms),
    }

    # Structured log first — this is best-effort and never raises
    logger.info(
        "ads_agent_interaction",
        extra={"structured": row},
    )

    # BigQuery audit — tolerate failures (don't break user traffic)
    try:
        clients = init_clients()
        bq = clients["bq"]
        table_id = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_AUDIT_TABLE}"
        errors = bq.insert_rows_json(table_id, [row])
        if errors:
            logger.warning(
                "BQ audit insert returned errors",
                extra={"structured": {"errors": str(errors)}},
            )
    except Exception as e:
        logger.warning(
            "BQ audit insert FAILED (tolerated)",
            extra={"structured": {"error": str(e)}},
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def answer(question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """End-to-end RAG pipeline with safety filtering and audit logging."""
    if not session_id:
        session_id = str(uuid.uuid4())
    started = time.time()

    # 1. Prompt sanitization
    prompt_ok, prompt_filters = sanitize_prompt(question)
    if not prompt_ok:
        latency_ms = int((time.time() - started) * 1000)
        log_interaction(
            session_id=session_id,
            user_prompt=question,
            prompt_blocked=True,
            prompt_filters=prompt_filters,
            retrieved_chunks=[],
            llm_response="",
            response_blocked=False,
            response_filters=[],
            latency_ms=latency_ms,
        )
        return {
            "blocked": True,
            "stage": "prompt",
            "filters": prompt_filters,
            "answer": "Your question was blocked by safety filters.",
            "session_id": session_id,
            "latency_ms": latency_ms,
            "sources": [],
        }

    # 2a. Retrieval
    try:
        chunks = search_kb(question)
    except Exception as e:
        logger.error(
            "Vector search FAILED",
            extra={"structured": {"error": str(e)}},
        )
        chunks = []

    # 2b. Live tools — each is keyword-gated and runs only when relevant.
    live_blocks: List[str] = []

    intent_511 = needs_511_data(question)
    if intent_511:
        events = fetch_511_data(category=intent_511, limit=10)
        block = format_511_context(intent_511, events)
        if block:
            live_blocks.append(block)

    intent_weather = needs_weather_data(question)
    if intent_weather:
        city, lat, lon = detect_alaska_location(question)
        wx = fetch_weather_data(intent_weather, lat, lon)
        block = format_weather_context(city, wx)
        if block:
            live_blocks.append(block)

    live_data = "\n\n".join(live_blocks)

    # 3. Generation
    prompt = build_prompt(question, chunks, live_data=live_data)
    try:
        llm_text = call_gemini(prompt)
    except Exception as e:
        logger.error(
            "Gemini call FAILED",
            extra={"structured": {"error": str(e)}},
        )
        llm_text = ""

    # 4. Response sanitization
    response_ok, response_filters = sanitize_response(llm_text) if llm_text else (True, [])
    latency_ms = int((time.time() - started) * 1000)

    if not response_ok:
        log_interaction(
            session_id=session_id,
            user_prompt=question,
            prompt_blocked=False,
            prompt_filters=prompt_filters,
            retrieved_chunks=chunks,
            llm_response=llm_text,
            response_blocked=True,
            response_filters=response_filters,
            latency_ms=latency_ms,
        )
        return {
            "blocked": True,
            "stage": "response",
            "filters": response_filters,
            "answer": "The response was blocked by safety filters. Please contact the Alaska Department of Snow directly.",
            "session_id": session_id,
            "latency_ms": latency_ms,
            "sources": [],
        }

    # 5. Audit + return
    log_interaction(
        session_id=session_id,
        user_prompt=question,
        prompt_blocked=False,
        prompt_filters=prompt_filters,
        retrieved_chunks=chunks,
        llm_response=llm_text,
        response_blocked=False,
        response_filters=response_filters,
        latency_ms=latency_ms,
    )

    unique_sources = sorted({c["source_file"] for c in chunks})
    return {
        "blocked": False,
        "answer": llm_text or "I don't know — please contact the Alaska Department of Snow directly.",
        "sources": unique_sources,
        "session_id": session_id,
        "latency_ms": latency_ms,
    }
