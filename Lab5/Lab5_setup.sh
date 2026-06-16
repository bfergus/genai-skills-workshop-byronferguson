#!/usr/bin/env bash
# Lab5_setup.sh — End-to-end setup for the Alaska Department of Snow agent.
# Automates Phases 1 (prep), 2 (loading), and 4 (deployment) from the runbook.
# Phase 3 (local testing) is left manual on purpose — you should eyeball the
# UI and confirm tests pass before deploying.
#
# Usage:
#   bash Lab5_setup.sh              # full run
#   bash Lab5_setup.sh --skip-prep  # skip API/IAM/template setup
#   bash Lab5_setup.sh --skip-load  # skip ingest (KB already loaded)
#   bash Lab5_setup.sh --skip-deploy
#   bash Lab5_setup.sh --prep-only
#
# Prereqs:
#   - gcloud, gsutil, python3, pip installed and on PATH
#   - You are already authenticated:
#       gcloud auth login
#       gcloud auth application-default login

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — edit these before running
# ---------------------------------------------------------------------------
PROJECT_ID="${PROJECT_ID:-qwiklabs-gcp-00-16d0362ac1ac}"
LOCATION="${LOCATION:-us-east4}"
SERVICE_NAME="${SERVICE_NAME:-ads-agent}"
BQ_DATASET="${BQ_DATASET:-ads}"
PROMPT_TEMPLATE_ID="${PROMPT_TEMPLATE_ID:-ads-prompt-template}"
RESPONSE_TEMPLATE_ID="${RESPONSE_TEMPLATE_ID:-ads-response-template}"

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
SKIP_PREP=false
SKIP_LOAD=false
SKIP_DEPLOY=false

for arg in "$@"; do
  case $arg in
    --skip-prep)   SKIP_PREP=true ;;
    --skip-load)   SKIP_LOAD=true ;;
    --skip-deploy) SKIP_DEPLOY=true ;;
    --prep-only)   SKIP_LOAD=true; SKIP_DEPLOY=true ;;
    -h|--help)
      grep '^#' "$0" | head -25
      exit 0
      ;;
    *) echo "Unknown flag: $arg"; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
banner() {
  echo
  echo "============================================================"
  echo "  $1"
  echo "============================================================"
}

require_var() {
  if [ -z "${!1}" ]; then        # only checks for empty
    echo "ERROR: $1 is not set."
    exit 1
  fi
}

require_var PROJECT_ID

PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

echo "Project        : $PROJECT_ID"
echo "Project number : $PROJECT_NUMBER"
echo "Region         : $LOCATION"
echo "Service name   : $SERVICE_NAME"
echo "Runtime SA     : $RUNTIME_SA"

gcloud config set project "$PROJECT_ID" --quiet

# ---------------------------------------------------------------------------
# PHASE 1 — Prep
# ---------------------------------------------------------------------------
if [ "$SKIP_PREP" = false ]; then
  banner "PHASE 1: Enabling APIs"
  gcloud services enable \
    aiplatform.googleapis.com \
    bigquery.googleapis.com \
    modelarmor.googleapis.com \
    storage.googleapis.com \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    logging.googleapis.com \
    --project="$PROJECT_ID"

  banner "PHASE 1: Granting IAM roles to Cloud Run runtime SA"
  for role in \
    roles/aiplatform.user \
    roles/bigquery.dataEditor \
    roles/bigquery.jobUser \
    roles/modelarmor.user \
    roles/logging.logWriter
  do
    echo "  $role"
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
      --member="serviceAccount:${RUNTIME_SA}" \
      --role="$role" \
      --condition=None \
      --quiet > /dev/null
  done

  banner "PHASE 1: Installing Model Armor Python client"
  pip install --quiet google-cloud-modelarmor

  banner "PHASE 1: Creating Model Armor templates"
  python3 <<PYEOF
import os
from google.cloud import modelarmor_v1
from google.cloud.modelarmor_v1 import (
    ModelArmorClient, CreateTemplateRequest, Template,
)
from google.api_core.exceptions import AlreadyExists

PROJECT_ID = "${PROJECT_ID}"
LOCATION   = "${LOCATION}"
PARENT     = f"projects/{PROJECT_ID}/locations/{LOCATION}"

client = ModelArmorClient(
    client_options={"api_endpoint": f"modelarmor.{LOCATION}.rep.googleapis.com"}
)

PiSettings  = modelarmor_v1.PiAndJailbreakFilterSettings
MalSettings = modelarmor_v1.MaliciousUriFilterSettings
RaiSettings = modelarmor_v1.RaiFilterSettings
SdpBasic    = modelarmor_v1.SdpBasicConfig
SdpFilter   = modelarmor_v1.SdpFilterSettings
Confidence  = modelarmor_v1.DetectionConfidenceLevel
RaiType     = modelarmor_v1.RaiFilterType

RAI = [
    RaiSettings.RaiFilter(filter_type=RaiType.HATE_SPEECH,       confidence_level=Confidence.HIGH),
    RaiSettings.RaiFilter(filter_type=RaiType.HARASSMENT,        confidence_level=Confidence.HIGH),
    RaiSettings.RaiFilter(filter_type=RaiType.SEXUALLY_EXPLICIT, confidence_level=Confidence.MEDIUM_AND_ABOVE),
    RaiSettings.RaiFilter(filter_type=RaiType.DANGEROUS,         confidence_level=Confidence.MEDIUM_AND_ABOVE),
]

def prompt_template():
    return Template(filter_config=modelarmor_v1.FilterConfig(
        pi_and_jailbreak_filter_settings=PiSettings(
            filter_enforcement=PiSettings.PiAndJailbreakFilterEnforcement.ENABLED,
            confidence_level=Confidence.MEDIUM_AND_ABOVE),
        malicious_uri_filter_settings=MalSettings(
            filter_enforcement=MalSettings.MaliciousUriFilterEnforcement.ENABLED),
        rai_settings=RaiSettings(rai_filters=RAI),
    ))

def response_template():
    return Template(filter_config=modelarmor_v1.FilterConfig(
        pi_and_jailbreak_filter_settings=PiSettings(
            filter_enforcement=PiSettings.PiAndJailbreakFilterEnforcement.ENABLED,
            confidence_level=Confidence.MEDIUM_AND_ABOVE),
        malicious_uri_filter_settings=MalSettings(
            filter_enforcement=MalSettings.MaliciousUriFilterEnforcement.ENABLED),
        rai_settings=RaiSettings(rai_filters=RAI),
        sdp_settings=SdpFilter(basic_config=SdpBasic(
            filter_enforcement=SdpBasic.SdpBasicConfigEnforcement.ENABLED)),
    ))

for tid, builder in [
    ("${PROMPT_TEMPLATE_ID}",   prompt_template),
    ("${RESPONSE_TEMPLATE_ID}", response_template),
]:
    try:
        result = client.create_template(request=CreateTemplateRequest(
            parent=PARENT, template_id=tid, template=builder()))
        print(f"  Created : {result.name}")
    except AlreadyExists:
        print(f"  Exists  : {PARENT}/templates/{tid} (skipped)")
PYEOF

  banner "PHASE 1: Creating BigQuery dataset"
  bq --location="$LOCATION" mk -d --description "Alaska Dept of Snow knowledge base" "${PROJECT_ID}:${BQ_DATASET}" 2>/dev/null \
    && echo "  Dataset created: ${BQ_DATASET}" \
    || echo "  Dataset exists : ${BQ_DATASET} (ok)"
fi

# ---------------------------------------------------------------------------
# PHASE 2 — Loading (run the ingest notebook as a script)
# ---------------------------------------------------------------------------
if [ "$SKIP_LOAD" = false ]; then
  banner "PHASE 2: Installing ingest dependencies"
  pip install --quiet -r Lab5_requirements.txt pypdf beautifulsoup4 || true

  banner "PHASE 2: Running ingest (Lab5_ingest.py)"
  # # %% cell markers are treated as comments by plain python, so the file
  # runs end-to-end as a normal script.
  PROJECT_ID="$PROJECT_ID" LOCATION="$LOCATION" python3 Lab5_ingest.py
fi

# ---------------------------------------------------------------------------
# PHASE 4 — Deployment
# ---------------------------------------------------------------------------
if [ "$SKIP_DEPLOY" = false ]; then
  banner "PHASE 4: Deploying to Cloud Run"
  PROJECT_ID="$PROJECT_ID" \
  LOCATION="$LOCATION" \
  SERVICE_NAME="$SERVICE_NAME" \
    bash Lab5_deploy.sh
fi

banner "Setup complete"
echo
echo "Next steps:"
echo "  - Verify rows in BigQuery:"
echo "      bq query --use_legacy_sql=false \"SELECT COUNT(*) FROM \\\`${PROJECT_ID}.${BQ_DATASET}.ads_kb\\\`\""
echo "  - Open the Cloud Run URL printed above and ask a test question"
echo "  - Run tests locally:    pytest Lab5_tests.py -v"
echo "  - Run evaluation:       python3 Lab5_eval.py"
