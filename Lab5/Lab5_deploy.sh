#!/usr/bin/env bash
# Lab5_deploy.sh — build and deploy the ADS agent to Cloud Run.
#
# Usage:
#   PROJECT_ID=your-gcp-project-id bash Lab5_deploy.sh
#
set -euo pipefail

# Resolve project ID at runtime via gcloud if not explicitly set
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
if [ -z "${PROJECT_ID}" ]; then
  echo "ERROR: Could not resolve project ID. Set PROJECT_ID or run 'gcloud config set project ...'"
  exit 1
fi
REGION="${REGION:-us-east4}"
SERVICE_NAME="${SERVICE_NAME:-ads-agent}"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:$(date +%Y%m%d-%H%M%S)"

echo "=========================================="
echo " Alaska Department of Snow — Cloud Run deploy"
echo "=========================================="
echo " Project : ${PROJECT_ID}"
echo " Region  : ${REGION}"
echo " Service : ${SERVICE_NAME}"
echo " Image   : ${IMAGE}"
echo "=========================================="

# Ensure gcloud points at the right project
gcloud config set project "${PROJECT_ID}" >/dev/null

# Build the container with Cloud Build.
# gcloud builds submit requires the Dockerfile to be named "Dockerfile".
# Copy Lab5_Dockerfile into place for the build, then remove it afterwards.
echo ">> Staging Dockerfile..."
cp Lab5_Dockerfile Dockerfile
trap 'rm -f Dockerfile' EXIT

# Stage a .gcloudignore so Cloud Build doesn't upload the entire repo
cat > .gcloudignore <<'IGNORE'
.git
.gitignore
__pycache__
*.pyc
.venv
venv
*.ipynb_checkpoints
Lab1_*
Lab2_*
Lab3_*
Lab5_eval_results.csv
IGNORE

echo ">> Building container image with Cloud Build..."
gcloud builds submit \
  --tag "${IMAGE}" \
  --project "${PROJECT_ID}" \
  .

# Deploy to Cloud Run
echo ">> Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --max-instances 10 \
  --concurrency 40 \
  --timeout 60 \
  --set-env-vars "PROJECT_ID=${PROJECT_ID},LOCATION=${REGION},BQ_DATASET=ads,BQ_KB_TABLE=ads_kb,BQ_AUDIT_TABLE=ads_audit,EMBED_MODEL=text-embedding-004,GEMINI_MODEL=gemini-2.5-flash,PROMPT_ARMOR_TEMPLATE=ads-prompt-template,RESPONSE_ARMOR_TEMPLATE=ads-response-template,ALASKA_511_API_KEY=${ALASKA_511_API_KEY:-},NWS_USER_AGENT=${NWS_USER_AGENT:-ads-agent (contact: ads@alaska.example.gov)}" \
  --project "${PROJECT_ID}"

URL="$(gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --format='value(status.url)')"
echo "=========================================="
echo " Deployed: ${URL}"
echo "=========================================="
