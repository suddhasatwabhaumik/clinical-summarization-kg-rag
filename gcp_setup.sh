#!/bin/bash
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

PROJECT_ID="suddhasatwa-data-projects"
REGION="us-central1"
ZONE="us-central1-a"

echo "=========================================================="
# Title
echo "GCP Setup Script: Clinical Summarization KG-RAG"
echo "=========================================================="

# 1. Check prerequisites
echo "Checking CLI prerequisites..."
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud (GCP SDK) CLI is not installed. Please install it first."
    exit 1
fi

if ! command -v terraform &> /dev/null; then
    echo "Error: terraform CLI is not installed. Please install it first."
    exit 1
fi

echo "Prerequisites verified!"

# 2. Authenticate and set project config
echo "Configuring active GCP project context to: ${PROJECT_ID}..."
gcloud config set project "$PROJECT_ID"

# 3. Enable necessary APIs
echo "Enabling GCP service APIs (Secret Manager, BigQuery, Vertex AI)..."
gcloud services enable \
    secretmanager.googleapis.com \
    bigquery.googleapis.com \
    aiplatform.googleapis.com \
    --project="$PROJECT_ID"


# 4. Provision infrastructure via Terraform
echo "Initializing and running Terraform configurations..."
# Using Makefile helper targets
make tf-init
make tf-apply PROJECT_ID="$PROJECT_ID" REGION="$REGION" ZONE="$ZONE"


# 6. Summary and next steps
echo ""
echo "=========================================================="
echo "DEPLOYMENT COMPLETE!"
echo "=========================================================="
echo "How to Operate:"
echo "1. Build and push the container image to Artifact Registry:"
echo "   make build-image"
echo ""
echo "2. Execute the pipeline job remotely on Cloud Run:"
echo "   make run-remote-pipeline-gcp"
echo "=========================================================="



