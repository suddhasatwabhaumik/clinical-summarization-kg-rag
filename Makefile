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

.PHONY: tf-init tf-plan tf-apply tf-destroy setup-env run-pipeline check-status clean upload-data run-pipeline-gcp setup-and-run create-mock-data generate-synthetic-data run-gcp-experiment help show-results run-evaluator export-stats

.DEFAULT_GOAL := help

PROJECT_ID ?= suddhasatwa-data-projects
REGION ?= us-central1
ZONE ?= us-central1-a
ROWS ?= 200000

help:
	@echo "Available Makefile Targets:"
	@echo "  tf-init                 - Initialize Terraform"
	@echo "  tf-plan                 - Dry-run and check Terraform changes"
	@echo "  tf-apply                - Deploy infrastructure changes"
	@echo "  tf-destroy              - Teardown all GCP infrastructure"
	@echo "  setup-env               - Prepare python environment (uv)"
	@echo "  run-pipeline            - Run clinical RAG pipeline locally"
	@echo "  run-pipeline-gcp        - Run pipeline locally pointing to GCP bucket paths"
	@echo "  generate-synthetic-data - Generate $(ROWS) synthetic clinical records via Gemini"
	@echo "  run-gcp-experiment      - Generate synthetic data and execute E2E GCP pipeline"
	@echo "  run-evaluator           - Run E2E Capstone Evaluator loop (Standard, CoT, GoT)"
	@echo "  export-stats            - Export statistical tables (ROUGE, BERTScore, CER)"
	@echo "  show-results            - Fetch and display aggregated statistics from BigQuery"
	@echo "  setup-and-run           - Run full infrastructure setup and execute E2E flow"
	@echo "  clean                   - Clean local caches and data"

# Terraform Targets
tf-init:
	cd terraform && terraform init


tf-plan:
	cd terraform && terraform plan -var="project_id=$(PROJECT_ID)" -var="region=$(REGION)" -var="zone=$(ZONE)"

tf-apply:
	cd terraform && terraform apply -var="project_id=$(PROJECT_ID)" -var="region=$(REGION)" -var="zone=$(ZONE)" -target=module.security -auto-approve
	@echo "Sleeping 25 seconds for Service Account propagation across GCP..."
	sleep 25
	cd terraform && terraform apply -var="project_id=$(PROJECT_ID)" -var="region=$(REGION)" -var="zone=$(ZONE)" -auto-approve


tf-destroy:
	cd terraform && terraform destroy -var="project_id=$(PROJECT_ID)" -var="region=$(REGION)" -var="zone=$(ZONE)" -auto-approve

# Python Environment Setup
setup-env:
	uv venv .venv
	. .venv/bin/activate && uv pip install -r requirements.txt
	. .venv/bin/activate && uv pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz


# Run Pipeline components
run-pipeline:
	@echo "Executing pipeline..."
	export SKIP_SCISPACY=True && . .venv/bin/activate && python -m src.data_processor data/raw/discharge.csv data/processed/processed_data.csv
	@echo "Pipeline executed. Run your database insertions or evaluation reports."

# GCP Service Verification status
check-status:
	@echo "Checking Compute Engine VM Neo4j Container status..."
	gcloud compute instances describe clinical-rag-vm \
		--project=$(PROJECT_ID) \
		--zone=$(ZONE) \
		--format="value(status)"
	@echo "Checking VM Container metadata..."
	gcloud compute instances describe clinical-rag-vm \
		--project=$(PROJECT_ID) \
		--zone=$(ZONE) \
		--format="value(metadata.items.gce-container-declaration)"

# Clean Local cache and generated outputs
clean:
	rm -rf __pycache__ src/__pycache__
	rm -rf .pytest_cache
	rm -rf data/processed/*

# GCP Storage upload
upload-data:
	@if [ ! -f data/raw/discharge.csv ]; then \
		make generate-synthetic-data; \
	fi
	@echo "Uploading local discharge.csv to GCS bucket gs://$(PROJECT_ID)-mimic..."
	gcloud storage cp data/raw/discharge.csv gs://$(PROJECT_ID)-mimic/raw/discharge.csv


# Local GCP execution (GCS paths, includes automatic data upload if needed)
run-pipeline-gcp: upload-data
	@echo "Executing pipeline locally using GCP Storage files..."
	export SKIP_SCISPACY=True && . .venv/bin/activate && python -m src.data_processor gs://$(PROJECT_ID)-mimic/raw/discharge.csv gs://$(PROJECT_ID)-mimic/processed/processed_data.csv

# Fetch and print aggregated statistics and metrics from BigQuery
show-results:
	@echo "Fetching experiment stats from BigQuery..."
	. .venv/bin/activate && python src/show_experiment_results.py

# Run E2E Capstone Evaluator loop (Generates summaries, evaluates, and writes stats to BigQuery)
run-evaluator:
	@echo "Executing E2E Capstone Evaluator across clinical notes..."
	. .venv/bin/activate && python -m src.run_capstone_evaluator --limit $(or $(N),158)

# Export aggregated summaries formatted for Capstone statistical tables
export-stats:
	@echo "Compiling statistical tables for ROUGE, BERTScore, and CER..."
	. .venv/bin/activate && python -m src.export_hypothesis_tables

# Run embedding models comparative evaluation (RQ2 ANOVA)
run-embedding-eval:
	@echo "Evaluating retrieval precision across ClinicalBERT, BioLinkBERT, and PubMedBERT..."
	. .venv/bin/activate && python -m src.evaluate_embeddings

# Export template reviews spreadsheet for clinicians (RQ4 Deck)
export-clinician-deck:
	@echo "Exporting clinician evaluation sheet template to data/processed/clinician_review_deck.csv..."
	. .venv/bin/activate && python -m src.clinician_correlation --action=export

# Run Spearman Correlation Analysis (RQ4 Correlation)
run-correlation:
	@echo "Calculating Spearman Rank Correlation between automated scores and clinician ratings..."
	. .venv/bin/activate && python -m src.clinician_correlation --action=correlate




# Complete E2E Setup and Run Flow
setup-and-run:
	@echo "Launching complete setup and run orchestrator..."
	./setup_and_run.sh

# Generate Mock MIMIC Data
create-mock-data:
	@echo "Generating mock MIMIC-IV discharge notes..."
	mkdir -p data/raw
	python3 -c "import csv; f=open('data/raw/discharge.csv', 'w', newline=''); w=csv.writer(f); w.writerow(['hadm_id', 'text']); w.writerow([10001, 'Patient [** Name **] is a 65yo male presenting with shortness of breath. Diagnosed with acute Asthma. Administered Albuterol inhaler. Status improved.']); w.writerow([10002, 'Patient presented with severe chest pain. Diagnosed with Myocardial Infarction. Prescribed Aspirin and Metoprolol. Referred to cardiology.']); f.close()"
	@echo "Mock data created successfully at data/raw/discharge.csv"

# Generate synthetic notes via Gemini on Vertex AI
generate-synthetic-data:
	@echo "Generating $(ROWS) synthetic clinical records..."
	. .venv/bin/activate && python src/generate_synthetic_data.py $(ROWS) data/raw/discharge.csv

# Step 1 + Step 2: Generate synthetic dataset and execute E2E GCP pipeline
run-gcp-experiment: generate-synthetic-data run-pipeline-gcp
	@echo "GCP Experiment execution finished!"





