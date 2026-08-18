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

# 1. Create BigQuery Dataset
resource "google_bigquery_dataset" "eda_dataset" {
  dataset_id                  = "clinical_summarization_eda"
  project                     = var.project_id
  location                    = var.region
  description                 = "Dataset containing clinical EDA and evaluation metrics"
  delete_contents_on_destroy  = false
}

# 2. Table: eda_entity_frequencies
resource "google_bigquery_table" "entity_frequencies" {
  dataset_id = google_bigquery_dataset.eda_dataset.dataset_id
  project    = var.project_id
  table_id   = "eda_entity_frequencies"
  deletion_protection = false

  schema = <<EOF
[
  {
    "name": "cui",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "UMLS Concept Unique Identifier"
  },
  {
    "name": "name",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "Canonical UMLS entity name"
  },
  {
    "name": "semantic_type",
    "type": "STRING",
    "mode": "NULLABLE",
    "description": "Biomedical semantic type"
  },
  {
    "name": "frequency",
    "type": "INTEGER",
    "mode": "REQUIRED",
    "description": "Occurrence count"
  },
  {
    "name": "last_updated",
    "type": "TIMESTAMP",
    "mode": "REQUIRED",
    "description": "Record update timestamp"
  }
]
EOF
}

# 3. Table: pipeline_evaluation_metrics
resource "google_bigquery_table" "evaluation_metrics" {
  dataset_id = google_bigquery_dataset.eda_dataset.dataset_id
  project    = var.project_id
  table_id   = "pipeline_evaluation_metrics"
  deletion_protection = false

  schema = <<EOF
[
  {
    "name": "hadm_id",
    "type": "INTEGER",
    "mode": "REQUIRED",
    "description": "Patient admission unique identifier"
  },
  {
    "name": "template_type",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "Prompting template pattern (standard, cot, got)"
  },
  {
    "name": "rouge1",
    "type": "FLOAT",
    "mode": "REQUIRED",
    "description": "ROUGE-1 F1 score"
  },
  {
    "name": "rouge2",
    "type": "FLOAT",
    "mode": "REQUIRED",
    "description": "ROUGE-2 F1 score"
  },
  {
    "name": "rougeL",
    "type": "FLOAT",
    "mode": "REQUIRED",
    "description": "ROUGE-L F1 score"
  },
  {
    "name": "bertscore_f1",
    "type": "FLOAT",
    "mode": "REQUIRED",
    "description": "BERTScore F1 score"
  },
  {
    "name": "creola_cer",
    "type": "FLOAT",
    "mode": "REQUIRED",
    "description": "CREOLA Clinical Error Rate"
  },
  {
    "name": "entity_f1",
    "type": "FLOAT",
    "mode": "REQUIRED",
    "description": "Entity exact-match F1 score"
  },
  {
    "name": "generated_at",
    "type": "TIMESTAMP",
    "mode": "REQUIRED",
    "description": "Record creation timestamp"
  }
]
EOF
}

# 4. Table: graph_nodes
resource "google_bigquery_table" "graph_nodes" {
  dataset_id = google_bigquery_dataset.eda_dataset.dataset_id
  project    = var.project_id
  table_id   = "graph_nodes"
  deletion_protection = false

  schema = <<EOF
[
  {
    "name": "cui",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "UMLS Concept Unique Identifier"
  },
  {
    "name": "name",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "Canonical UMLS entity name"
  },
  {
    "name": "types",
    "type": "STRING",
    "mode": "REPEATED",
    "description": "Biomedical semantic types"
  }
]
EOF
}

# 5. Table: graph_edges
resource "google_bigquery_table" "graph_edges" {
  dataset_id = google_bigquery_dataset.eda_dataset.dataset_id
  project    = var.project_id
  table_id   = "graph_edges"
  deletion_protection = false

  schema = <<EOF
[
  {
    "name": "cui_from",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "Source Concept CUI"
  },
  {
    "name": "cui_to",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "Target Concept CUI"
  },
  {
    "name": "rel_type",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "Relationship type label"
  },
  {
    "name": "updated_at",
    "type": "TIMESTAMP",
    "mode": "REQUIRED",
    "description": "Record creation/update timestamp"
  }
]
EOF
}

# 6. Grant BigQuery Data Editor role to the Service Account
resource "google_bigquery_dataset_access" "runner_write" {
  dataset_id    = google_bigquery_dataset.eda_dataset.dataset_id
  project       = var.project_id
  role          = "roles/bigquery.dataEditor"
  user_by_email = var.runner_service_account_email
}

