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

# 1. Create pipeline runner Service Account
resource "google_service_account" "kg_rag_runner" {
  account_id   = "kg-rag-runner"
  display_name = "KG-RAG Pipeline Runner"
  project      = var.project_id
}

# 2. Grant Vertex AI User role to the Service Account
resource "google_project_iam_member" "vertex_ai_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.kg_rag_runner.email}"
}

# 3. Grant BigQuery Job User role to the Service Account
resource "google_project_iam_member" "bq_job_user" {
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.kg_rag_runner.email}"
}

# 4. Create Secret Manager secret for Neo4j password
resource "google_secret_manager_secret" "neo4j_password" {
  secret_id = "clinical-rag-neo4j-password"
  project   = var.project_id

  replication {
    auto {}
  }
}

# 5. Grant Secret Manager access to the service account
resource "google_secret_manager_secret_iam_member" "accessor" {
  project   = var.project_id
  secret_id = google_secret_manager_secret.neo4j_password.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.kg_rag_runner.email}"
}
