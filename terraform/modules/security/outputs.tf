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

output "runner_service_account_email" {
  description = "The email of the pipeline runner service account."
  value       = google_service_account.kg_rag_runner.email
}

output "secret_id" {
  description = "The ID of the Secret Manager secret for Neo4j."
  value       = google_secret_manager_secret.neo4j_password.secret_id
}
