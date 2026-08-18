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

output "gcs_bucket_url" {
  description = "The URL of the GCS bucket."
  value       = module.storage.bucket_url
}

output "bigquery_dataset_id" {
  description = "The BigQuery dataset ID."
  value       = module.bigquery.dataset_id
}


