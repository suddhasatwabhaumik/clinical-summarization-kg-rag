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

# 1. Provision clinical GCS bucket
resource "google_storage_bucket" "mimic_bucket" {
  name                        = "${var.project_id}-mimic"
  project                     = var.project_id
  location                    = var.region
  force_destroy               = false
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = true
  }
}

# 2. Grant Storage Object Admin/User role to the runner service account
resource "google_storage_bucket_iam_member" "runner_access" {
  bucket = google_storage_bucket.mimic_bucket.name
  role   = "roles/storage.objectUser"
  member = "serviceAccount:${var.runner_service_account_email}"
}



