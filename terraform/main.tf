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

terraform {
  required_version = ">= 1.3.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.38.0"
    }

  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# 1. Security & IAM configuration module
module "security" {
  source     = "./modules/security"
  project_id = var.project_id
}

# 2. Storage Bucket configuration module
module "storage" {
  source                       = "./modules/storage"
  project_id                   = var.project_id
  region                       = var.region
  runner_service_account_email = module.security.runner_service_account_email
}

# 3. BigQuery Dataset & Tables configuration module
module "bigquery" {
  source                       = "./modules/bigquery"
  project_id                   = var.project_id
  region                       = var.region
  runner_service_account_email = module.security.runner_service_account_email
}



