<!--
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# GCP Infrastructure Provisioning via Terraform

This folder contains the Terraform modules to automatically provision storage, databases, and compute nodes on Google Cloud Platform (GCP).

---

## Architecture Overview

The Terraform structure is divided into four modules:
1. **Security Module (`modules/security`)**: Provisions the pipeline Service Account (`kg-rag-runner`) with least-privilege IAM bindings, and sets up Secret Manager for Neo4j database passwords.
2. **Storage Module (`modules/storage`)**: Provisions the Google Cloud Storage bucket (`suddhasatwa-data-projects-mimic`) with encryption and public access prevention enabled.
3. **BigQuery Module (`modules/bigquery`)**: Provisions the dataset and table schemas (`eda_entity_frequencies` and `pipeline_evaluation_metrics`) for analytics.
4. **Compute Module (`modules/compute`)**: Provisions a private Compute Engine VM running Container-Optimized OS (COS) and launches a Neo4j database container.

---

## Inputs & Variables

Configure these variables inside your `terraform.tfvars` or pass them via command line (the Makefile passes defaults automatically):

| Variable | Description | Type | Default |
|---|---|---|---|
| `project_id` | GCP Project ID | `string` | `suddhasatwa-data-projects` |
| `region` | GCP Regional Location | `string` | `us-central1` |
| `zone` | Compute zone | `string` | `us-central1-a` |

---

## Outputs

After running `terraform apply`, the configuration exposes:

| Output | Description |
|---|---|
| `gcs_bucket_url` | Google Cloud Storage Bucket URI |
| `bigquery_dataset_id` | BigQuery analytical dataset ID |
| `vm_private_ip` | Private internal IP address of the GCE VM |

---

## Execution Guide

All execution is managed using the root `Makefile` targets:

### 1. Initialize Working Directory
Downloads provider plugins and initializes modules:
```bash
make tf-init
```

### 2. Preview Resource Changes
See what resources will be created before applying:
```bash
make tf-plan
```

### 3. Provision Infrastructure
Deploy all GCS, Secret Manager, BigQuery, and GCE resources:
```bash
make tf-apply
```

### 4. Seed Neo4j Credentials
Once infrastructure is applied, populate the password in Secret Manager:
```bash
echo -n "your_password" | gcloud secrets versions add clinical-rag-neo4j-password --data-file=- --project=suddhasatwa-data-projects
```

### 5. Tear Down Resources
To delete all resources and avoid running cost accrual:
```bash
make tf-destroy
```
