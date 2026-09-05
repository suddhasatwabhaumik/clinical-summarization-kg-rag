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

# Clinical Summarization KG-RAG Pipeline

An advanced, hybrid Knowledge Graph-Retrieval Augmented Generation (KG-RAG) pipeline designed to summarize unstructured MIMIC-IV discharge summaries using semantic search (ChromaDB) and clinical logical connections (Neo4j) to mitigate hallucinations.

---

## Directory Structure

```text
clinical-summarization-kg-rag/
├── data/
│   ├── raw/                  # Source CSV notes (e.g. discharge.csv)
│   └── processed/            # Cleaned notes, extraction JSONs, Chroma DB storage
├── notebooks/                # Research and experimental notebooks
│   ├── 01_data_preprocessing.ipynb           # MIMIC-IV cleaning, regex PHI masking, UMLS NER
│   ├── 02_knowledge_graph_construction.ipynb # BigQuery Graph schema, triple ingestion, multi-hop paths
│   └── 03_rag_pipeline_and_eval.ipynb       # Dense vector search, meta-prompting, metrics evaluation
├── src/
│   ├── __init__.py           # Package init
│   ├── data_processor.py     # MIMIC-IV notes loading, cleaning, UMLS CUI linker
│   ├── graph_builder.py      # BigQuery/Neo4j graph schema, bulk merge ingestion
│   ├── rag_engine.py         # Chunker, Embedding factory, hybrid retrieval, prompt templates
│   ├── evaluation.py         # ROUGE, BERTScore, CREOLA CER, Entity F1 calculator
│   ├── evaluate_embeddings.py# Comparative embedding model evaluation (RQ2 ANOVA)
│   ├── clinician_correlation.py # Clinician review deck exporter and Spearman correlation (RQ4)
│   ├── export_hypothesis_tables.py # BigQuery stats compiler for ANOVA/t-test tables
│   ├── generate_synthetic_data.py # Large-scale MIMIC-IV synthetic dataset generator
│   ├── run_evaluator.py          # E2E evaluative generation loop across prompt strategies
│   └── show_experiment_results.py # Terminal CLI experiment results formatter
├── terraform/                # Infrastructure-as-code modules for GCP deployment
│   ├── main.tf
│   ├── variables.tf
│   ├── outputs.tf
│   └── modules/ ...
├── Makefile                  # System-wide orchestration Makefile
├── requirements.txt          # Python dependencies
└── README.md                 # Main documentation and guide
```

---

## Research Notebooks

The methodology phases referenced in the research paper can be interactively reproduced step-by-step using the provided Jupyter Notebooks:

1. **[01_data_preprocessing.ipynb](notebooks/01_data_preprocessing.ipynb)**: Data ingestion, regex de-identification, biomedical entity extraction, and UMLS Metathesaurus CUI linking.
2. **[02_knowledge_graph_construction.ipynb](notebooks/02_knowledge_graph_construction.ipynb)**: Knowledge graph schema initialization, node/edge ingestion, and multi-hop (2-hop) neighborhood graph traversal.
3. **[03_rag_pipeline_and_eval.ipynb](notebooks/03_rag_pipeline_and_eval.ipynb)**: Dense semantic retrieval, meta-prompting (Standard, CoT, Self-Consistency, GoT), Gemini LLM synthesis, CREOLA CER error scoring, and statistical hypothesis testing (RQ1–RQ4).

---

## Local Prerequisites & Setup

### 1. Python Environment Setup
Activate a virtual environment and install requirements:

```bash
# Prepare environment and models using Makefile
make setup-env
```

This installs Python packages and downloads the specialized UMLS model `en_core_sci_sm` for scispacy.

### 2. Database (Neo4j & ChromaDB)
- **ChromaDB**: Runs locally inside `data/processed/chroma_db` using persistent folder structures.
- **Neo4j**: Can be run via Docker:
  ```bash
  docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/your_password -d neo4j:5.12.0
  ```

---

## GCP Deployment (suddhasatwa-data-projects)

To deploy the pipeline production-ready on GCP, use the Terraform modules located in the `terraform/` directory.

### 1. GCP Architecture & Flow
1. **Compute Engine VM**: Runs the processing pipelines and hosts the Neo4j Docker container in a secure, private VPC subnet.
2. **Cloud Storage (GCS)**: Stores raw MIMIC CSV logs and output summaries.
3. **Secret Manager**: Securely stores Neo4j passwords, retrieved dynamically by VM instances on start.
4. **Google BigQuery**: Stores EDA and performance metrics in structured tables.
5. **Looker Studio**: Connects to BigQuery for visualization.

### 2. Complete Setup & Pipeline Execution Orchestration
To deploy all GCP resources, configure secrets, upload your data, and execute the remote pipeline in a single step, make sure your raw summaries are located at `data/raw/discharge.csv`, and run:
```bash
make setup-and-run
```

Alternatively, you can run the provisioning steps individually:
```bash
make tf-init
make tf-plan
make tf-apply
```


See [terraform/README.md](terraform/README.md) for detailed inputs and configuration guidelines.

---

## Execution Guide

Orchestrate the execution phases locally or inside your VM:

### Step 1: Preprocess MIMIC notes and Extract UMLS Entities
Place your raw clinical notes file at `data/raw/discharge.csv`. Alternatively, you can auto-generate a small synthetic mock dataset (2 rows) or generate a research-scale synthetic dataset (200,000 rows) using Gemini on Vertex AI to feed the pipeline:

**Option A: Quick Mock Dataset (2 rows)**
```bash
make create-mock-data
```

**Option B: Research-Scale Synthetic Dataset (200,000 rows via Vertex AI Gemini)**
```bash
make generate-synthetic-data
```


Once the raw data is ready, run the local pipeline:
```bash
make run-pipeline
```

Or trigger the fully automated remote GCP pipeline:
```bash
make run-remote-pipeline-gcp
```


### Step 2: Build the Medical Knowledge Graph
Ingest concepts into Neo4j:
```python
from src.graph_builder import GraphBuilder
import pandas as pd
import json

# Connection uses environment variables or GCP Secret Manager
builder = GraphBuilder()
builder.connect()
builder.create_constraints_and_indexes()

# Load processed data
df = pd.read_csv("data/processed/processed_data.csv")

# Bulk ingest concepts
for idx, row in df.iterrows():
    entities = json.loads(row['entities'])
    builder.ingest_entities(entities)

builder.close()
```

### Step 3: Run Retrieval & Evaluation
Evaluate generated output using precision metrics:
```python
from src.evaluation import EvaluationEngine

evaluator = EvaluationEngine()
reference = "Patient has history of asthma. Prescribed albuterol."
candidate = "Patient has history of acute asthma and is given albuterol."

# ROUGE & BERTScore
print(evaluator.calculate_rouge_scores(reference, candidate))
print(evaluator.calculate_bertscore(reference, candidate))

# CREOLA CER
# Formula: CER = (5*E_fab + 4*E_neg + 3*E_cau + 2*E_ctx) / S_total
print("CREOLA CER:", evaluator.calculate_creola_cer(e_fab=0, e_neg=1, e_cau=0, e_ctx=0, s_total=5))
```

---

## BigQuery Analytics Schema

The pipeline exports analytical details into the following tables in the `clinical_summarization_eda` dataset:

### 1. `eda_entity_frequencies`
Tracks clinical entity occurrence stats:
- `cui` (STRING, Required): UMLS Unique CUI
- `name` (STRING, Required): Canonical name
- `semantic_type` (STRING, Nullable): Semantic domain
- `frequency` (INTEGER, Required): Frequency count
- `last_updated` (TIMESTAMP, Required): Update timestamp

### 2. `pipeline_evaluation_metrics`
Tracks performance comparison metrics across prompt variants:
- `hadm_id` (INTEGER, Required): Unique admission ID
- `template_type` (STRING, Required): Prompt variant (`standard`, `cot`, `got`)
- `rouge1` / `rouge2` / `rougeL` (FLOAT, Required): ROUGE F-measures
- `bertscore_f1` (FLOAT, Required): BERTScore F1
- `creola_cer` (FLOAT, Required): CREOLA Clinical Error Rate
- `entity_f1` (FLOAT, Required): Entity exact match F-measure
- `generated_at` (TIMESTAMP, Required): Evaluation timestamp
