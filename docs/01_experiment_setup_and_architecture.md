# 01. Experiment Setup and Environment Architecture

This document details the complete experimental setup, research design, system architecture, and cloud infrastructure required to execute and reproduce the Clinical Summarization KG-RAG research.

---

## 1. Research Objectives & Formal Hypotheses

Automating clinical handover summaries via autoregressive Large Language Models (LLMs) presents severe patient safety risks due to **clinical hallucinations** (fabricated diagnoses, contraindicated drug recommendations) and **critical omissions** (dropping drug allergies or chronic comorbidities). 

This research investigates whether coupling dense semantic retrieval with an enterprise Knowledge Graph (KG) and Graph-of-Thought (GoT) prompting eliminates these risks.

### Research Questions (RQs) and Hypotheses

* **RQ1 (Clinical Hallucination Mitigation)**: Does grounding LLM summarization with multi-hop knowledge graph paths significantly reduce the CREOLA Clinical Error Rate (CER) compared to baseline vector-only RAG?
  * $H_0$: $\mu_{\text{CER, KG-RAG}} \ge \mu_{\text{CER, Baseline RAG}}$
  * $H_1$: $\mu_{\text{CER, KG-RAG}} < \mu_{\text{CER, Baseline RAG}}$
  * *Test*: Two-tailed Paired Samples $t$-test ($\alpha = 0.05$).

* **RQ2 (Biomedical Dense Embedding Precision)**: Which domain-adapted embedding representation (Bio_ClinicalBERT, BioLinkBERT, or PubMedBERT) provides the highest concept retrieval precision over unstructured clinical EHR notes?
  * $H_0$: $\mu_{\text{precision, Bio\_ClinicalBERT}} = \mu_{\text{precision, BioLinkBERT}} = \mu_{\text{precision, PubMedBERT}}$
  * $H_1$: At least one embedding representation exhibits significantly different retrieval precision.
  * *Test*: One-Way Analysis of Variance (ANOVA) with Tukey HSD post-hoc test.

* **RQ3 (Meta-Prompting Topology Efficacy)**: How do advanced prompting paradigms (Standard Zero-Shot, Chain-of-Thought, Self-Consistency, and Graph-of-Thought) affect factual correctness, entity retention, and clinical error rates?
  * $H_0$: Mean CER is invariant across prompting strategies.
  * $H_1$: Graph-of-Thought prompting achieves significantly lower CER and higher Entity F1.
  * *Test*: Repeated Measures ANOVA across prompt conditions.

* **RQ4 (Automated vs. Human Clinician Safety Correlation)**: Do automated NLP metrics (ROUGE-L, BERTScore F1, CREOLA CER) correlate significantly with blinded expert clinician safety ratings?
  * $H_0$: Spearman rank correlation $\rho = 0$.
  * $H_1$: Spearman rank correlation $\rho \neq 0$.
  * *Test*: Two-tailed Spearman Rank Correlation against blinded physician Likert evaluations.

---

## 2. End-to-End System Architecture

The research system employs a **dual-path neuro-symbolic architecture**:

```
+-----------------------------------------------------------------------------------+
|                              EHR INGESTION TIER                                   |
|   MIMIC-IV Discharge Summaries (N = 158 Inpatient Encounters)                     |
|   Regex Cleaning | HIPAA Bracket Removal | Abbreviation Normalization             |
+-----------------------------------------+-----------------------------------------+
                                          |
                     +--------------------+--------------------+
                     |                                         |
                     v                                         v
+------------------------------------+   +------------------------------------+
|        VECTOR ENCODING PATH        |   |       KNOWLEDGE GRAPH PATH         |
|                                    |   |                                    |
| 512-Token Sliding Windows          |   | scispacy Named Entity Recognition  |
| (64-Token Overlap)                 |   | (en_core_sci_sm Pipeline)          |
|                                    |   |                                    |
| Domain Embeddings (d = 768):       |   | UMLS Metathesaurus CUI Linking     |
| - Bio_ClinicalBERT                 |   | Semantic Types & Relationship Map  |
| - BioLinkBERT                      |   |                                    |
| - PubMedBERT                       |   | Enterprise BigQuery Graph Store:   |
|                                    |   | - graph_nodes (Concept Vertices)   |
| Vector Store:                      |   | - graph_edges (Typed Relational)   |
| Persistent ChromaDB Collection     |   |                                    |
+-----------------+------------------+   +-----------------+------------------+
                  |                                        |
                  +-------------------+--------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------------+
|                            HYBRID RETRIEVAL & FUSION                              |
|   - Vector Search: Top-3 Dense Semantic Chunks (Cosine Distance)                  |
|   - Graph Traversal: 2-Hop Subgraph Neighborhoods via BigQuery SQL                |
|   - Path Linearization: (ConceptA) -[:REL]-> (ConceptB) -[:REL]-> (ConceptC)      |
+-------------------------------------+---------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------------+
|                        GRAPH-OF-THOUGHT (GoT) PROMPTING                           |
|   Injects ontological constraints, clinical facts, and reasoning directives:      |
|   "Reconcile logical medical paths with unstructured narrative text..."           |
+-------------------------------------+---------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------------+
|                        GENERATIVE INFERENCE TIER                                  |
|   Google Vertex AI Generative Foundation Models:                                  |
|   - Gemini 3 Pro (Primary Analytical Reasoner, T = 0.2)                         |
|   - Gemini 3 Flash (High-Throughput Verification)                               |
|   Exponential Backoff Retry Strategy | Deterministic Fallbacks                    |
+-------------------------------------+---------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------------+
|                     EVALUATION & BIGQUERY LOGGING TIER                            |
|   - Automated NLP: ROUGE-1/2/L, BERTScore F1, scispacy Entity F1                  |
|   - Safety Metric: CREOLA Clinical Error Rate (CER)                               |
|   - Expert Deck: Blinded Clinician Reviews (N = 100, Likert 1-5 Safety)           |
|   - BigQuery Tables: experiment_results, clinician_evaluations                    |
+-----------------------------------------------------------------------------------+
```

---

## 3. Infrastructure & Cloud Architecture

The computational infrastructure leverages Google Cloud Platform (GCP) for high-scale enterprise storage, serverless graph querying, and foundation model inference.

### Cloud Components
* **Google BigQuery**:
  * Acts as the serverless Knowledge Graph store (`graph_nodes`, `graph_edges`).
  * Stores longitudinal evaluation metrics (`experiment_results`, `clinician_evaluations`).
  * Executes multi-hop SQL joins with sub-second latency across hundreds of thousands of clinical concepts.
* **Google Vertex AI**:
  * Hosts Gemini 3 Pro and Gemini 3 Flash models in `us-central1`.
  * Configured with temperature $T = 0.2$ for deterministic clinical synthesis.
* **Google Cloud Storage (GCS)**:
  * Persistent storage bucket (`gs://suddhasatwa-clinical-rag-data/`) for raw discharge notes, processed parquet files, and vector index snapshots.
* **Terraform Infrastructure-as-Code (IaC)**:
  * Modular infrastructure defined in `terraform/` for one-click deployment of BigQuery datasets, tables, IAM roles, and storage buckets.

### Local Compute Environment
* **OS**: macOS / Linux (Ubuntu 22.04 LTS recommended for cloud deployment).
* **Python Runtime**: Python 3.11 or Python 3.12.
* **GPU Requirements**: CPU-only execution is fully supported for ChromaDB inference with SentenceTransformers. CUDA/MPS acceleration can optionally be used for batch embedding indexing.

---

## 4. Software Dependencies & Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/suddhasatwabhaumik/clinical-summarization-kg-rag.git
cd clinical-summarization-kg-rag
```

### Step 2: Virtual Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Install Biomedical Language Models
Install the scispacy biomedical model for entity extraction:
```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
```

### Core Dependency Inventory
* `google-cloud-bigquery` (>= 3.25.0): BigQuery graph store and evaluation queries.
* `google-cloud-storage` (>= 2.18.0): Cloud storage persistence.
* `google-cloud-aiplatform` (>= 1.60.0): Vertex AI Gemini client.
* `chromadb` (>= 0.5.5): In-memory and persistent vector database.
* `transformers` (>= 4.44.0) & `sentence-transformers` (>= 3.0.0): HuggingFace biomedical encoders.
* `spacy` (>= 3.7.0) & `scispacy` (>= 0.5.4): Clinical entity recognition.
* `rouge-score` (>= 0.1.2) & `bert-score` (>= 0.3.13): NLP lexical and semantic evaluation.
* `pandas` (>= 2.2.0), `scipy` (>= 1.13.0), `statsmodels` (>= 0.14.0): Statistical hypothesis testing.
* `matplotlib` (>= 3.9.0) & `seaborn` (>= 0.13.0): Publication-ready 300-DPI visual generation.

---

## 5. Configuration & Environment Variables

Configure your local environment by creating a `.env` file or exporting variables:

```bash
# Google Cloud Project Configuration
export GCP_PROJECT="suddhasatwa-data-projects"
export GCP_REGION="us-central1"
export BQ_DATASET="clinical_summarization_eda"

# Google Cloud Service Account Authentication
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.gcp/sa-credentials.json"

# Direct Gemini API key for offline/local SDK development
export GEMINI_API_KEY="your-gemini-api-key"
```

---

## 6. Execution Pipeline Workflow

The research workflow can be executed using the root Makefile or shell scripts:

1. **Infrastructure Provisioning**:
   ```bash
   cd terraform
   terraform init
   terraform apply -auto-approve
   cd ..
   ```
2. **Data Preprocessing & Ingestion**:
   ```bash
   python src/data_processor.py
   ```
3. **Knowledge Graph Ingestion**:
   ```bash
   python src/graph_builder.py
   ```
4. **End-to-End Pipeline Demo**:
   ```bash
   python demo_pipeline.py
   ```
5. **Full Statistical Evaluation & Hypothesis Testing**:
   ```bash
   python src/run_evaluator.py
   python src/evaluate_embeddings.py
   python src/clinician_correlation.py
   ```
6. **Academic Manuscript Compilation**:
   ```bash
   python src/latex_generator.py
   ```
