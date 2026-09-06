# 05. Prompting Topologies and LLM Generation

This document details the meta-prompting topologies evaluated in this research, their mathematical formulations, and how generative synthesis is orchestrated via Google Vertex AI.

---

## 1. Meta-Prompting Topologies Evaluated (RQ3)

To determine how prompting strategies interact with factual retention and clinical hallucinations, four distinct prompting topologies were benchmarked across the clinical cohort:

```
                                  +---------------------------------------+
                                  |         INPUT CLINICAL CONTEXT        |
                                  |    (EHR Notes + Retrieved Context)    |
                                  +-------------------+-------------------+
                                                      |
         +--------------------+-----------------------+-----------------------+--------------------+
         |                    |                                               |                    |
         v                    v                                               v                    v
+-----------------+  +-----------------+                            +-----------------+  +-----------------+
|   Standard      |  |    Chain of     |                            |      Self       |  |    Graph of     |
|   Zero-Shot     |  |     Thought     |                            |   Consistency   |  |     Thought     |
|     (Base)      |  |      (CoT)      |                            |     (K = 3)     |  |      (GoT)      |
+--------+--------+  +--------+--------+                            +--------+--------+  +--------+--------+
         |                    |                                              |                    |
         | Direct             | Step-by-Step                                 | Sample K=3 at      | Ontological
         | Summary            | Diagnostic                                   | T = 0.7, Select    | Triples Injected
         | Request            | Reasoning                                    | ROUGE-L Medoid     | as Constraints
         |                    |                                              |                    |
         v                    v                                              v                    v
  [CER = 0.450]        [CER = 0.250]                                  [CER = 0.180]        [CER = 0.120]
  [Entity = 0.50]      [Entity = 0.50]                                [Entity = 0.50]      [Entity = 0.75]
```

---

## 2. Formal Prompt Template Specifications

All prompt templates are defined using LangChain `PromptTemplate` in `src/rag_engine.py`.

### 1. Standard Zero-Shot Template
Instructs the model to generate a direct summary from unstructured text and semantic chunks without intermediate reasoning:
```
You are an expert clinical summarization assistant. Review the clinical notes and context below to generate a clear, professional clinical summary.

Context (Semantic Text Chunks):
{context}

Clinical Notes:
{clinical_notes}

Clinical Summary:
```

### 2. Chain-of-Thought (CoT) Template
Forces the LLM to articulate intermediate diagnostic and pharmacological reasoning steps prior to writing the final summary:
```
You are an expert clinical summarization assistant. Reconcile the notes and context step-by-step to produce the summary.

Instructions:
1. Identify the chief complaint, primary diagnoses, and patient history.
2. Outline key clinical interventions, lab findings, and medications.
3. List discharge recommendations, medications, and follow-up plans.
4. Synthesize the above points into a clean, cohesive discharge summary.

Context (Semantic Text Chunks):
{context}

Clinical Notes:
{clinical_notes}

Reasoning Steps & Summary:
```

### 3. Self-Consistency Ensembling ($K = 3, T = 0.7$)
Samples $K = 3$ independent generation candidates from the model at an elevated temperature ($T = 0.7$) to introduce stochastic diversity. 

The consensus summary $S^*$ is selected by computing the pairwise ROUGE-L overlap across all pairs and identifying the **medoid**:
$$\arg\max_{S_i} \frac{1}{K-1} \sum_{j \neq i} \text{ROUGE-L}(S_i, S_j)$$

This suppresses idiosyncratic single-run hallucinations, reducing CER from 0.45 to 0.18.

### 4. Graph-of-Thought (GoT) Template (Proposed Architecture)
Explicitly injects linearized multi-hop relational paths from the Knowledge Graph into the prompt payload. The LLM is strictly instructed to treat these paths as inviolable ground-truth constraints:
```
You are an expert clinical summarization assistant. You are provided with semantic text chunks and logical relationships extracted from a medical knowledge graph.

Logical Medical Connections (Knowledge Graph):
{graph_paths}

Semantic Details (Text Chunks):
{context}

Clinical Notes:
{clinical_notes}

Instructions:
1. Reconcile the logical medical concepts/paths (e.g. drugs treating diseases, symptoms indicating conditions) with the unstructured notes.
2. Structure the summary, ensuring the relations and factual chunks are accurately represented.
3. Do not include fabricated statements that contradict the text chunks or graph paths.

Clinical Summary:
```

---

## 3. Vertex AI LLM Generation Engine

Generative inference is executed through the `LLMSummarizer` class in `src/rag_engine.py`, connecting to Google Cloud Vertex AI.

### Model Parameters
* **Target Model**: `gemini-3-pro-preview` (Long-context multi-modal foundation model).
* **Secondary Model**: `gemini-3-flash` (Optimized for latency and cost).
* **Region**: `us-central1`.
* **Inference Hyperparameters**:
  ```python
  GenerationConfig(
      temperature=0.2,       # Deterministic synthesis
      top_p=0.95,            # Nucleus sampling bound
      top_k=40,              # Vocabulary filtering
      max_output_tokens=1024 # Discharge handover budget
  )
  ```

### Why Graph-of-Thought Succeeds
While Zero-Shot and CoT prompting achieve high lexical overlap (ROUGE-1 $\approx 0.88$), their clinical error rates remain unacceptably high ($\text{CER} = 0.45$ and $0.25$). Autoregressive decoders prioritize fluent n-gram continuations, frequently omitting negative statements (such as drug allergies) because allergy statements appear infrequently relative to positive diagnostic assertions.

By converting implicit medical associations into explicit prompt constraints, **Graph-of-Thought achieves $\text{CER} = 0.12$ and boosts core medical entity recall to $0.75$**.
