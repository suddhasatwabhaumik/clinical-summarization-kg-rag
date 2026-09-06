# 02. Data Pipeline and Preprocessing

This guide describes the complete data processing pipeline used in this research, from raw Electronic Health Record (EHR) discharge notes to sanitized, chunked semantic segments and structured clinical entity extractions.

---

## 1. Clinical Cohort Selection & Curation

The primary clinical corpus is extracted from the **MIMIC-IV** (v2.2) clinical database (Johnson et al., 2023), accessed via PhysioNet under Data Use Agreement.

### Inclusion and Exclusion Criteria
* **Inclusion Criteria**:
  * Adult inpatient hospital admissions (age $\ge 18$).
  * Fully finalized, non-truncated discharge summaries (`discharge.csv.gz`).
  * Presence of validated hospital admission identifiers (`hadm_id`) and subject identifiers (`subject_id`).
  * Explicit documentation of history of present illness, hospital course, discharge medications, and follow-up recommendations.
* **Exclusion Criteria**:
  * Encounters lacking narrative clinical course text.
  * Notes exceeding maximum context token budget without segmental break points.
  * Duplicate addenda or preliminary radiology/pathology-only reports.

### Statistical Power Analysis
An a priori statistical power analysis was conducted to establish cohort size requirements across all research questions at significance level $\alpha = 0.05$ and statistical power $(1 - \beta) = 0.80$:
* **RQ1 (Paired $t$-test)**: Minimum sample size required for an anticipated large effect size ($d \ge 0.5$) is $N = 34$.
* **RQ2 (One-Way ANOVA across 3 groups)**: Minimum sample size required for medium effect size ($f = 0.25$) is $N = 158$.
* **Final Study Cohort**: Exactly $N = 158$ clinical encounters were analyzed, fully satisfying the maximal sample size requirements across all statistical tests.

---

## 2. Text Normalization & HIPAA De-Identification Sanitization

Raw EHR narratives contain irregular line breaks, inconsistent typographical artifacts, non-standard medical abbreviations, and de-identification brackets.

The normalization module in `src/data_processor.py` applies the following transformations:

### Regex Sanitation Rules
1. **HIPAA Placeholder Removal**:
   MIMIC discharge summaries mask Protected Health Information (PHI) with bracketed placeholders such as `[** 2182-7-14 **]`, `[** First Name (STitle) 1234 **]`, or `[** Hospital 12 **]`. These are stripped or normalized to generic references:
   ```python
   # Normalize date placeholders
   text = re.sub(r'\[\*\*\s*(?:Doctor|First Name|Last Name|Hospital|Location|Date).*?\*\*\]', '[DEID]', text)
   ```
2. **Whitespace and Line Break Regularization**:
   Erratic line wraps inserted by legacy EHR terminals are consolidated into standard paragraphs while preserving semantic section headers (`CHIEF COMPLAINT:`, `DISCHARGE MEDICATIONS:`).
3. **Medical Abbreviation Standardization**:
   Common clinical shorthand (e.g., `q.d.` $\to$ `daily`, `b.i.d.` $\to$ `twice daily`, `p.o.` $\to$ `orally`, `prn` $\to$ `as needed`) are normalized to prevent embedding mismatches.

---

## 3. Biomedical Entity Recognition (NER) & UMLS CUI Linking

To build the knowledge graph and guide retrieval, unstructured text is processed through domain-specific NLP pipelines.

```
+------------------------------------------------------------------------------------+
|                               UNSTRUCTURED NOTE                                    |
|   "Patient diagnosed with severe acute asthma exacerbation. Prescribed albuterol   |
|    nebulizer. Documented severe anaphylactic allergy to penicillin."               |
+-----------------------------------------+------------------------------------------+
                                          |
                                          v  scispacy (en_core_sci_sm)
+------------------------------------------------------------------------------------+
|                         EXTRACTED BIOMEDICAL ENTITIES                              |
|   - "acute asthma exacerbation" [DISEASE]                                          |
|   - "albuterol" [CHEMICAL / PHARMACOLOGIC]                                         |
|   - "penicillin" [CHEMICAL / PHARMACOLOGIC]                                        |
+-----------------------------------------+------------------------------------------+
                                          |
                                          v  UMLS Metathesaurus Linker
+------------------------------------------------------------------------------------+
|                         CANONICAL CUIs & SEMANTIC TYPES                            |
|   - Asthma Exacerbation: C0004096 (T047: Disease or Syndrome)                      |
|   - Albuterol: C0001927 (T121: Pharmacologic Substance)                            |
|   - Penicillin: C0030842 (T195: Antibiotic / Allergen)                             |
+------------------------------------------------------------------------------------+
```

### UMLS Linking Workflow
1. **Tokenization and Part-of-Speech Tagging**: Using the `en_core_sci_sm` model from `scispacy`.
2. **Concept Normalization**: Named entities are mapped to canonical Concept Unique Identifiers (CUIs) in the Unified Medical Language System (UMLS).
3. **Semantic Filtering**: Entities are categorized into validated semantic groups:
   * **Disorders / Diseases**: Congestive Heart Failure, Type 2 Diabetes, Sepsis.
   * **Chemicals & Drugs**: Metformin, Lisinopril, Albuterol, Penicillin.
   * **Signs & Symptoms**: Dyspnea, Diaphoresis, Wheezing, Peripheral Edema.

---

## 4. Sliding Window Text Chunking

Dense vector search requires chunk sizes that fit within the transformer's maximum context length while maintaining coherent semantic boundaries.

### Chunking Specification
* **Chunker Implementation**: `ClinicalTextChunker` in `src/rag_engine.py`.
* **Tokenizer**: Hugging Face `AutoTokenizer` initialized with the target embedding model.
* **Window Size**: 512 contiguous tokens.
* **Sliding Overlap**: 64 tokens.

```
Note Stream:  [0 .................... 512]
Overlap:                     [448 .... 512]
Next Chunk:                  [448 .................... 960]
```

The 64-token overlap guarantees that clinical sentences spanning chunk boundaries (e.g., a medication list where the drug name is at token 510 and the dosage instruction is at token 514) are not severed.

---

## 5. Synthetic Benchmarking Data Generation

To allow researchers to test and benchmark the entire pipeline offline without requiring immediate PhysioNet credentialed access or multi-gigabyte MIMIC downloads, `src/generate_synthetic_data.py` provides an authentic synthetic data generator.

### Characteristics of the Synthetic Corpus
* **Schema Parity**: Exactly mirrors MIMIC-IV `discharge.csv` structure (`note_id`, `subject_id`, `hadm_id`, `note_type`, `note_seq`, `charttime`, `text`).
* **Clinical Realism**: Generates complex multi-system inpatient discharge summaries featuring realistic comorbidities (e.g., COPD + Congestive Heart Failure + Atrial Fibrillation), inpatient procedures, hospital courses, and discharge plans.
* **Controlled Ground Truth**: Injects known contraindications and drug allergies to rigorously evaluate whether the KG-RAG pipeline prevents critical omissions.

### Generating the Synthetic Corpus
```bash
python src/generate_synthetic_data.py
```
This produces:
* `data/processed/processed_data.csv`: Cleaned, chunked dataset for vector indexing.
* `data/processed/clinician_review_deck.csv`: Curated case evaluation deck for clinician correlation studies.
