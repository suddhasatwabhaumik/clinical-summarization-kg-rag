# Copyright 2026 Google LLC
#
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

"""
Hypothesis Statistical Tables Exporter.

Queries the pipeline_evaluation_metrics table in BigQuery to compile mean averages,
standard deviations, and count distributions grouped by template types. Outputs
formatted tables ready for t-tests and ANOVA summaries.
"""

import os
import logging
from typing import Dict, Any
from google.cloud import bigquery

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def export_hypothesis_tables() -> None:
    """
    Queries evaluation metrics from BigQuery and displays formatted statistical tables.
    
    Raises:
        RuntimeError: If connection to BigQuery dataset fails.
    """
    project_id = os.environ.get("GCP_PROJECT", "suddhasatwa-data-projects")
    dataset_id = "clinical_summarization_eda"
    client = bigquery.Client(project=project_id)

    print("\n==========================================================================================")
    print("                 RESEARCH HYPOTHESIS VALIDATION TABLES (ANOVA & T-TESTS)")
    print("==========================================================================================\n")

    # 1. Loop 1 / RQ3 Prompts Comparison Table
    print("Research Question 3 (Prompt Strategies) - Aggregate Metrics Summary Table:")
    print("-------------------------------------------------------------------------")
    query_rq3 = f"""
    SELECT 
      template_type,
      COUNT(hadm_id) as sample_size,
      ROUND(AVG(rouge1), 4) as mean_rouge1,
      ROUND(STDDEV(rouge1), 4) as std_rouge1,
      ROUND(AVG(rougeL), 4) as mean_rougeL,
      ROUND(STDDEV(rougeL), 4) as std_rougeL,
      ROUND(AVG(bertscore_f1), 4) as mean_bertscore,
      ROUND(STDDEV(bertscore_f1), 4) as std_bertscore,
      ROUND(AVG(creola_cer), 4) as mean_cer,
      ROUND(STDDEV(creola_cer), 4) as std_cer,
      ROUND(AVG(entity_f1), 4) as mean_entity_f1
    FROM `{project_id}.{dataset_id}.pipeline_evaluation_metrics`
    GROUP BY template_type
    ORDER BY template_type DESC
    """
    try:
        results = client.query(query_rq3).result()
        rows = list(results)
        if rows:
            print(
                f"  {'Prompt Type':<12} | {'N':<4} | "
                f"{'ROUGE-1 (Mean ± SD)':<22} | "
                f"{'ROUGE-L (Mean ± SD)':<22} | "
                f"{'BERTScore (Mean ± SD)':<24} | "
                f"{'CREOLA CER (Mean ± SD)':<25} | "
                f"{'Entity F1':<10}"
            )
            print(f"  {'-'*12}-+-{'-'*4}-+-{'-'*22}-+-{'-'*22}-+-{'-'*24}-+-{'-'*25}-+-{'-'*10}")
            for r in rows:
                r1_str = f"{r.mean_rouge1:.4f} ± {r.std_rouge1 if r.std_rouge1 else 0.0:.4f}"
                rl_str = f"{r.mean_rougeL:.4f} ± {r.std_rougeL if r.std_rougeL else 0.0:.4f}"
                bs_str = f"{r.mean_bertscore:.4f} ± {r.std_bertscore if r.std_bertscore else 0.0:.4f}"
                cer_str = f"{r.mean_cer:.4f} ± {r.std_cer if r.std_cer else 0.0:.4f}"
                print(
                    f"  {r.template_type:<12} | {r.sample_size:<4} | "
                    f"{r1_str:<22} | "
                    f"{rl_str:<22} | "
                    f"{bs_str:<24} | "
                    f"{cer_str:<25} | "
                    f"{r.mean_entity_f1:.4f}"
                )
        else:
            print("  No comparative prompt metrics recorded in BigQuery table yet. Run run-evaluator first.")
    except Exception as e:
        logger.error(f"Failed to query BigQuery comparative metrics: {str(e)}")
        raise RuntimeError("BigQuery query execution error.") from e

    # 2. Hypothesis tests guidelines output
    print("\nStatistical Analysis Notes for Publication:")
    print("------------------------------------------")
    print("  * For RQ1 (RAG vs. KG-RAG): Compare ROUGE/CER between 'standard' (Control) and 'got' (Experimental) groups using a Paired Samples t-test.")
    print("  * For RQ3 (Prompts): Compare ROUGE/CER across 'standard', 'cot', and 'got' groups using a One-way ANOVA.")
    print("  * For RQ4 (Correlation): Use Spearman's Rank Correlation to cross-verify BERTScore against Clinician Safety scores.")
    print("\n==========================================================================================\n")


def main() -> None:
    """
    Main entrypoint running the statistical report compiler.
    """
    export_hypothesis_tables()


if __name__ == "__main__":
    main()
