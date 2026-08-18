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

import os
import logging
from google.cloud import bigquery

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    project_id = os.environ.get("GCP_PROJECT", "suddhasatwa-data-projects")
    dataset_id = "clinical_summarization_eda"
    client = bigquery.Client(project=project_id)

    print("\n==========================================================")
    print("      CLINICAL SUMMARIZATION KG-RAG EXPERIMENT REPORT")
    print("==========================================================\n")

    # 1. Graph Summary Stats
    print("1. Knowledge Graph Database Summary:")
    print("-------------------------------------")
    try:
        nodes_query = f"SELECT COUNT(*) as count FROM `{project_id}.{dataset_id}.graph_nodes`"
        edges_query = f"SELECT COUNT(*) as count FROM `{project_id}.{dataset_id}.graph_edges`"
        
        nodes_count = list(client.query(nodes_query).result())[0].get("count")
        edges_count = list(client.query(edges_query).result())[0].get("count")
        
        print(f"  Total Unique Medical Concepts (Nodes) : {nodes_count}")
        print(f"  Total Concept Co-occurrences (Edges)   : {edges_count}")
    except Exception as e:
        print(f"  Failed to retrieve graph metrics: {str(e)}")

    # 2. Top 10 extracted concepts
    print("\n2. Top 10 Most Frequent Medical Concepts (EDA):")
    print("-----------------------------------------------")
    try:
        eda_query = f"""
        SELECT cui, name, semantic_type, frequency 
        FROM `{project_id}.{dataset_id}.eda_entity_frequencies` 
        ORDER BY frequency DESC 
        LIMIT 10
        """
        results = client.query(eda_query).result()
        print(f"  {'CUI':<10} | {'Concept Canonical Name':<35} | {'Semantic Type':<30} | {'Frequency':<10}")
        print(f"  {'-'*10}-+-{'-'*35}-+-{'-'*30}-+-{'-'*10}")
        for r in results:
            print(f"  {r.cui:<10} | {r.name:<35} | {r.semantic_type:<30} | {r.frequency:<10}")
    except Exception as e:
        print(f"  Failed to retrieve concept frequency metrics: {str(e)}")

    # 3. Pipeline Evaluation Metrics Summary
    print("\n3. Pipeline Performance Metrics (Evaluation Summary):")
    print("-----------------------------------------------------")
    try:
        eval_query = f"""
        SELECT 
          template_type,
          COUNT(*) as total_runs,
          ROUND(AVG(rouge1), 4) as avg_rouge1,
          ROUND(AVG(rouge2), 4) as avg_rouge2,
          ROUND(AVG(rougeL), 4) as avg_rougeL,
          ROUND(AVG(creola_cer), 4) as avg_creola_cer,
          ROUND(AVG(entity_f1), 4) as avg_entity_f1
        FROM `{project_id}.{dataset_id}.pipeline_evaluation_metrics`
        GROUP BY template_type
        """
        results = client.query(eval_query).result()
        rows = list(results)
        if rows:
            print(f"  {'Prompt Template':<15} | {'Runs':<5} | {'ROUGE-1':<8} | {'ROUGE-2':<8} | {'ROUGE-L':<8} | {'CREOLA CER':<10} | {'Entity F1':<10}")
            print(f"  {'-'*15}-+-{'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}")
            for r in rows:
                print(f"  {r.template_type:<15} | {r.total_runs:<5} | {r.avg_rouge1:<8} | {r.avg_rouge2:<8} | {r.avg_rougeL:<8} | {r.avg_creola_cer:<10} | {r.avg_entity_f1:<10}")
        else:
            print("  No evaluation metrics recorded yet. Run demo_pipeline.py or the evaluator script.")
    except Exception as e:
        print(f"  Failed to retrieve evaluation metrics: {str(e)}")

    print("\n==========================================================\n")

if __name__ == "__main__":
    main()
