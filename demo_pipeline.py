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
Clinical Summarization KG-RAG Demo pipeline execution.

Runs a simple E2E mock pipeline to verify ingestion, database population, and
KG-RAG path retrieval modules local operations.
"""

import os
import json
import pandas as pd
import logging
from src.data_processor import DataProcessor
from src.graph_builder import GraphBuilder
from src.rag_engine import RAGEngine
from src.evaluation import EvaluationEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main() -> None:
    """
    Executes mock clinical note preprocessing, vector indexing, and graph traversal.
    """
    logger.info("=========================================")
    logger.info("STARTING CLINICAL SUMMARIZATION KG-RAG DEMO")
    logger.info("=========================================")

    # Define paths
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    raw_csv = os.path.join(raw_dir, "discharge.csv")
    processed_csv = os.path.join(processed_dir, "processed_data.csv")

    # 1. Create a mock MIMIC-IV discharge notes dataset if it doesn't exist
    logger.info("Step 1: Creating mock MIMIC-IV notes dataset...")
    mock_data = {
        "hadm_id": [10001, 10002],
        "text": [
            "Patient [** Name **] is a 65yo male presenting with shortness of breath. Diagnosed with acute Asthma. Administered Albuterol inhaler. Status improved.",
            "Patient presented with severe chest pain. Diagnosed with Myocardial Infarction. Prescribed Aspirin and Metoprolol. Referred to cardiology."
        ]
    }
    df_mock = pd.DataFrame(mock_data)
    df_mock.to_csv(raw_csv, index=False)
    logger.info(f"Created mock file at {raw_csv}")

    # 2. Run Data Processor
    logger.info("\nStep 2: Running Data Processor...")
    processor = DataProcessor()
    
    # Let's override extract_entities if scispacy is not fully installed/downloaded
    # to make the demo runnable immediately.
    try:
        processor.load_nlp_pipeline()
    except Exception as e:
        logger.warning("Could not initialize full scispacy UMLS pipeline. Using fallback mock entity extractor.")
        def mock_extract(text: str):
            entities = []
            if "Asthma" in text:
                entities.append({"text": "Asthma", "cui": "C0004096", "name": "Asthma", "types": ["Disease or Syndrome"]})
            if "Albuterol" in text:
                entities.append({"text": "Albuterol", "cui": "C0001617", "name": "Albuterol", "types": ["Pharmacologic Substance"]})
            if "Myocardial Infarction" in text:
                entities.append({"text": "Myocardial Infarction", "cui": "C0027051", "name": "Myocardial Infarction", "types": ["Disease or Syndrome"]})
            if "Aspirin" in text:
                entities.append({"text": "Aspirin", "cui": "C0004057", "name": "Aspirin", "types": ["Pharmacologic Substance"]})
            return entities
        processor.extract_entities = mock_extract

    processor.process_csv(raw_csv, processed_csv)

    # 3. Build Graph (BigQuery)
    logger.info("\nStep 3: Ingesting into BigQuery Knowledge Graph...")
    graph_builder = GraphBuilder()
    
    # Try connecting to BigQuery. If it fails, we will mock graph traversal.
    graph_active = False
    try:
        graph_builder.connect()
        graph_builder.create_constraints_and_indexes()
        
        # Load processed data and ingest
        df_processed = pd.read_csv(processed_csv)
        for idx, row in df_processed.iterrows():
            entities = json.loads(row['entities'])
            graph_builder.ingest_entities(entities)
            
        logger.info("Successfully populated BigQuery database.")
        graph_active = True
    except Exception as e:
        logger.warning(f"BigQuery database connection failed: {str(e)}")
        logger.warning("The demo will proceed using mocked graph paths.")

    # 4. RAG Engine Indexing & Retrieval
    logger.info("\nStep 4: Running RAG Engine indexing and retrieval...")
    
    # Use a basic default embedding function for the local demo to avoid downloading large models
    # if sentence-transformers is not yet ready.
    rag = RAGEngine(persist_dir="data/processed/chroma_db", embedding_model="bio_clinicalbert")
    
    # Load processed data
    df_processed = pd.read_csv(processed_csv)
    
    # Index notes
    for idx, row in df_processed.iterrows():
        rag.index_clinical_note(hadm_id=int(row['hadm_id']), note_text=row['cleaned_text'])

    # Retrieve semantic chunks
    query = "treatments for Asthma"
    logger.info(f"Querying ChromaDB for: '{query}'...")
    semantic_chunks = rag.retrieve_semantic_context(query, top_k=2)
    logger.info(f"Retrieved Chunks: {semantic_chunks}")

    # Graph Retrieval
    logger.info("Retrieving Graph Paths...")
    if graph_active:
        paths = graph_builder.get_2_hop_subgraph("C0004096") # CUI for Asthma
    else:
        # Mock paths since BigQuery is offline
        paths = [{
            "start_name": "Asthma",
            "start_cui": "C0004096",
            "nodes": ["Asthma", "Albuterol"],
            "cuis": ["C0004096", "C0001617"],
            "relations": ["ASSOCIATED_WITH"]
        }]
    
    formatted_paths = rag.format_graph_paths(paths)
    logger.info(f"Logical paths found:\n{formatted_paths}")

    # Generate prompts
    logger.info("\nGenerating Prompt templates...")
    prompts = rag.get_prompt_templates()
    
    standard_prompt = prompts["standard"].format(
        context="\n".join(semantic_chunks),
        clinical_notes=df_processed.iloc[0]['cleaned_text']
    )
    got_prompt = prompts["got"].format(
        context="\n".join(semantic_chunks),
        clinical_notes=df_processed.iloc[0]['cleaned_text'],
        graph_paths=formatted_paths
    )

    logger.info("--- Graph-of-Thought (GoT) Prompt Preview ---")
    print(got_prompt)

    # Close graph connection if open
    if graph_active:
        graph_builder.close()

    # 5. Evaluation metrics
    logger.info("\nStep 5: Testing Evaluation Engine...")
    eval_engine = EvaluationEngine()
    
    reference_summary = "A 65yo male presented with shortness of breath. Diagnosed with acute asthma and treated with Albuterol."
    candidate_summary = "Patient presented with shortness of breath and was diagnosed with asthma. He was prescribed Albuterol."

    rouge_scores = eval_engine.calculate_rouge_scores(reference_summary, candidate_summary)
    logger.info(f"ROUGE scores: {rouge_scores}")

    # CREOLA CER
    # Let's assume we identified 1 negation error in a summary of 5 sentences
    cer = eval_engine.calculate_creola_cer(e_fab=0, e_neg=1, e_cau=0, e_ctx=0, s_total=5)
    logger.info(f"CREOLA Clinical Error Rate (CER): {cer}")

    # Entity F1
    ref_entities = {"C0004096", "C0001617"} # Asthma, Albuterol
    cand_entities = {"C0004096", "C0001617"} # Asthma, Albuterol
    ent_f1 = eval_engine.calculate_entity_retrieval_metrics(ref_entities, cand_entities)
    logger.info(f"Entity F1 Metrics: {ent_f1}")

    # Write evaluation metrics to BigQuery
    try:
        project_id = os.environ.get("GCP_PROJECT", "suddhasatwa-data-projects")
        eval_engine.write_evaluation_metrics_to_bigquery(
            hadm_id=10001,
            template_type="got",
            rouge_scores=rouge_scores,
            bertscore_f1=0.85,
            creola_cer=cer,
            entity_f1=ent_f1["f1"],
            project_id=project_id
        )
    except Exception as e:
        logger.warning(f"Failed to write evaluation metrics to BigQuery: {str(e)}")

    logger.info("=========================================")
    logger.info("DEMO COMPLETED SUCCESSFULLY!")
    logger.info("=========================================")

if __name__ == "__main__":
    main()
