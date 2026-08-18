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
from typing import List, Dict, Any, Optional
import pandas as pd
from google.cloud import bigquery

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Whitelist of relationship types to protect against injection via dynamic relationship labels
ALLOWED_RELATIONS = {
    "TREATS",
    "CAUSES",
    "HAS_SYMPTOM",
    "ASSOCIATED_WITH",
    "PREVENTS",
    "DIAGNOSES",
    "INDICATES",
    "CONTRAINDICATES",
    "IS_A"
}

class GraphBuilder:
    """
    Orchestrates ingestion of medical concepts and edges into BigQuery Graph Store.
    
    Attributes:
        project_id (str): Target Google Cloud Project ID.
        dataset_id (str): Target BigQuery dataset.
    """
    
    def __init__(self, dataset_id: str = "clinical_summarization_eda", project_id: Optional[str] = None) -> None:
        """
        Initializes the BigQuery Graph database client connector.
        
        Args:
            dataset_id (str): BigQuery dataset identifier.
            project_id (Optional[str]): GCP Project ID. Resolves from env if empty.
        """
        self.project_id = project_id or os.environ.get("GCP_PROJECT", "suddhasatwa-data-projects")
        self.dataset_id = dataset_id
        self.client: Optional[bigquery.Client] = None

    def connect(self) -> None:
        """
        Establishes connection to the BigQuery client.
        """
        if self.client is not None:
            return
            
        try:
            logger.info(f"Connecting to BigQuery project '{self.project_id}' dataset '{self.dataset_id}'...")
            self.client = bigquery.Client(project=self.project_id)
            logger.info("Connected successfully to BigQuery.")
        except Exception as e:
            logger.error(f"Failed to connect to BigQuery: {str(e)}")
            raise

    def close(self) -> None:
        """
        Closes the BigQuery client (no-op for BigQuery REST client).
        """
        self.client = None
        logger.info("BigQuery client reference cleared.")

    def create_constraints_and_indexes(self) -> None:
        """
        Verifies that graph tables exist in the BigQuery dataset.
        """
        self.connect()
        logger.info("Verifying BigQuery graph tables are provisioned...")
        try:
            self.client.get_table(f"{self.project_id}.{self.dataset_id}.graph_nodes")
            self.client.get_table(f"{self.project_id}.{self.dataset_id}.graph_edges")
            logger.info("Graph tables successfully verified!")
        except Exception as e:
            logger.error(f"Failed to verify tables. Ensure Terraform has provisioned them: {str(e)}")
            raise

    def add_concept_node(self, cui: str, name: str, types: List[str]) -> None:
        """
        Creates or updates a Concept node safely inside BigQuery using a MERGE query.
        
        Args:
            cui (str): UMLS concept unique identifier.
            name (str): Canonical term label name.
            types (List[str]): Semantic types categorizations.
        """
        query = f"""
        MERGE `{self.project_id}.{self.dataset_id}.graph_nodes` T
        USING (SELECT @cui AS cui, @name AS name, @types AS types) S
        ON T.cui = S.cui
        WHEN MATCHED THEN
          UPDATE SET T.name = S.name, T.types = S.types
        WHEN NOT MATCHED THEN
          INSERT (cui, name, types) VALUES (S.cui, S.name, S.types)
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("cui", "STRING", cui),
                bigquery.ScalarQueryParameter("name", "STRING", name),
                bigquery.ArrayQueryParameter("types", "STRING", types)
            ]
        )
        self.client.query(query, job_config=job_config).result()

    def add_relationship(self, cui_a: str, cui_b: str, rel_type: str) -> None:
        """
        Creates a relationship between two Concept nodes safely in the edges table using a MERGE query.
        
        Args:
            cui_a (str): CUI of the starting concept node.
            cui_b (str): CUI of the target concept node.
            rel_type (str): Relationship label type name.
        """
        # Validate relationship type
        normalized_rel = rel_type.upper().replace(" ", "_")
        if normalized_rel not in ALLOWED_RELATIONS:
            logger.warning(f"Relationship type '{rel_type}' not in whitelist. Defaulting to 'ASSOCIATED_WITH'.")
            normalized_rel = "ASSOCIATED_WITH"

        query = f"""
        MERGE `{self.project_id}.{self.dataset_id}.graph_edges` T
        USING (SELECT @cui_from AS cui_from, @cui_to AS cui_to, @rel_type AS rel_type, CURRENT_TIMESTAMP() AS updated_at) S
        ON T.cui_from = S.cui_from AND T.cui_to = S.cui_to
        WHEN NOT MATCHED THEN
          INSERT (cui_from, cui_to, rel_type, updated_at) VALUES (S.cui_from, S.cui_to, S.rel_type, S.updated_at)
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("cui_from", "STRING", cui_a),
                bigquery.ScalarQueryParameter("cui_to", "STRING", cui_b),
                bigquery.ScalarQueryParameter("rel_type", "STRING", normalized_rel)
            ]
        )
        self.client.query(query, job_config=job_config).result()

    def ingest_entities(self, note_entities: List[Dict[str, Any]]) -> None:
        """
        Ingests extracted entities from clinical notes into the BigQuery graph tables.
        """
        self.connect()
        # 1. Create/Merge nodes
        for ent in note_entities:
            cui = ent.get("cui")
            name = ent.get("name")
            types = ent.get("types", [])
            if cui and name:
                self.add_concept_node(cui, name, types)

        # 2. Build co-occurrence relationships within the note (ASSOCIATED_WITH)
        cuis = [ent.get("cui") for ent in note_entities if ent.get("cui")]
        unique_cuis = list(set(cuis))
        
        # Link sequential pairs
        for i in range(len(unique_cuis) - 1):
            self.add_relationship(unique_cuis[i], unique_cuis[i+1], "ASSOCIATED_WITH")

    def ingest_bulk_entities(self, all_notes_entities: List[List[Dict[str, Any]]]) -> None:
        """
        High-performance bulk ingestion of entities and co-occurrences into BigQuery.
        Collects all unique nodes/edges in memory and merges them in a single batch load.
        """
        self.connect()
        logger.info(f"Preparing bulk ingestion for {len(all_notes_entities)} notes...")

        nodes_map = {}
        edges_set = set()

        for note_entities in all_notes_entities:
            # 1. Collect nodes
            for ent in note_entities:
                cui = ent.get("cui")
                name = ent.get("name")
                types = ent.get("types", [])
                if cui and name:
                    if cui not in nodes_map:
                        nodes_map[cui] = {
                            "cui": cui,
                            "name": name,
                            "types": set(types)
                        }
                    else:
                        nodes_map[cui]["types"].update(types)

            # 2. Collect co-occurrence relationships (edges)
            cuis = [ent.get("cui") for ent in note_entities if ent.get("cui")]
            unique_cuis = list(set(cuis))
            for i in range(len(unique_cuis) - 1):
                cui_from = unique_cuis[i]
                cui_to = unique_cuis[i+1]
                if cui_from > cui_to:
                    cui_from, cui_to = cui_to, cui_from
                edges_set.add((cui_from, cui_to, "ASSOCIATED_WITH"))

        if not nodes_map:
            logger.info("No nodes to ingest.")
            return

        logger.info(f"Unique nodes to ingest: {len(nodes_map)}")
        logger.info(f"Unique edges to ingest: {len(edges_set)}")

        # Convert to DataFrames
        nodes_rows = []
        for cui, details in nodes_map.items():
            nodes_rows.append({
                "cui": cui,
                "name": details["name"],
                "types": list(details["types"])
            })
        nodes_df = pd.DataFrame(nodes_rows)

        edges_rows = []
        for cui_from, cui_to, rel_type in edges_set:
            edges_rows.append({
                "cui_from": cui_from,
                "cui_to": cui_to,
                "rel_type": rel_type
            })
        edges_df = pd.DataFrame(edges_rows)

        # Upload staging tables and run MERGE
        nodes_staging_id = f"{self.project_id}.{self.dataset_id}.graph_nodes_staging"
        edges_staging_id = f"{self.project_id}.{self.dataset_id}.graph_edges_staging"

        try:
            # Stage Nodes
            logger.info("Uploading nodes staging table...")
            nodes_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
            nodes_job = self.client.load_table_from_dataframe(nodes_df, nodes_staging_id, job_config=nodes_config)
            nodes_job.result()

            # Merge Nodes
            logger.info("Merging staging nodes into production graph_nodes...")
            nodes_merge_query = f"""
            MERGE `{self.project_id}.{self.dataset_id}.graph_nodes` T
            USING (
              SELECT 
                S.cui,
                S.name,
                IF(T.cui IS NOT NULL, ARRAY(SELECT DISTINCT x FROM UNNEST(ARRAY_CONCAT(T.types, S.types)) x), S.types) AS types
              FROM `{nodes_staging_id}` S
              LEFT JOIN `{self.project_id}.{self.dataset_id}.graph_nodes` T
              ON S.cui = T.cui
            ) S
            ON T.cui = S.cui
            WHEN MATCHED THEN
              UPDATE SET T.name = S.name, T.types = S.types
            WHEN NOT MATCHED THEN
              INSERT (cui, name, types) VALUES (S.cui, S.name, S.types)
            """
            self.client.query(nodes_merge_query).result()

            # Stage Edges
            logger.info("Uploading edges staging table...")
            edges_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
            edges_job = self.client.load_table_from_dataframe(edges_df, edges_staging_id, job_config=edges_config)
            edges_job.result()

            # Merge Edges
            logger.info("Merging staging edges into production graph_edges...")
            edges_merge_query = f"""
            MERGE `{self.project_id}.{self.dataset_id}.graph_edges` T
            USING `{edges_staging_id}` S
            ON (T.cui_from = S.cui_from AND T.cui_to = S.cui_to) OR (T.cui_from = S.cui_to AND T.cui_to = S.cui_from)
            WHEN NOT MATCHED THEN
              INSERT (cui_from, cui_to, rel_type, updated_at) VALUES (S.cui_from, S.cui_to, S.rel_type, CURRENT_TIMESTAMP())
            """
            self.client.query(edges_merge_query).result()

            logger.info("Bulk merge E2E ingestion successfully finished!")

        except Exception as e:
            logger.error(f"Bulk ingestion failed: {str(e)}")
            raise
        finally:
            # Clean staging tables
            logger.info("Cleaning staging tables...")
            self.client.delete_table(nodes_staging_id, not_found_ok=True)
            self.client.delete_table(edges_staging_id, not_found_ok=True)

    def ingest_predefined_relationships(self, relationships: List[Dict[str, str]]) -> None:
        """
        Bulk ingests explicit clinical links into BigQuery tables.
        """
        self.connect()
        for item in relationships:
            cui_from = item.get("cui_from")
            cui_to = item.get("cui_to")
            rel = item.get("rel", "ASSOCIATED_WITH")
            
            if cui_from and cui_to:
                # Make sure nodes exist
                self.add_concept_node(cui_from, f"Concept_{cui_from}", [])
                self.add_concept_node(cui_to, f"Concept_{cui_to}", [])
                # Add edge
                self.add_relationship(cui_from, cui_to, rel)

    def get_2_hop_subgraph(self, concept_cui: str) -> List[Dict[str, Any]]:
        """
        Retrieves the 2-hop community subgraph around a starting concept using SQL joins.
        """
        self.connect()
        query = f"""
        WITH hop1_norm AS (
          SELECT 
            IF(cui_from = @cui, cui_from, cui_to) AS start_cui,
            IF(cui_from = @cui, cui_to, cui_from) AS neighbor_cui,
            rel_type AS rel1
          FROM `{self.project_id}.{self.dataset_id}.graph_edges`
          WHERE cui_from = @cui OR cui_to = @cui
        ),
        hop2 AS (
          SELECT 
            h.start_cui,
            h.neighbor_cui AS hop1_cui,
            IF(e.cui_from = h.neighbor_cui, e.cui_to, e.cui_from) AS hop2_cui,
            h.rel1,
            e.rel_type AS rel2
          FROM hop1_norm h
          JOIN `{self.project_id}.{self.dataset_id}.graph_edges` e
            ON (e.cui_from = h.neighbor_cui AND e.cui_to != h.start_cui)
            OR (e.cui_to = h.neighbor_cui AND e.cui_from != h.start_cui)
        )
        SELECT 
          n1.name AS start_name,
          h.start_cui,
          [n1.name, n2.name, n3.name] AS node_names,
          [h.start_cui, h.hop1_cui, h.hop2_cui] AS node_cuis,
          [h.rel1, h.rel2] AS rel_types
        FROM hop2 h
        JOIN `{self.project_id}.{self.dataset_id}.graph_nodes` n1 ON n1.cui = h.start_cui
        JOIN `{self.project_id}.{self.dataset_id}.graph_nodes` n2 ON n2.cui = h.hop1_cui
        JOIN `{self.project_id}.{self.dataset_id}.graph_nodes` n3 ON n3.cui = h.hop2_cui
        
        UNION ALL
        
        SELECT 
          n1.name AS start_name,
          h.start_cui,
          [n1.name, n2.name] AS node_names,
          [h.start_cui, h.neighbor_cui] AS node_cuis,
          [h.rel1] AS rel_types
        FROM hop1_norm h
        JOIN `{self.project_id}.{self.dataset_id}.graph_nodes` n1 ON n1.cui = h.start_cui
        JOIN `{self.project_id}.{self.dataset_id}.graph_nodes` n2 ON n2.cui = h.neighbor_cui
        LIMIT 25
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("cui", "STRING", concept_cui)
            ]
        )
        
        subgraph_paths = []
        result = self.client.query(query, job_config=job_config).result()
        for record in result:
            subgraph_paths.append({
                "start_name": record.get("start_name"),
                "start_cui": record.get("start_cui"),
                "nodes": record.get("node_names"),
                "cuis": record.get("node_cuis"),
                "relations": record.get("rel_types")
            })
        return subgraph_paths

if __name__ == "__main__":
    # Example execution (local debug testing)
    builder = GraphBuilder()
    try:
        builder.connect()
        builder.create_constraints_and_indexes()
        logger.info("BigQuery graph configuration initialized successfully.")
    except Exception as e:
        logger.error(f"Error during graph configuration: {str(e)}")
    finally:
        builder.close()
