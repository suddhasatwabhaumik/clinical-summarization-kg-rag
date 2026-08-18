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
import re
import json
import logging
import pandas as pd
import spacy
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, spacy_model_name: str = "en_core_sci_sm"):
        """
        Initializes the DataProcessor with a scispacy model.
        
        Args:
            spacy_model_name: Name of the scispacy/spaCy model to use for entity extraction.
        """
        self.spacy_model_name = spacy_model_name
        self.nlp = None
        self.linker = None

    def load_nlp_pipeline(self) -> None:
        """
        Loads the spaCy pipeline and UMLS entity linker.
        Defers loading until needed to save resources.
        
        Raises:
            OSError: If the scispacy model is not installed.
            ImportError: If scispacy library is not installed or EntityLinker cannot be imported.
        """
        if self.nlp is not None or os.environ.get("SKIP_SCISPACY") == "True":
            return
            
        logger.info(f"Loading spaCy model: {self.spacy_model_name}...")
        try:
            self.nlp = spacy.load(self.spacy_model_name)
        except OSError:
            logger.error(
                f"Model '{self.spacy_model_name}' not found. Please install it first. "
                f"For example: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/{self.spacy_model_name}-0.5.4.tar.gz"
            )
            raise

        logger.info("Adding UMLS EntityLinker to pipeline...")
        # Check scispacy import to avoid import errors when running in environments without scispacy installed yet
        try:
            from scispacy.linking import EntityLinker
            # Add the entity linker if not already present
            if "scispacy_linker" not in self.nlp.pipe_names:
                self.nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
            self.linker = self.nlp.get_pipe("scispacy_linker")
        except ImportError:
            logger.error("scispacy library is not installed or EntityLinker cannot be imported.")
            raise

    def clean_text(self, text: Any) -> str:
        """
        Cleans unstructured clinical text.
        Removes de-identification brackets (e.g., [** ... **]), normalizes line breaks, and whitespace.
        
        Args:
            text: Raw string input from clinical notes.
            
        Returns:
            Cleaned and normalized text string.
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""

        # Remove de-identification brackets like [** ... **]
        # Matches brackets and anything inside them up to the closing brackets
        cleaned = re.sub(r'\[\*\*.*?\*\*\]', '', text)
        
        # Replace multiple consecutive spaces/tabs with a single space
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        
        # Normalize multiple newlines to a double newline (separating paragraphs/sections)
        cleaned = re.sub(r'\n+', '\n', cleaned)
        
        return cleaned.strip()

    def _extract_entities_mock(self, text: str) -> List[Dict[str, Any]]:
        """
        Fallback mock entity extractor when scispacy is bypassed.
        Uses basic keyword/capitalized noun extraction and hashes terms deterministically to CUIs.
        """
        import hashlib
        predefined = {
            "asthma": ("C0004096", "Asthma", ["Disease or Syndrome"]),
            "albuterol": ("C0001617", "Albuterol", ["Pharmacologic Substance"]),
            "myocardial": ("C0027051", "Myocardial Infarction", ["Disease or Syndrome"]),
            "infarction": ("C0027051", "Myocardial Infarction", ["Disease or Syndrome"]),
            "aspirin": ("C0004057", "Aspirin", ["Pharmacologic Substance"]),
            "metoprolol": ("C0026827", "Metoprolol", ["Pharmacologic Substance"]),
            "cardiology": ("C0007189", "Cardiology Specialty", ["Specialty"]),
            "dyspnea": ("C0013404", "Dyspnea", ["Sign or Symptom"]),
            "pain": ("C0030193", "Pain", ["Sign or Symptom"]),
            "shortness": ("C0013404", "Dyspnea", ["Sign or Symptom"]),
            "breath": ("C0013404", "Dyspnea", ["Sign or Symptom"])
        }
        
        words = re.findall(r'\b[A-Za-z]{4,}\b', text)
        extracted = []
        seen_cuis = set()
        
        for word in words:
            word_lower = word.lower()
            if word_lower in predefined:
                cui, canonical_name, sem_types = predefined[word_lower]
                if cui not in seen_cuis:
                    seen_cuis.add(cui)
                    extracted.append({
                        "text": word,
                        "cui": cui,
                        "name": canonical_name,
                        "score": 1.0,
                        "types": sem_types
                    })
            elif word[0].isupper() and word_lower not in {"patient", "administered", "referred", "status", "improved", "prescribed", "severe", "diagnosed", "presenting", "presented"}:
                h = hashlib.md5(word_lower.encode()).hexdigest()
                cui = f"C{int(h[:6], 16):07d}"[:8]
                if cui not in seen_cuis:
                    seen_cuis.add(cui)
                    extracted.append({
                        "text": word,
                        "cui": cui,
                        "name": word.capitalize(),
                        "score": 0.8,
                        "types": ["Biomedical Concept"]
                    })
        return extracted

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extracts biomedical concepts and links them to UMLS CUIs using scispacy.
        
        Args:
            text: Cleaned clinical text.
            
        Returns:
            A list of dictionaries representing extracted entities:
            [{"text": "...", "cui": "...", "name": "...", "score": 0.9, "types": [...]}]
        """
        if not text:
            return []

        if os.environ.get("SKIP_SCISPACY") == "True":
            return self._extract_entities_mock(text)

        self.load_nlp_pipeline()
        doc = self.nlp(text)
        
        extracted_entities = []
        for ent in doc.ents:
            # Check if entity has linked UMLS concepts
            if ent._.kb_ents:
                # Get the highest-scoring candidate concept
                best_match = ent._.kb_ents[0]
                cui, score = best_match
                
                # Fetch detailed concept info from the linker knowledge base
                concept = self.linker.kb.cui_to_entity.get(cui)
                if concept:
                    canonical_name = concept.canonical_name
                    semantic_types = list(concept.types)
                    
                    extracted_entities.append({
                        "text": ent.text,
                        "cui": cui,
                        "name": canonical_name,
                        "score": float(score),
                        "types": semantic_types,
                        "start_char": ent.start_char,
                        "end_char": ent.end_char
                    })
                    
        return extracted_entities

    def process_csv(self, input_filepath: str, output_filepath: str) -> None:
        """
        Loads clinical notes from input_filepath, cleans texts, extracts entities, and saves processed file.
        
        Args:
            input_filepath: Path to raw discharge.csv
            output_filepath: Path to save processed CSV or JSON.
        """
        if not input_filepath.startswith("gs://") and not os.path.exists(input_filepath):
            logger.error(f"Input file not found at {input_filepath}")
            raise FileNotFoundError(f"Input file not found at {input_filepath}")

        logger.info(f"Loading data from {input_filepath}...")
        df = pd.read_csv(input_filepath)

        logger.info(f"Loaded dataset with {len(df)} rows.")

        # Check for missing hadm_id rows
        if 'hadm_id' not in df.columns:
            logger.error("Dataset lacks required 'hadm_id' column.")
            raise ValueError("Dataset lacks required 'hadm_id' column.")

        # Handle rows with missing hadm_id
        missing_hadm = df['hadm_id'].isna()
        if missing_hadm.any():
            num_missing = missing_hadm.sum()
            logger.warning(f"Found {num_missing} rows with missing 'hadm_id'. Removing them.")
            df = df.dropna(subset=['hadm_id'])
            # Convert to int/str representation safely
            df['hadm_id'] = df['hadm_id'].astype(int)

        # We assume the unstructured text column is named 'text' or 'description' or 'note'
        text_col = None
        for col in ['text', 'note', 'description', 'notes']:
            if col in df.columns:
                text_col = col
                break

        if not text_col:
            logger.error("Could not find a text/note column (checked 'text', 'note', 'description', 'notes').")
            raise ValueError("No text column found in CSV.")

        logger.info(f"Using column '{text_col}' for clinical notes.")

        # Clean text
        logger.info("Cleaning clinical notes...")
        df['cleaned_text'] = df[text_col].apply(self.clean_text)

        # Extract entities row-by-row
        logger.info("Extracting UMLS concepts from cleaned texts (this might take a few minutes)...")
        self.load_nlp_pipeline()

        entities_list = []
        for idx, row in df.iterrows():
            if idx % 100 == 0 and idx > 0:
                logger.info(f"Processed {idx} rows...")
            
            entities = self.extract_entities(row['cleaned_text'])
            entities_list.append(json.dumps(entities))

        df['entities'] = entities_list

        # Create output directory if it does not exist (only for local outputs)
        if not output_filepath.startswith("gs://"):
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        
        # Save output
        logger.info(f"Saving processed dataset to {output_filepath}...")
        df.to_csv(output_filepath, index=False)
        logger.info("Processing complete!")
        
        # Automatically load entity metrics to BigQuery
        project_id = os.environ.get("GCP_PROJECT", "suddhasatwa-data-projects")
        self.write_entity_frequencies_to_bigquery(output_filepath, project_id=project_id)
        self.write_knowledge_graph_to_bigquery(output_filepath, project_id=project_id)



    def write_entity_frequencies_to_bigquery(self, processed_filepath: str, project_id: str = "suddhasatwa-data-projects") -> None:
        """
        Calculates entity frequencies from processed CSV and writes them to BigQuery.
        
        Args:
            processed_filepath (str): Path to the processed CSV file.
            project_id (str): Target Google Cloud Project ID.
        """
        logger.info("Reading processed data to aggregate entity frequencies...")
        try:
            df = pd.read_csv(processed_filepath)
        except Exception as e:
            logger.error(f"Failed to read processed file from {processed_filepath}: {str(e)}")
            return

        if 'entities' not in df.columns:
            logger.error("No 'entities' column found in processed data.")
            return

        # Count frequencies
        from collections import Counter
        from datetime import datetime
        
        entity_counter = Counter()
        entity_info = {} # Maps CUI -> (Name, Semantic Type)

        for _, row in df.iterrows():
            if pd.isna(row['entities']):
                continue
            try:
                entities = json.loads(row['entities'])
                for ent in entities:
                    cui = ent.get("cui")
                    name = ent.get("name")
                    types = ent.get("types", [])
                    if cui and name:
                        entity_counter[cui] += 1
                        sem_type = types[0] if types else "Unknown"
                        entity_info[cui] = (name, sem_type)
            except Exception as e:
                logger.warning(f"Error parsing entity row: {str(e)}")

        if not entity_counter:
            logger.warning("No valid entities found to write to BigQuery.")
            return

        # Construct rows
        rows = []
        for cui, freq in entity_counter.items():
            name, sem_type = entity_info[cui]
            # BigQuery expects timestamps in UTC datetime objects or strings
            rows.append({
                "cui": cui,
                "name": name,
                "semantic_type": sem_type,
                "frequency": freq,
                "last_updated": datetime.utcnow()
            })

        df_bq = pd.DataFrame(rows)

        try:
            from google.cloud import bigquery
            logger.info(f"Writing entity frequencies to BigQuery table clinical_summarization_eda.eda_entity_frequencies in project '{project_id}'...")
            client = bigquery.Client(project=project_id)
            table_id = "clinical_summarization_eda.eda_entity_frequencies"
            
            job_config = bigquery.LoadJobConfig(
                write_disposition="WRITE_APPEND"
            )
            
            job = client.load_table_from_dataframe(df_bq, table_id, job_config=job_config)
            job.result()
            logger.info("Successfully wrote EDA statistics to BigQuery!")
        except Exception as e:
            logger.warning(f"BigQuery export failed (GCP SDK might not be loaded): {str(e)}")


    def write_knowledge_graph_to_bigquery(self, processed_filepath: str, project_id: str = "suddhasatwa-data-projects") -> None:
        """
        Loads processed entities from processed CSV and ingests them into the BigQuery Graph nodes/edges.
        
        Args:
            processed_filepath (str): Path to the processed CSV file.
            project_id (str): Target Google Cloud Project ID.
        """
        logger.info("Ingesting entities and relationships into BigQuery Knowledge Graph...")
        try:
            df = pd.read_csv(processed_filepath)
        except Exception as e:
            logger.error(f"Failed to read processed file from {processed_filepath}: {str(e)}")
            return

        if 'entities' not in df.columns:
            logger.error("No 'entities' column found in processed data.")
            return

        from src.graph_builder import GraphBuilder
        builder = GraphBuilder(project_id=project_id)
        try:
            builder.connect()
            builder.create_constraints_and_indexes()
            
            all_notes_entities = []
            for _, row in df.iterrows():
                if pd.isna(row['entities']):
                    continue
                try:
                    entities = json.loads(row['entities'])
                    all_notes_entities.append(entities)
                except Exception as e:
                    logger.warning(f"Failed to parse row entities: {str(e)}")
                    
            builder.ingest_bulk_entities(all_notes_entities)
            logger.info("Ingestion to BigQuery Knowledge Graph complete!")
        except Exception as e:
            logger.error(f"GraphBuilder ingestion failed: {str(e)}")
        finally:
            builder.close()



if __name__ == "__main__":
    # Example execution (local debug testing)
    import sys
    if len(sys.argv) < 3:
        print("Usage: python data_processor.py <input_csv> <output_csv>")
    else:
        processor = DataProcessor()
        processor.process_csv(sys.argv[1], sys.argv[2])
