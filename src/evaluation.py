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

import logging
from typing import Dict, List, Set, Any, Optional
from rouge_score import rouge_scorer
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class EvaluationEngine:
    def __init__(self):
        """
        Engine for evaluating clinical summarization metrics.
        """
        pass

    def calculate_rouge_scores(self, reference: str, candidate: str) -> Dict[str, float]:
        """
        Calculates ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.
        
        Args:
            reference: Reference clinical summary (ground truth).
            candidate: Generated clinical summary.
            
        Returns:
            Dictionary containing ROUGE scores: {"rouge1": 0.85, ...}
        """
        if not reference or not candidate:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        
        return {
            "rouge1": float(scores['rouge1'].fmeasure),
            "rouge2": float(scores['rouge2'].fmeasure),
            "rougeL": float(scores['rougeL'].fmeasure)
        }

    def calculate_bertscore(self, reference: str, candidate: str, lang: str = "en") -> Dict[str, float]:
        """
        Calculates BERTScore Precision, Recall, and F1.
        
        Args:
            reference: Reference clinical summary.
            candidate: Generated clinical summary.
            lang: Language code.
            
        Returns:
            Dictionary containing BERTScore: {"precision": 0.9, "recall": 0.9, "f1": 0.9}
        """
        if not reference or not candidate:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        try:
            from bert_score import score
            # Run BERTScore computation
            P, R, F1 = score([candidate], [reference], lang=lang, verbose=False)
            
            return {
                "precision": float(P.mean().item()),
                "recall": float(R.mean().item()),
                "f1": float(F1.mean().item())
            }
        except ImportError:
            logger.warning("bert-score library not installed or cannot import 'score'. Returning 0.0.")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        except Exception as e:
            logger.error(f"Error calculating BERTScore: {str(e)}")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def calculate_creola_cer(self, e_fab: int, e_neg: int, e_cau: int, e_ctx: int, s_total: int) -> float:
        """
        Calculates the CREOLA Clinical Error Rate (CER).
        Formula:
            CER = (5*E_fab + 4*E_neg + 3*E_cau + 2*E_ctx) / S_total
            
        Args:
            e_fab: Count of fabricated clinical information (hallucinations).
            e_neg: Count of clinical negation / contradiction errors.
            e_cau: Count of clinical causal / attribution errors.
            e_ctx: Count of contextual distortion / misinterpretation errors.
            s_total: Total sentence count in the generated summary.
            
        Returns:
            Calculated Clinical Error Rate (float). Returns 0.0 if total sentences is 0.
        """
        if s_total <= 0:
            logger.warning("Total sentence count (S_total) must be greater than 0 to calculate CREOLA CER. Returning 0.0.")
            return 0.0

        penalty_sum = (5 * e_fab) + (4 * e_neg) + (3 * e_cau) + (2 * e_ctx)
        cer = penalty_sum / s_total
        return float(cer)

    def calculate_entity_retrieval_metrics(self, reference_cuis: Set[str], candidate_cuis: Set[str]) -> Dict[str, float]:
        """
        Calculates Entity-level retrieval precision, recall, and F1-score of matching medical concepts.
        
        Args:
            reference_cuis: Set of unique canonical concept IDs (CUIs) in references.
            candidate_cuis: Set of unique canonical concept IDs (CUIs) in the generated candidate.
            
        Returns:
            Dictionary containing {"precision": 0.8, "recall": 0.7, "f1": 0.75}
        """
        if not reference_cuis and not candidate_cuis:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        if not reference_cuis or not candidate_cuis:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        overlap = reference_cuis.intersection(candidate_cuis)
        intersection_count = len(overlap)
        
        precision = intersection_count / len(candidate_cuis)
        recall = intersection_count / len(reference_cuis)
        
        if (precision + recall) == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }

    def write_evaluation_metrics_to_bigquery(
        self,
        hadm_id: int,
        template_type: str,
        rouge_scores: Dict[str, float],
        bertscore_f1: float,
        creola_cer: float,
        entity_f1: float,
        project_id: str = "suddhasatwa-data-projects"
    ) -> None:
        """
        Writes summarization evaluation metrics directly to BigQuery for Looker Studio visualization.
        
        Args:
            hadm_id (int): Unique patient admission identifier.
            template_type (str): Active prompt structure type (standard/cot/got).
            rouge_scores (Dict[str, float]): ROUGE lexical f-measures dictionary.
            bertscore_f1 (float): Contextual BERTScore similarity metric.
            creola_cer (float): Calculated clinical error rate.
            entity_f1 (float): UMLS exact entities match retrieval index.
            project_id (str): Target Google Cloud Project ID.
        """
        import pandas as pd
        from datetime import datetime
        
        row = {
            "hadm_id": int(hadm_id),
            "template_type": str(template_type),
            "rouge1": float(rouge_scores.get("rouge1", 0.0)),
            "rouge2": float(rouge_scores.get("rouge2", 0.0)),
            "rougeL": float(rouge_scores.get("rougeL", 0.0)),
            "bertscore_f1": float(bertscore_f1),
            "creola_cer": float(creola_cer),
            "entity_f1": float(entity_f1),
            "generated_at": datetime.utcnow()
        }
        
        df_bq = pd.DataFrame([row])
        
        try:
            from google.cloud import bigquery
            logger.info(f"Writing evaluation metrics for hadm_id {hadm_id} to BigQuery...")
            client = bigquery.Client(project=project_id)
            table_id = "clinical_summarization_eda.pipeline_evaluation_metrics"
            
            job_config = bigquery.LoadJobConfig(
                write_disposition="WRITE_APPEND"
            )
            
            job = client.load_table_from_dataframe(df_bq, table_id, job_config=job_config)
            job.result()
            logger.info("Successfully wrote evaluation metrics to BigQuery!")
        except Exception as e:
            logger.warning(f"BigQuery export failed (GCP SDK might not be loaded): {str(e)}")


if __name__ == "__main__":
    # Example execution (local debug testing)
    engine = EvaluationEngine()
    ref = "Patient diagnosed with asthma and prescribed albuterol."
    cand = "Patient has acute asthma and was treated with albuterol inhaler."
    
    rouge = engine.calculate_rouge_scores(ref, cand)
    logger.info(f"ROUGE scores: {rouge}")
    
    # Test CER
    cer = engine.calculate_creola_cer(e_fab=0, e_neg=1, e_cau=0, e_ctx=1, s_total=2)
    logger.info(f"CREOLA CER: {cer}")
    
    # Test Entity Retrieval F1
    ref_cuis = {"C0004096", "C0001617"} # Asthma, Albuterol
    cand_cuis = {"C0004096", "C0001617", "C0011991"} # Asthma, Albuterol, Inhaler
    ent_metrics = engine.calculate_entity_retrieval_metrics(ref_cuis, cand_cuis)
    logger.info(f"Entity retrieval: {ent_metrics}")
