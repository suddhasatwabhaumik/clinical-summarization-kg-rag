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
import csv
import random
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# List of rich pre-defined clinical templates to scale data programmatically
DEFAULT_TEMPLATES = [
    "Patient {name} is a {age}yo {gender} presenting with {symptom}. Diagnosed with {diagnosis}. Administered {medication}. Status improved.",
    "Patient {name} ({age}yo {gender}) admitted for acute {symptom}. Workup confirmed {diagnosis}. Started on {medication} daily. Follow up in 2 weeks.",
    "The patient is a {age} year old {gender} who presented to the ED with complaints of {symptom}. Clinical findings are consistent with {diagnosis}. Discharge instructions include taking {medication}.",
    "Discharge summary for {name}, a {age}yo {gender} diagnosed with {diagnosis} after experiencing {symptom}. Treated successfully with {medication}.",
    "Subject presented with {symptom} and was admitted. Final diagnosis is {diagnosis}. Patient discharged home on {medication} therapy."
]

# Vocabulary lists for synthetic scaling
MALE_NAMES = ["James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph", "Thomas", "Charles"]
FEMALE_NAMES = ["Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan", "Jessica", "Sarah", "Karen"]
SYMPTOMS = ["shortness of breath", "severe chest pain", "high fever and cough", "severe abdominal pain", "dizziness and confusion", "nausea and vomiting", "acute joint pain", "chronic headache", "blurred vision"]
CLINICAL_MAP = [
    {
        "symptom": "shortness of breath",
        "diagnosis": "acute Asthma",
        "medication": "Albuterol inhaler"
    },
    {
        "symptom": "severe chest pain",
        "diagnosis": "Myocardial Infarction",
        "medication": "Aspirin and Metoprolol"
    },
    {
        "symptom": "high fever and cough",
        "diagnosis": "Bacterial Pneumonia",
        "medication": "Amoxicillin"
    },
    {
        "symptom": "severe abdominal pain",
        "diagnosis": "Acute Appendicitis",
        "medication": "IV Antibiotics and referred for Appendectomy"
    },
    {
        "symptom": "dizziness and confusion",
        "diagnosis": "Dehydration and Electrolyte Imbalance",
        "medication": "IV Fluids"
    },
    {
        "symptom": "chronic joint pain",
        "diagnosis": "Rheumatoid Arthritis",
        "medication": "Methotrexate"
    },
    {
        "symptom": "blurry vision and headache",
        "diagnosis": "Hypertension",
        "medication": "Lisinopril"
    }
]

class SyntheticDataGenerator:
    """
    Generates large synthetic MIMIC-IV notes dataset using seed templates.
    
    Attributes:
        project_id (str): Target Google Cloud Project ID.
        location (str): Region hosted GCP Vertex AI endpoint.
        templates (List[str]): Seed clinical templates dictionary lists.
    """
    
    def __init__(self, project_id: str = "suddhasatwa-data-projects", location: str = "us-central1") -> None:
        """
        Initializes the generator with GCP environment specifications.
        
        Args:
            project_id (str): Target GCP Project ID.
            location (str): Location hosted model deployments.
        """
        self.project_id = project_id
        self.location = location
        self.templates = list(DEFAULT_TEMPLATES)

    def fetch_templates_from_gemini(self) -> None:
        """
        Uses Vertex AI Gemini to generate high-quality clinical templates to seed the generator.
        """
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
            
            logger.info("Initializing Vertex AI connection...")
            vertexai.init(project=self.project_id, location=self.location)
            
            logger.info("Querying Gemini on Vertex AI to generate seed clinical templates...")
            model = GenerativeModel("gemini-1.5-flash")
            
            prompt = """
            Generate 5 different realistic, brief clinical note summary templates. 
            Use placeholders like {name}, {age}, {gender}, {symptom}, {diagnosis}, {medication} inside the text.
            Return ONLY the templates, one per line. No introduction, no formatting.
            """
            
            response = model.generate_content(prompt)
            lines = [line.strip() for line in response.text.split("\n") if line.strip()]
            
            if lines:
                self.templates = lines
                logger.info(f"Successfully loaded {len(self.templates)} clinical templates from Gemini.")
            else:
                logger.warning("Gemini returned empty templates. Falling back to default templates.")
        except Exception as e:
            logger.warning(f"Failed to fetch templates from Gemini (GCP Auth/SDK missing): {str(e)}.")
            logger.info("Proceeding with high-quality local templates.")

    def generate_dataset(self, output_filepath: str, num_rows: int = 200000) -> None:
        """
        Generates and saves num_rows of synthetic summaries to output_filepath.
        
        Args:
            output_filepath (str): Target output file path to write CSV dataset.
            num_rows (int): Total number of rows to scale output (default: 200000).
        """
        # Ensure parent directories exist
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        
        logger.info(f"Generating {num_rows} synthetic clinical note rows...")
        
        with open(output_filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write MIMIC-IVNote schema headers
            writer.writerow(["note_id", "subject_id", "hadm_id", "note_type", "text"])
            
            # Start generating rows
            for i in range(num_rows):
                note_id = 9000000 + i
                subject_id = 50000 + i
                hadm_id = 100000 + i
                
                # Pick gender and name
                gender = random.choice(["male", "female"])
                name = random.choice(MALE_NAMES) if gender == "male" else random.choice(FEMALE_NAMES)
                age = random.randint(18, 90)
                
                # Pick clinical mapping
                clinical = random.choice(CLINICAL_MAP)
                symptom = clinical["symptom"]
                diagnosis = clinical["diagnosis"]
                medication = clinical["medication"]
                
                # Format a random template
                template = random.choice(self.templates)
                text = template.format(
                    name=name,
                    age=age,
                    gender=gender,
                    symptom=symptom,
                    diagnosis=diagnosis,
                    medication=medication
                )
                
                writer.writerow([note_id, subject_id, hadm_id, "Discharge Summary", text])
                
                if (i + 1) % 50000 == 0:
                    logger.info(f"Generated {i + 1} rows...")

        logger.info(f"Dataset successfully created and saved to {output_filepath}!")

if __name__ == "__main__":
    import sys
    
    num_rows = 200000
    if len(sys.argv) > 1:
        try:
            num_rows = int(sys.argv[1])
        except ValueError:
            print("Invalid row count. Using default 200,000.")
            
    output_path = "data/raw/discharge.csv"
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
        
    generator = SyntheticDataGenerator()
    generator.fetch_templates_from_gemini()
    generator.generate_dataset(output_path, num_rows)
