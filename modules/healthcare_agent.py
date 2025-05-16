from core.agent import BaseAgent
from core.retriever import HybridRetriever
from core.tools import WebSearchTool
from core.memory import ClinicalMemory
from typing import Dict, Any
import json

class HealthcareAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.specialization = "healthcare"
        self.clinical_retriever = HybridRetriever(
            index_name="clinical_knowledge",
            embedding_model="biobert",
            reranker="colbert"
        )
        self.tools.register_tool("clinical_guidelines_search", self.search_clinical_guidelines)
        self.tools.register_tool("drug_interaction_check", self.check_drug_interactions)
        self.memory = ClinicalMemory()
        
    def search_clinical_guidelines(self, query: str, max_results: int = 5) -> str:
        """Search evidence-based clinical guidelines"""
        results = self.clinical_retriever.search(
            query=query,
            filters={"document_type": "guideline"},
            max_results=max_results
        )
        return json.dumps(results)
    
    def check_drug_interactions(self, medications: list) -> str:
        """Check for potential drug-drug interactions"""
        interaction_data = self.clinical_retriever.search(
            query=",".join(medications),
            filters={"document_type": "drug_interactions"}
        )
        return interaction_data
    
    def process_patient_data(self, patient_record: Dict) -> Dict:
        """Process and analyze patient health records"""
        # HIPAA-compliant processing
        self.memory.store_episodic(patient_record)
        analysis = self.llm.generate(
            prompt=f"Analyze this patient record: {patient_record}",
            temperature=0.1  # Low creativity for medical accuracy
        )
        return {
            "analysis": analysis,
            "followup_questions": self.generate_clinical_questions(patient_record)
        }
    
    def generate_clinical_questions(self, record: Dict) -> list:
        """Generate relevant clinical questions based on patient data"""
        return self.llm.generate(
            prompt="Generate clinical questions for this patient case",
            response_format={"type": "json_object"}
        )
