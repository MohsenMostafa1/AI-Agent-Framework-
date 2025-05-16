from core.agent import BaseAgent
from core.tools import ProductSearchTool, RecommendationEngine
from core.memory import CustomerMemory
import pandas as pd

class ECommerceAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.specialization = "ecommerce"
        self.tools.register_tool("analyze_sales_data", self.analyze_sales)
        self.tools.register_tool("generate_product_descriptions", self.generate_descriptions)
        self.tools.register_tool("personalize_recommendations", self.personalize_recommendations)
        self.memory = CustomerMemory()
        
    def analyze_sales(self, sales_data: pd.DataFrame) -> Dict:
        """Analyze sales data and identify trends"""
        analysis = self.llm.generate(
            prompt=f"Analyze this sales data: {sales_data.head().to_dict()}",
            response_format={"type": "json_object"}
        )
        return {
            "analysis": analysis,
            "recommendations": self.generate_sales_recommendations(sales_data)
        }
    
    def generate_descriptions(self, product_info: Dict) -> str:
        """Generate SEO-optimized product descriptions"""
        return self.llm.generate(
            prompt=f"Write product description for: {product_info}",
            temperature=0.5
        )
    
    def personalize_recommendations(self, customer_id: str) -> List[str]:
        """Generate personalized product recommendations"""
        history = self.memory.get_customer_history(customer_id)
        return self.llm.generate(
            prompt=f"Suggest products for customer with history: {history}",
            response_format={"type": "json_object"}
        )
    
    def generate_sales_recommendations(self, data: pd.DataFrame) -> Dict:
        """Generate data-driven business recommendations"""
        return self.llm.generate(
            prompt="Suggest business improvements based on sales data",
            response_format={"type": "json_object"}
        )
