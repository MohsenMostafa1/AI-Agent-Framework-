from typing import List, Dict
from ..base_plugin import BasePlugin, PluginInput, PluginOutput
from pydantic import BaseModel, Field

class CartInput(PluginInput):
    cart_items: List[Dict[str, Any]] = Field(..., description="Items in the shopping cart")
    user_profile: Dict[str, Any] = Field({}, description="User profile information")
    constraints: Dict[str, Any] = Field({}, description="Optimization constraints")

class CartOutput(PluginOutput):
    optimized_cart: List[Dict[str, Any]] = Field(..., description="Optimized cart items")
    savings: Dict[str, float] = Field(..., description="Estimated savings")
    recommendations: List[str] = Field([], description="Optimization recommendations")

class CartOptimizationPlugin(BasePlugin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.optimizer = None
    
    async def initialize(self):
        from ecommerce_models import CartOptimizer
        self.optimizer = CartOptimizer(self.config)
        await self.optimizer.load_model()
        self.initialized = True
    
    async def execute(self, input_data: CartInput) -> CartOutput:
        if not self.initialized:
            raise RuntimeError("Plugin not initialized")
        
        result = await self.optimizer.optimize(
            input_data.cart_items,
            input_data.user_profile,
            input_data.constraints
        )
        
        return CartOutput(
            optimized_cart=result.get("optimized_cart", []),
            savings=result.get("savings", {}),
            recommendations=result.get("recommendations", [])
        )
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "Cart Optimization Plugin",
            "version": "1.0.0",
            "description": "Optimizes shopping carts for cost, shipping, and preferences"
        }
    
    @property
    def input_schema(self) -> type[PluginInput]:
        return CartInput
    
    @property
    def output_schema(self) -> type[PluginOutput]:
        return CartOutput
