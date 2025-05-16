from typing import List, Dict
from ..base_plugin import BasePlugin, PluginInput, PluginOutput
from pydantic import BaseModel, Field

class SegmentationInput(PluginInput):
    customers: List[Dict[str, Any]] = Field(..., description="Customer data")
    segmentation_type: str = Field("rfm", description="Type of segmentation to perform")
    parameters: Dict[str, Any] = Field({}, description="Segmentation parameters")

class SegmentationOutput(PluginOutput):
    segments: Dict[str, List[str]] = Field(..., description="Customer segments")
    segment_profiles: Dict[str, Dict[str, Any]] = Field(..., description="Segment characteristics")
    visualization: str = Field("", description="Segmentation visualization data")

class CustomerSegmentationPlugin(BasePlugin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.segmenter = None
    
    async def initialize(self):
        from ecommerce_models import CustomerSegmenter
        self.segmenter = CustomerSegmenter(self.config)
        await self.segmenter.load_model()
        self.initialized = True
    
    async def execute(self, input_data: SegmentationInput) -> SegmentationOutput:
        if not self.initialized:
            raise RuntimeError("Plugin not initialized")
        
        segments = await self.segmenter.segment(
            input_data.customers,
            input_data.segmentation_type,
            input_data.parameters
        )
        
        return SegmentationOutput(
            segments=segments.get("segments", {}),
            segment_profiles=segments.get("profiles", {}),
            visualization=segments.get("visualization", "")
        )
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "Customer Segmentation Plugin",
            "version": "1.0.0",
            "description": "Segments customers for targeted marketing"
        }
    
    @property
    def input_schema(self) -> type[PluginInput]:
        return SegmentationInput
    
    @property
    def output_schema(self) -> type[PluginOutput]:
        return SegmentationOutput
