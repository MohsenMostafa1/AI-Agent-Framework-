from dataclasses import dataclass
from enum import Enum
import torch

class QuantType(Enum):
    BITSANDBYTES_4BIT = "4bit"
    BITSANDBYTES_8BIT = "8bit"
    GPTQ = "gptq"
    AWQ = "awq"

@dataclass
class QuantizationConfig:
    quant_type: QuantType
    compute_dtype: torch.dtype = torch.float16
    double_quant: bool = True
    quant_storage: torch.dtype = torch.uint8
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load quantization config from YAML file"""
        # Implementation omitted for brevity
        pass

class QuantizationHandler:
    def __init__(self, config: QuantizationConfig):
        self.config = config
        
    def apply(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply quantization to model based on config"""
        if self.config.quant_type == QuantType.BITSANDBYTES_4BIT:
            return self._apply_bnb_4bit(model)
        elif self.config.quant_type == QuantType.BITSANDBYTES_8BIT:
            return self._apply_bnb_8bit(model)
        else:
            raise ValueError(f"Unsupported quantization type: {self.config.quant_type}")
    
    def _apply_bnb_4bit(self, model: torch.nn.Module) -> torch.nn.Module:
        import bitsandbytes as bnb
        return bnb.quantize.quantize_model_4bit(
            model,
            quant_type="nf4",
            compute_dtype=self.config.compute_dtype,
            double_quant=self.config.double_quant,
            quant_storage=self.config.quant_storage
        )
    
    def _apply_bnb_8bit(self, model: torch.nn.Module) -> torch.nn.Module:
        import bitsandbytes as bnb
        return bnb.quantize.quantize_model_8bit(
            model,
            quant_storage=self.config.quant_storage
        )
