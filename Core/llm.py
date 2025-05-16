from typing import Optional, Union, List
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import bitsandbytes as bnb
from core.quantization import QuantizationConfig

class QuantizedLLM:
    def __init__(self, 
                 model_name: str,
                 quant_config: Optional[QuantizationConfig] = None,
                 device: str = "auto"):
        self.model_name = model_name
        self.quant_config = quant_config
        self.device = self._resolve_device(device)
        self._load_model()
        
    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_model(self):
        """Load model with optional quantization"""
        if self.quant_config and self.quant_config.quant_type == "4bit":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                load_in_4bit=True,
                device_map=self.device,
                quantization_config=bnb.transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            )
        elif self.quant_config and self.quant_config.quant_type == "8bit":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                load_in_8bit=True,
                device_map=self.device
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device
            )
            
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
    def generate(self, 
                 prompt: str,
                 max_tokens: int = 512,
                 temperature: float = 0.7,
                 **kwargs) -> str:
        """Generate text from prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def bind_tools(self, tools: List[Any]):
        """Prepare the model for tool calling"""
        if hasattr(self.model, 'bind_tools'):
            self.model.bind_tools(tools)
        else:
            # Fallback implementation
            tool_descriptions = [f"{t.name}: {t.description}" for t in tools]
            self.system_prompt = f"""You have access to these tools:
            {', '.join(tool_descriptions)}"""
