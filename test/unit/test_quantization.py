import pytest
import torch
from core.quantization import QuantizationManager
from core.llm import LLMWrapper

class TestQuantization:
    @pytest.fixture
    def quant_manager(self):
        return QuantizationManager()
    
    @pytest.fixture
    def sample_model(self):
        # Mock a small model for testing
        return torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2)
    
    def test_4bit_quantization(self, quant_manager, sample_model):
        quantized_model = quant_manager.quantize_model(sample_model, bits=4)
        assert hasattr(quantized_model, 'quantized_layers'), "Model should have quantized layers"
        assert quantized_model.quantized_layers == 4, "All linear layers should be quantized"
        
        # Verify memory reduction
        original_size = sum(p.numel() for p in sample_model.parameters())
        quantized_size = quant_manager.estimate_memory_usage(quantized_model)
        assert quantized_size < original_size * 0.4, "4-bit model should use <40% of original memory"

    def test_8bit_quantization(self, quant_manager, sample_model):
        quantized_model = quant_manager.quantize_model(sample_model, bits=8)
        assert hasattr(quantized_model, 'quantized_layers'), "Model should have quantized layers"
        
        # Verify inference works
        test_input = torch.randn(1, 10)
        output = quantized_model(test_input)
        assert output.shape == (1, 2), "Output shape should match original model"

    def test_quantization_compatibility(self, quant_manager):
        llm = LLMWrapper(model_name="tiny-llama")  # Using a small test model
        original_output = llm.generate("Test prompt")
        
        quantized_llm = quant_manager.quantize_llm(llm, bits=4)
        quantized_output = quantized_llm.generate("Test prompt")
        
        # Verify outputs are similar (allow for quantization noise)
        assert torch.allclose(
            original_output.logits, 
            quantized_output.logits, 
            atol=0.1
        ), "Quantized outputs should be similar to original"
