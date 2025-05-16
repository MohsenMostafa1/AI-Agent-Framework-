import torch
import torch.onnx
import tensorrt as trt
import onnxruntime as ort
from pathlib import Path
from typing import Optional

class InferenceOptimizer:
    def __init__(self, 
                 model: torch.nn.Module,
                 sample_input: torch.Tensor,
                 device: str = "cuda"):
        self.model = model
        self.sample_input = sample_input
        self.device = device
        self.model.eval().to(device)
        
    def export_onnx(self, 
                   output_path: str,
                   opset_version: int = 15) -> Path:
        """Export model to ONNX format"""
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True)
        
        torch.onnx.export(
            self.model,
            self.sample_input.to(self.device),
            str(output_path),
            opset_version=opset_version,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            }
        )
        return output_path
    
    def optimize_with_onnxruntime(self,
                                onnx_path: str,
                                optimized_path: str) -> ort.InferenceSession:
        """Optimize ONNX model with ORT"""
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Use CUDA execution provider if available
        providers = ["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
        
        session = ort.InferenceSession(
            onnx_path,
            sess_options,
            providers=providers
        )
        
        # Save optimized model
        if optimized_path:
            optimized_model = session.get_model()
            ort.save_model(optimized_model, optimized_path)
            
        return session
    
    def convert_to_tensorrt(self,
                          onnx_path: str,
                          trt_engine_path: str,
                          fp16: bool = True) -> trt.ICudaEngine:
        """Convert ONNX model to TensorRT engine"""
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        
        # Parse ONNX model
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise ValueError("ONNX parsing failed")
                
        # Build TRT engine
        config = builder.create_builder_config()
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            
        profile = builder.create_optimization_profile()
        input_shape = self.sample_input.shape
        profile.set_shape(
            "input",
            min=(1, *input_shape[1:]),
            opt=(32, *input_shape[1:]),  # Optimal batch size
            max=(128, *input_shape[1:])
        )
        config.add_optimization_profile(profile)
        
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("TensorRT engine build failed")
            
        # Save engine
        with open(trt_engine_path, "wb") as f:
            f.write(engine.serialize())
            
        return engine
    
    @staticmethod
    def load_trt_engine(trt_engine_path: str) -> trt.ICudaEngine:
        """Load a serialized TensorRT engine"""
        logger = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(logger)
        with open(trt_engine_path, "rb") as f:
            return runtime.deserialize_cuda_engine(f.read())
