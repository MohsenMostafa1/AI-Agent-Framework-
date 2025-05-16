import time
import statistics
import pytest
from core.agent import Agent
from core.llm import LLMWrapper

class TestLatency:
    @pytest.fixture
    def agent(self):
        llm = LLMWrapper(model_name="tiny-llama")  # Small test model
        return Agent(llm=llm)

    def test_single_request_latency(self, agent):
        start_time = time.perf_counter()
        response = agent.process("Test latency")
        latency = time.perf_counter() - start_time
        
        print(f"\nSingle request latency: {latency:.3f} seconds")
        assert latency < 1.0, "Single request should complete under 1 second"

    def test_multiple_requests_latency(self, agent):
        latencies = []
        prompts = [
            "What is AI?",
            "Explain machine learning",
            "Tell me about neural networks",
            "Describe deep learning",
            "What are transformers?"
        ]
        
        for prompt in prompts:
            start_time = time.perf_counter()
            agent.process(prompt)
            latencies.append(time.perf_counter() - start_time)
        
        avg_latency = statistics.mean(latencies)
        std_dev = statistics.stdev(latencies)
        
        print(f"\nAverage latency: {avg_latency:.3f} ± {std_dev:.3f} seconds")
        assert avg_latency < 1.5, "Average latency should be under 1.5 seconds"
        assert std_dev < avg_latency * 0.5, "Latency should be consistent"

    def test_cold_vs_warm_start(self, agent):
        # Cold start (first request)
        cold_start = time.perf_counter()
        agent.process("Cold start test")
        cold_time = time.perf_counter() - cold_start
        
        # Warm start (subsequent requests)
        warm_start = time.perf_counter()
        agent.process("Warm start test")
        warm_time = time.perf_counter() - warm_start
        
        print(f"\nCold start: {cold_time:.3f}s, Warm start: {warm_time:.3f}s")
        assert warm_time < cold_time, "Warm start should be faster than cold start"
        assert warm_time < cold_time * 0.8, "Warm start should be significantly faster"

    def test_token_generation_speed(self, agent):
        prompt = "Explain the concept of artificial intelligence in detail."
        
        start_time = time.perf_counter()
        response = agent.process(prompt, max_tokens=100)
        total_time = time.perf_counter() - start_time
        
        tokens = len(response.split())
        tokens_per_second = tokens / total_time
        
        print(f"\nToken generation speed: {tokens_per_second:.1f} tokens/second")
        assert tokens_per_second > 10, "Should generate at least 10 tokens/second"
