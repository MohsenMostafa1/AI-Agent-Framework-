import pytest
import time
from concurrent.futures import ThreadPoolExecutor
from core.agent import Agent
from core.llm import LLMWrapper
from core.memory import VectorMemory

class TestStress:
    @pytest.fixture
    def agent(self):
        # Create a lightweight agent for testing
        llm = LLMWrapper(model_name="tiny-llama")  # Small test model
        memory = VectorMemory(index_name="stress_test")
        return Agent(llm=llm, memory=memory)

    def test_concurrent_requests(self, agent):
        def make_request(prompt):
            start_time = time.time()
            response = agent.process(prompt)
            return time.time() - start_time
        
        prompts = [f"Test prompt {i}" for i in range(100)]  # 100 concurrent prompts
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            times = list(executor.map(make_request, prompts))
        
        avg_time = sum(times) / len(times)
        print(f"\nAverage response time: {avg_time:.2f}s for 100 concurrent requests")
        
        assert avg_time < 2.0, "Average response time should be under 2 seconds"
        assert max(times) < 5.0, "No single request should take more than 5 seconds"

    def test_memory_usage_under_load(self, agent):
        memory_usage = []
        
        for i in range(500):  # 500 sequential requests
            agent.process(f"Memory test {i}")
            if i % 50 == 0:
                # Get memory usage (simplified for test)
                mem = agent.memory.estimate_usage()
                memory_usage.append(mem)
                print(f"Memory after {i} requests: {mem}MB")
        
        # Verify memory growth is linear or sublinear
        growth_factor = memory_usage[-1] / memory_usage[0] if memory_usage[0] > 0 else 1
        assert growth_factor < 5, "Memory usage should not grow excessively"

    @pytest.mark.skip(reason="Requires GPU resources")
    def test_gpu_memory_pressure(self, agent):
        # This would require actual GPU testing environment
        pass
