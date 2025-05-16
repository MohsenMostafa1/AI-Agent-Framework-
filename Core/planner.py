from typing import List, Dict, Any
from enum import Enum
import networkx as nx

class TaskStatus(Enum):
    PENDING = 0
    IN_PROGRESS = 1
    COMPLETED = 2
    FAILED = 3

class TaskNode:
    def __init__(self, 
                 description: str,
                 dependencies: List[str] = None,
                 expected_output: str = None):
        self.description = description
        self.dependencies = dependencies or []
        self.status = TaskStatus.PENDING
        self.expected_output = expected_output

class Planner:
    def __init__(self, llm: Any):
        self.llm = llm
        self.task_graph = nx.DiGraph()
        
    def create_plan(self, goal: str) -> Dict[str, TaskNode]:
        """Decompose a high-level goal into actionable tasks"""
        prompt = f"""Break down the following goal into specific tasks:
Goal: {goal}

Output as JSON with tasks and dependencies:"""
        
        response = self.llm.generate(prompt)
        try:
            task_dict = json.loads(response)
            self._build_graph(task_dict)
            return task_dict
        except json.JSONDecodeError:
            return self._fallback_plan(goal)
    
    def _build_graph(self, task_dict: Dict[str, Any]):
        """Convert task dictionary into graph structure"""
        self.task_graph.clear()
        
        for task_name, task_data in task_dict.items():
            self.task_graph.add_node(task_name, 
                                   description=task_data['description'],
                                   status=TaskStatus.PENDING)
            
            for dep in task_data.get('dependencies', []):
                self.task_graph.add_edge(dep, task_name)
    
    def get_next_tasks(self) -> List[str]:
        """Return tasks that are ready to execute (dependencies met)"""
        ready_tasks = []
        for node in self.task_graph.nodes():
            if self.task_graph.nodes[node]['status'] == TaskStatus.PENDING:
                predecessors = list(self.task_graph.predecessors(node))
                if all(self.task_graph.nodes[p]['status'] == TaskStatus.COMPLETED 
                       for p in predecessors):
                    ready_tasks.append(node)
        return ready_tasks
    
    def update_task(self, task_name: str, status: TaskStatus):
        """Update the status of a task"""
        if task_name in self.task_graph.nodes:
            self.task_graph.nodes[task_name]['status'] = status
