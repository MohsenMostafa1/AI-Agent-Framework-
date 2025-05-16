from typing import Dict, Any, List, Tuple
import numpy as np
from pydantic import BaseModel
import torch
import torch.nn as nn

class FeedbackItem(BaseModel):
    input_text: str
    output_text: str
    rating: float  # 0-1 scale
    corrections: List[str] = []
    metadata: Dict[str, Any] = {}

class RewardModel(nn.Module):
    """Simple neural network for predicting reward scores"""
    def __init__(self, input_size: int = 768, hidden_size: int = 256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class FeedbackHandler:
    def __init__(self, 
                 llm_embedder: Any,
                 reward_model: Optional[RewardModel] = None):
        self.llm_embedder = llm_embedder
        self.reward_model = reward_model or self._default_reward_model()
        self.feedback_buffer = []
        
    def add_feedback(self, feedback: FeedbackItem):
        """Store feedback for later processing"""
        self.feedback_buffer.append(feedback)
        
    def calculate_self_score(self, input_text: str, output_text: str) -> float:
        """Score the quality of output without human feedback"""
        # Get embeddings
        input_emb = self.llm_embedder.embed(input_text)
        output_emb = self.llm_embedder.embed(output_text)
        
        # Calculate similarity
        similarity = np.dot(input_emb, output_emb) / (
            np.linalg.norm(input_emb) * np.linalg.norm(output_emb)
        )
        
        # Additional scoring factors
        length_penalty = min(1, len(output_text) / 1000)  # Prefer concise outputs
        
        return (similarity * 0.7 + length_penalty * 0.3)
    
    def train_reward_model(self, 
                         epochs: int = 3,
                         batch_size: int = 32):
        """Train the reward model on collected feedback"""
        if not self.feedback_buffer:
            return
            
        # Convert feedback to training data
        X, y = [], []
        for fb in self.feedback_buffer:
            emb = self.llm_embedder.embed(fb.input_text + fb.output_text)
            X.append(emb)
            y.append(fb.rating)
            
        X = torch.tensor(np.array(X), dtype=torch.float32)
        y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)
        
        # Training loop
        optimizer = torch.optim.Adam(self.reward_model.parameters())
        loss_fn = nn.MSELoss()
        
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                preds = self.reward_model(batch_X)
                loss = loss_fn(preds, batch_y)
                loss.backward()
                optimizer.step()
                
    def get_reward(self, input_text: str, output_text: str) -> float:
        """Predict reward score for given input/output pair"""
        with torch.no_grad():
            emb = self.llm_embedder.embed(input_text + output_text)
            emb_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
            return self.reward_model(emb_tensor).item()
    
    def _default_reward_model(self) -> RewardModel:
        """Create a default reward model if none provided"""
        return RewardModel()
