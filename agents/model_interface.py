import anthropic
import os
import json
import numpy as np
from dotenv import load_dotenv
from .prompt import build_agent_prompt

load_dotenv()

class ModelInterface:
    def infer(self, perception, agent_state):
        """
        Returns (rgb, confidence)
        """
        raise NotImplementedError

class AgentModel(ModelInterface):
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    def infer(self, perception, agent_state):
        """
        Given a perception (local patch of pixels) and agent state attributes,
        use Claude to infer a coherent RGB color and confidence score.
        
        Args:
            perception: numpy array of shape (H, W, 3) with RGB values
            agent_state: AgentState object with agent attributes
        
        Returns:
            ((r, g, b), confidence) tuple
        """
        # Prepare perception data for Claude
        perception_stats = {
            "shape": perception.shape,
            "mean_rgb": perception.mean(axis=(0, 1)).tolist(),
            "std_rgb": perception.std(axis=(0, 1)).tolist(),
            "min_rgb": perception.min(axis=(0, 1)).tolist(),
            "max_rgb": perception.max(axis=(0, 1)).tolist(),
        }
        
        # Prepare agent state for Claude
        agent_info = {
            "agent_id": agent_state.agent_id,
            "fov_radius": agent_state.fov_radius,
            "temperature": agent_state.temperature,
            "bias_contrast": agent_state.bias_contrast,
            "bias_smoothness": agent_state.bias_smoothness,
            "bias_edge": agent_state.bias_edge,
        }
        
        prompt = build_agent_prompt(perception_stats, agent_info)
        
        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Parse the response
        try:
            response_text = message.content[0].text
            # Extract JSON from the response
            result = json.loads(response_text)
            r = int(result["r"])
            g = int(result["g"])
            b = int(result["b"])
            confidence = float(result["confidence"])
            
            # Clamp values to valid ranges
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))
            confidence = max(0.0, min(1.0, confidence))
            
            return (r, g, b), confidence
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback to a neutral gray if parsing fails
            print(f"Error parsing Claude response: {e}")
            return (128, 128, 128), 0.5
