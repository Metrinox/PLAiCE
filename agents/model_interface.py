import numpy as np

class ModelInterface:
    def infer(self, perception, agent_state):
        """
        Returns (rgb, confidence)
        """
        raise NotImplementedError

class AgentModel(ModelInterface):
    def __init__(self):
        pass
    
    def infer(self, perception, agent_state):
        """
        Given a perception (local patch of pixels) and agent state attributes,
        infer a coherent RGB color and confidence score without external APIs.
        
        Args:
            perception: numpy array of shape (H, W, 3) with RGB values
            agent_state: AgentState object with agent attributes
        
        Returns:
            ((r, g, b), confidence) tuple
        """
        if perception is None or perception.size == 0:
            return (128, 128, 128), 0.5

        base = perception.mean(axis=(0, 1))

        # Apply a light contrast bias around mid-gray.
        contrast = max(0.0, min(1.0, float(agent_state.bias_contrast)))
        base = 128.0 + (base - 128.0) * (1.0 + 0.5 * contrast)

        # Add temperature-controlled noise.
        temperature = max(0.0, float(agent_state.temperature))
        noise = np.random.normal(0.0, 25.0 * temperature, size=3)
        rgb = base + noise

        r, g, b = [int(max(0, min(255, c))) for c in rgb]
        confidence = max(0.0, min(1.0, 0.6 - 0.3 * temperature))

        return (r, g, b), confidence
