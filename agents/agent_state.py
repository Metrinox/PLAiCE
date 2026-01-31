from dataclasses import dataclass
from typing import Tuple

@dataclass
class AgentState:
    agent_id: int
    fov_radius: int                 # how much it can see
    temperature: float              # randomness
    bias_contrast: float            # stylistic bias
    bias_smoothness: float
    bias_edge: float
