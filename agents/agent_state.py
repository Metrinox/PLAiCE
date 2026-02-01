from dataclasses import dataclass
from typing import Tuple, Optional, Any

@dataclass
class AgentState:
    agent_id: int
    temperature: float              # randomness
    bias_contrast: float            # stylistic bias
    bias_smoothness: float
    bias_edge: float
    slice_bounds: Tuple[int, int, int, int] = (0, 0, 0, 0)
    last_guess: Optional[Any] = None
    top_x_proposals: int = 3000
    verbose: bool = False
