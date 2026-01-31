from dataclasses import dataclass
from typing import Tuple

@dataclass
class Proposal:
    agent_id: int
    region_id: Tuple[int, int]
    rgb: Tuple[int, int, int]
    confidence: float
    canvas_version: int
