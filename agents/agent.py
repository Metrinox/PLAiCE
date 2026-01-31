from agents.proposal import Proposal
from agents.perception import extract_fov

class Agent:
    def __init__(self, state, model):
        self.state = state
        self.model = model

    def step(self, canvas_snapshot, region_id, canvas_version):
        x, y = region_id

        perception = extract_fov(
            canvas_snapshot,
            x,
            y,
            self.state.fov_radius
        )

        rgb, confidence = self.model.infer(
            perception,
            self.state
        )

        return Proposal(
            agent_id=self.state.agent_id,
            region_id=region_id,
            rgb=rgb,
            confidence=confidence,
            canvas_version=canvas_version
        )
