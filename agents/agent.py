from agents.proposal import Proposal
from agents.pipeline import PipelineConfig, DiffusionPromptPipeline
from agents.prompt_generator import PromptGenerator
from PIL import Image
import numpy as np

class Agent:
    def __init__(self, state, model=None, pipeline_config=None):
        self.state = state
        self.model = model
        self.pipeline_config = pipeline_config or PipelineConfig()
        self.prompt_generator = PromptGenerator(device=self.pipeline_config.evaluator_device)
        self.diffuser = DiffusionPromptPipeline(self.pipeline_config)

    def _classify(self, fov_image: Image.Image) -> str:
        if self.state.last_guess is not None:
            last_guess = self.state.last_guess
            if last_guess.size != fov_image.size:
                last_guess = last_guess.resize(fov_image.size, resample=Image.LANCZOS)
            fov_image = Image.blend(fov_image, last_guess, alpha=0.5)
        return self.prompt_generator.generate_prompt_from_image(fov_image)

    def _diff_to_proposals(self, fov_np, gen_np, fov_origin, canvas_version):
        if fov_np.size == 0 or gen_np.size == 0:
            return []

        diff = np.linalg.norm(fov_np.astype(np.float32) - gen_np.astype(np.float32), axis=2)
        flat = diff.reshape(-1)
        top_x = min(self.state.top_x_proposals, flat.shape[0])
        if top_x <= 0:
            return []

        indices = np.argpartition(flat, -top_x)[-top_x:]
        max_diff = flat.max() if flat.size > 0 else 0.0
        h, w = diff.shape
        x0, y0 = fov_origin
        proposals = []

        for idx in indices:
            y = idx // w
            x = idx % w
            r, g, b = gen_np[y, x]
            confidence = float(flat[idx] / max_diff) if max_diff > 0 else 0.0
            proposals.append(
                Proposal(
                    agent_id=self.state.agent_id,
                    region_id=(x0 + x, y0 + y),
                    rgb=(int(r), int(g), int(b)),
                    confidence=confidence,
                    canvas_version=canvas_version,
                )
            )

        return proposals

    def step(self, fov, fov_origin, canvas_version):
        if fov is None or len(fov) == 0:
            return []

        fov_np = np.array(fov, dtype=np.uint8)
        fov_image = Image.fromarray(fov_np, mode="RGB")

        label = self._classify(fov_image)

        generated = self.diffuser.generate(label)
        generated = generated.resize(fov_image.size, resample=Image.LANCZOS).convert("RGB")

        self.state.last_guess = generated

        gen_np = np.array(generated, dtype=np.uint8)
        return self._diff_to_proposals(fov_np, gen_np, fov_origin, canvas_version)
