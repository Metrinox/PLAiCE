import numpy as np
from PIL import Image
from agents.evaluator.difference import ImageDifference
from agents.evaluator.proposals import generate_proposals

class Evaluator:
    def __init__(self, device="cpu"):
        self.diff_engine = ImageDifference(device=device)

    def evaluate(
        self,
        current_canvas_img: Image.Image,
        generated_img: Image.Image,
        top_x: int
    ):
        """
        Returns list of (x, y, rgb) proposals
        """
        current_np = np.array(current_canvas_img)
        generated_np = np.array(generated_img)

        diff_scores = self.diff_engine.compute_patch_difference(
            current_np, generated_np
        )

        proposals = generate_proposals(
            diff_scores,
            current_np,
            generated_np,
            top_x
        )

        return proposals
