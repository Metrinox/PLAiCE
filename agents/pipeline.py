"""
Pipeline integration for stateless, modular image generation and evaluation.

This module composes three independent components:
1. PromptGenerator: image → descriptive prompt (uses classifier)
2. Diffuser: prompt → PIL.Image
3. Evaluator: (current_img, generated_img) → pixel proposals

No global state. All data passed explicitly.
"""

from typing import List, Tuple, Optional
from PIL import Image
import numpy as np

# Type aliases for clarity
PixelProposal = Tuple[int, int, Tuple[int, int, int]]  # (x, y, (r, g, b))
RGB = Tuple[int, int, int]


class PipelineConfig:
    """Configuration for pipeline execution."""

    def __init__(
        self,
        image_size: int = 128,
        evaluator_device: str = "cpu",
        diffuser_device: Optional[str] = None,
        top_x_proposals: int = 10,
    ):
        """
        Args:
            image_size: Width and height of generated/canvas images (square)
            evaluator_device: Device for evaluator ("cpu", "cuda", "mps")
            diffuser_device: Device for diffuser (auto-select if None)
            top_x_proposals: Number of pixel proposals to extract
        """
        self.image_size = image_size
        self.evaluator_device = evaluator_device
        self.diffuser_device = diffuser_device
        self.top_x_proposals = top_x_proposals


class DiffusionPromptPipeline:
    """
    Stateless wrapper for text-to-image diffusion.

    Input: prompt (str)
    Output: PIL.Image (fixed size)
    """

    def __init__(self, config: PipelineConfig):
        """Initialize diffuser (lazy-load on first call)."""
        self.config = config
        self._diffuser = None

    def _get_diffuser(self):
        """Lazy-load diffuser on first use."""
        if self._diffuser is None:
            from diffusers import DiffusionPipeline
            import torch

            device = self.config.diffuser_device or self._auto_device()
            dtype = torch.bfloat16 if "cuda" in device else torch.float32

            self._diffuser = DiffusionPipeline.from_pretrained(
                "amused/amused-256",
                torch_dtype=dtype,
                device_map=device,
            )

        return self._diffuser

    @staticmethod
    def _auto_device():
        """Auto-select device: cuda > mps (macOS only) > cpu."""
        import torch
        import sys

        if torch.cuda.is_available():
            return "cuda"
        elif (
            sys.platform == "darwin"
            and getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            return "mps"
        else:
            return "cpu"

    def generate(self, prompt: str) -> Image.Image:
        """
        Generate image from text prompt.

        Args:
            prompt: Text description of the image to generate

        Returns:
            PIL.Image of size (image_size, image_size) in RGB mode
        """
        diffuser = self._get_diffuser()
        result = diffuser(prompt)
        image = result.images[0]

        # Ensure fixed size and RGB
        image = image.resize(
            (self.config.image_size, self.config.image_size),
            resample=Image.LANCZOS,
        )
        image = image.convert("RGB")

        return image


class EvaluationPipeline:
    """
    Stateless wrapper for image comparison and proposal generation.

    Inputs: current_canvas_img, generated_img, top_x
    Output: List of (x, y, rgb) pixel proposals
    """

    def __init__(self, config: PipelineConfig):
        """Initialize evaluator."""
        self.config = config
        # Lazy import to avoid unnecessary dependencies
        from agents.evaluator.evaluator import Evaluator

        self.evaluator = Evaluator(device=config.evaluator_device)

    def propose_pixels(
        self,
        current_canvas_img: Image.Image,
        generated_img: Image.Image,
    ) -> List[PixelProposal]:
        """
        Compare two images and propose pixel-level updates.

        Args:
            current_canvas_img: Current state of the canvas region (PIL.Image)
            generated_img: Generated image from diffuser (PIL.Image)

        Returns:
            List of (x, y, (r, g, b)) proposals, up to top_x_proposals
        """
        proposals = self.evaluator.evaluate(
            current_canvas_img,
            generated_img,
            top_x=self.config.top_x_proposals,
        )
        return proposals


class LocalRegionPipeline:
    """
    End-to-end composition: canvas image → prompt → diffused image → evaluation → proposals.

    Stateless integration of PromptGenerator, Diffuser, and Evaluator.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize pipeline with configuration.

        Args:
            config: PipelineConfig (uses defaults if None)
        """
        self.config = config or PipelineConfig()
        from agents.prompt_generator import PromptGenerator

        self.prompt_generator = PromptGenerator(device=config.evaluator_device)
        self.diffuser = DiffusionPromptPipeline(self.config)
        self.evaluator = EvaluationPipeline(self.config)

    def process_canvas_region_to_proposals(
        self,
        canvas_region: Image.Image,
    ) -> Tuple[str, Image.Image, List[PixelProposal]]:
        """
        End-to-end pipeline: canvas image → prompt → generated image → proposals.

        This is the main entry point for stateless composition.
        Analyzes the canvas using the classifier to generate an appropriate prompt.

        Args:
            canvas_region: Current image of the region (PIL.Image)

        Returns:
            Tuple of (prompt, generated_img, proposals)
        """
        # Step 1: Generate prompt from canvas region analysis
        prompt = self.prompt_generator.generate_prompt_from_image(canvas_region)

        # Step 2: Diffuser converts prompt to image
        generated_img = self.diffuser.generate(prompt)

        # Step 3: Evaluator compares and proposes pixels
        proposals = self.evaluator.propose_pixels(
            canvas_region,
            generated_img,
        )

        return prompt, generated_img, proposals

    def process_prompt_against_canvas(
        self,
        prompt: str,
        current_canvas_region: Image.Image,
    ) -> List[PixelProposal]:
        """
        Pipeline with explicit prompt: prompt + canvas → proposals.

        Useful for testing or when prompt is provided externally.

        Args:
            prompt: Text description of local visual intent
            current_canvas_region: Current image of the region (PIL.Image)

        Returns:
            List of (x, y, (r, g, b)) pixel proposals
        """
        # Step 1: Diffuser converts prompt to image
        generated_img = self.diffuser.generate(prompt)

        # Step 2: Evaluator compares and proposes pixels
        proposals = self.evaluator.propose_pixels(
            current_canvas_region,
            generated_img,
        )

        return proposals

    def process_prompt_to_image(self, prompt: str) -> Image.Image:
        """
        Generate image from prompt only (no canvas comparison).

        Useful for testing diffuser in isolation.

        Args:
            prompt: Text description

        Returns:
            PIL.Image of size (image_size, image_size)
        """
        return self.diffuser.generate(prompt)


# Convenience functions for simple use cases

def generate_and_evaluate(
    prompt: str,
    current_canvas_region: Image.Image,
    image_size: int = 128,
    evaluator_device: str = "cpu",
    top_x_proposals: int = 10,
) -> List[PixelProposal]:
    """
    One-shot composition: prompt + canvas → proposals.

    For simple cases where you don't need to reuse components.

    Args:
        prompt: Text description
        current_canvas_region: Current region image
        image_size: Size of generated image
        evaluator_device: Device for evaluator
        top_x_proposals: Number of proposals to return

    Returns:
        List of pixel proposals
    """
    config = PipelineConfig(
        image_size=image_size,
        evaluator_device=evaluator_device,
        top_x_proposals=top_x_proposals,
    )
    pipeline = LocalRegionPipeline(config)
    return pipeline.process_prompt_against_canvas(prompt, current_canvas_region)


def generate_from_canvas(
    canvas_region: Image.Image,
    image_size: int = 128,
    evaluator_device: str = "cpu",
    top_x_proposals: int = 10,
) -> Tuple[str, Image.Image, List[PixelProposal]]:
    """
    One-shot composition: canvas → prompt → proposals.

    Uses classifier to analyze canvas and generate an appropriate prompt.

    Args:
        canvas_region: Canvas region image
        image_size: Size of generated image
        evaluator_device: Device for evaluator
        top_x_proposals: Number of proposals to return

    Returns:
        Tuple of (prompt, generated_img, proposals)
    """
    config = PipelineConfig(
        image_size=image_size,
        evaluator_device=evaluator_device,
        top_x_proposals=top_x_proposals,
    )
    pipeline = LocalRegionPipeline(config)
    return pipeline.process_canvas_region_to_proposals(canvas_region)
