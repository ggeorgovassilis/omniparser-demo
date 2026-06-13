"""
OmniParser orchestration layer.
Coordinates detection, labelling, and OCR via dependency injection.
"""
import os
import logging

# ── PyTorch thread limits (configurable via docker-compose env vars) ──
# TORCH_INTRAOP_THREADS controls parallelism within individual PyTorch ops.
# TORCH_INTEROP_THREADS controls parallelism across independent PyTorch ops.
import torch
torch.set_num_threads(int(os.environ.get("TORCH_INTRAOP_THREADS", "4")))
torch.set_num_interop_threads(int(os.environ.get("TORCH_INTEROP_THREADS", "2")))

import warnings
from PIL import Image
from thefuzz import fuzz

from models import load_models
from detection import RegionDetector
from labelling import RegionLabeler
from ollama_labelling import OllamaRegionLabeler
from ocr import RegionOCR

logger = logging.getLogger("omniparser")

# Suppress Hugging Face generation warnings about beams/early_stopping
warnings.filterwarnings("ignore", message=".*`num_beams` is set to 1.*`early_stopping` is set to `True`.*")


class OmniParser:
    """Orchestrator that coordinates detection, labelling, and OCR.

    Loads shared models once and injects them into sub-components.
    """

    def __init__(self, weights_dir: str = "weights"):
        # ── Load shared models via centralized registry ──
        models = load_models(weights_dir)

        # ── Dependency injection: each sub-component gets what it needs ──
        self.detector = RegionDetector(
            yolo_model=models["yolo_model"],
            device=models["device"],
        )
        
        labelling_engine = os.environ.get("LABELLING_ENGINE", "florence").lower()
        if labelling_engine == "ollama":
            self.labeler = OllamaRegionLabeler(
                base_url=os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11435"),
                model_name=os.environ.get("OLLAMA_MODEL", "qwen3-vl:8b")
            )
        else:
            self.labeler = RegionLabeler(
                processor=models["processor"],
                caption_model=models["caption_model"],
                device=models["device"],
                dtype=models["dtype"],
            )

        self.ocr = RegionOCR(
            processor=models["processor"],
            caption_model=models["caption_model"],
            device=models["device"],
            dtype=models["dtype"],
        )

    def parse_screen(self, image: Image.Image) -> list[dict]:
        """Detect UI elements and label each one.

        Args:
            image: PIL Image (RGB screenshot).

        Returns:
            List of dicts with keys: id, label, bbox, center.
        """
        boxes = self.detector.detect(image)
        elements = self.labeler.label_regions(image, boxes)
        return elements

    def ocr_region(self, image: Image.Image, bbox: list) -> str:
        """Extract text from a region using Florence-2 OCR with tiling.

        Args:
            image: Full screenshot PIL Image.
            bbox: [x1, y1, x2, y2] pixel coordinates of the region.

        Returns:
            The extracted text from the region.
        """
        return self.ocr.extract_text(image, bbox)

    def find_elements(self, image: Image.Image, prompt: str, threshold: int = 50) -> list[dict]:
        """Parse screen and fuzzy-match element labels against a user prompt.

        Args:
            image: PIL Image (RGB screenshot).
            prompt: User search string (e.g. "find the submit button").
            threshold: Minimum fuzzy match score (0-100).

        Returns:
            List of matching element dicts sorted by match score descending.
        """
        logger.info("find_elements called with prompt: '%s'", prompt)
        elements = self.parse_screen(image)

        matches = []
        for el in elements:
            score = fuzz.token_set_ratio(prompt.lower(), el["label"].lower())
            if score >= threshold:
                el["match_score"] = score
                matches.append(el)

        matches.sort(key=lambda x: x["match_score"], reverse=True)
        logger.info("find_elements: %d matches found for '%s'", len(matches), prompt)
        return matches


# ── Singleton pattern for the API layer to reuse the loaded model ──
_parser_instance = None


def get_parser() -> OmniParser:
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = OmniParser()
    return _parser_instance
