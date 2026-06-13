"""
Region OCR using Florence-2.
Extracts text from screen regions with tiling for large crops.
"""
import os
import re
import logging
import torch
from PIL import Image

logger = logging.getLogger("omniparser")


class RegionOCR:
    """Extract text from screen regions using Florence-2 OCR with tiling."""

    def __init__(self, processor, caption_model, device: str, dtype):
        self.processor = processor
        self.caption_model = caption_model
        self.device = device
        self.dtype = dtype

    def extract_text(self, image: Image.Image, bbox: list) -> str:
        """Extract text from a region using Florence-2 OCR with tiling.

        Florence-2 resizes to 768x768 internally, so crops larger than that
        lose detail. Tiling splits them into overlapping chunks that each get
        full-resolution OCR, then stitches results.

        Args:
            image: Full screenshot PIL Image.
            bbox: [x1, y1, x2, y2] pixel coordinates of the region.

        Returns:
            The extracted text from the region.
        """
        x1, y1, x2, y2 = map(int, bbox)
        logger.info("OCR region: [%d, %d, %d, %d]", x1, y1, x2, y2)

        crop = image.crop((x1, y1, x2, y2))
        w, h = crop.width, crop.height

        TILE_SIZE = int(os.environ.get("OCR_TILE_SIZE", "768"))
        TILE_OVERLAP = int(os.environ.get("OCR_TILE_OVERLAP", "64"))
        MAX_TOKENS = int(os.environ.get("FLORENCE_MAX_TOKENS_OCR", "200"))

        if w <= TILE_SIZE and h <= TILE_SIZE:
            raw_text = self._ocr_single_tile(crop, MAX_TOKENS)
        else:
            # ── Tile a large region ──
            x_starts = list(range(0, max(1, w - TILE_OVERLAP), TILE_SIZE - TILE_OVERLAP))
            y_starts = list(range(0, max(1, h - TILE_OVERLAP), TILE_SIZE - TILE_OVERLAP))
            # Ensure right/bottom edge is covered
            if x_starts and x_starts[-1] + TILE_SIZE < w:
                x_starts.append(max(0, w - TILE_SIZE))
            if y_starts and y_starts[-1] + TILE_SIZE < h:
                y_starts.append(max(0, h - TILE_SIZE))
            if not x_starts:
                x_starts = [0]
            if not y_starts:
                y_starts = [0]

            total_tiles = len(x_starts) * len(y_starts)
            logger.info("OCR tiling: %dx%d → %d tiles (%dx%d each, overlap=%d)",
                        w, h, total_tiles, TILE_SIZE, TILE_SIZE, TILE_OVERLAP)

            tile_texts = []
            for y_start in y_starts:
                for x_start in x_starts:
                    x_end = min(x_start + TILE_SIZE, w)
                    y_end = min(y_start + TILE_SIZE, h)
                    tile = crop.crop((x_start, y_start, x_end, y_end))
                    tile_text = self._ocr_single_tile(tile, MAX_TOKENS)
                    if tile_text:
                        tile_texts.append(tile_text)

            raw_text = "\n".join(tile_texts)

        text = self._postprocess_ocr(raw_text)
        logger.info("OCR result: '%s'", text)
        return text

    def _ocr_single_tile(self, tile: Image.Image, max_tokens: int) -> str:
        """Run a single Florence-2 OCR forward pass on a tile."""
        w, h = tile.width, tile.height
        task_prompt = "<OCR>"
        inputs = self.processor(
            text=[task_prompt],
            images=[tile],
            return_tensors="pt"
        ).to(self.device)

        if self.device != "cpu":
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

        with torch.no_grad():
            generated_ids = self.caption_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_tokens,
                num_beams=1,
                early_stopping=False
            )

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        parsed = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(w, h)
        )
        return parsed[task_prompt].strip()

    @staticmethod
    def _postprocess_ocr(raw: str) -> str:
        """Clean Florence-2 OCR output with language-agnostic rules.

        Florence-2 occasionally omits spaces between words. The rules below
        fix common boundary artifacts without assuming English.
        """
        if not raw or not raw.strip():
            return ""

        # 1. lowercase→UPPERCASE boundary: "nThis" → "n This"
        raw = re.sub(r'([a-z])([A-Z])', r'\1 \2', raw)
        # 2. UPPERCASE_WORD→Capitalized boundary
        raw = re.sub(r'([A-Z]{3,})([A-Z][a-z])', r'\1 \2', raw)
        # 3. lowercase→digit & digit→lowercase boundaries
        raw = re.sub(r'([a-z])(\d)', r'\1 \2', raw)
        raw = re.sub(r'(\d)([a-z])', r'\1 \2', raw)
        # 4. Sentence boundary: period/comma/exclamation followed by capital
        raw = re.sub(r'([.!?,;:])([A-Z])', r'\1 \2', raw)
        # 5. Collapse multiple spaces and normalize newlines
        raw = re.sub(r'[ \t]+', ' ', raw)
        raw = re.sub(r'\n{3,}', '\n\n', raw)
        # 5. Trim each line
        lines = [line.strip() for line in raw.split('\n') if line.strip()]

        return '\n'.join(lines)
