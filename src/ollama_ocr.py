import os
import time
import base64
import requests
import io
import logging
from PIL import Image

logger = logging.getLogger("omniparser")

class OllamaRegionOCR:
    """Extract text from screen regions using an external Ollama vision model."""

    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name

    def extract_text(self, image: Image.Image, bbox: list) -> str:
        """Extract text from a region using Ollama.

        Args:
            image: Full screenshot PIL Image.
            bbox: [x1, y1, x2, y2] pixel coordinates of the region.

        Returns:
            The extracted text from the region.
        """
        x1, y1, x2, y2 = map(int, bbox)
        logger.info("Ollama OCR region: [%d, %d, %d, %d] using model %s", x1, y1, x2, y2, self.model_name)

        # Crop the bounding box
        crop = image.crop((x1, y1, x2, y2))
        
        # Convert PIL Image to Base64 JPEG string
        img_byte_arr = io.BytesIO()
        crop.save(img_byte_arr, format='JPEG')
        base64_encoded = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

        # Send request to Ollama's generate API
        prompt = "Extract all text visible in this image exactly as it appears. Output only the extracted text and nothing else. If there is no text, output nothing."
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [base64_encoded],
            "stream": False
        }

        try:
            start_time = time.time()
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=300)
            response.raise_for_status()
            data = response.json()
            text = data.get("response", "").strip()
            elapsed = time.time() - start_time
            logger.info("Ollama OCR completed in %.2fs. Result: '%s'", elapsed, text)
            return text
        except Exception as e:
            logger.error("Failed to extract text for region [%d, %d, %d, %d]: %s", x1, y1, x2, y2, e)
            return ""
