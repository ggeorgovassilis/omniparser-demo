import os
import time
import base64
import requests
import io
import logging
import concurrent.futures
from PIL import Image

logger = logging.getLogger("omniparser")

class OllamaRegionLabeler:
    """Label detected UI regions using an external Ollama vision model."""

    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        # Allow configuration of max workers for parallel HTTP requests
        self.max_workers = int(os.environ.get("OLLAMA_MAX_WORKERS", "4"))

    def label_regions(self, image: Image.Image, boxes: list[list[float]]) -> list[dict]:
        """Crop each bounding box from the image and generate a label using Ollama.

        Args:
            image: Full screenshot PIL Image.
            boxes: List of [xmin, ymin, xmax, ymax] boxes.

        Returns:
            List of dicts with keys: id, label, bbox, center.
        """
        logger.info("Ollama labelling started for %d crops using model %s (max concurrent requests: %d)", 
                    len(boxes), self.model_name, self.max_workers)
        
        elements = []
        prompt = "Describe this UI element in a few words, representing its function (e.g., 'search bar', 'submit button')."

        def process_box(i, box):
            xmin, ymin, xmax, ymax = map(int, box)
            
            # Crop the bounding box
            cropped_img = image.crop((xmin, ymin, xmax, ymax))
            
            # Convert PIL Image to Base64 JPEG string
            img_byte_arr = io.BytesIO()
            cropped_img.save(img_byte_arr, format='JPEG')
            base64_encoded = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

            # Send request to Ollama's generate API
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [base64_encoded],
                "stream": False
            }

            label = "unknown"
            try:
                logger.debug("Sending box %d to Ollama using model %s", i, self.model_name)
                start_time = time.time()
                response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=600)
                response.raise_for_status()
                data = response.json()
                label = data.get("response", "").strip()
                elapsed = time.time() - start_time
                logger.info("Ollama labelled box %d as '%s' in %.2fs", i, label, elapsed)
            except Exception as e:
                logger.error("Failed to generate label for box %d: %s", i, e)

            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0

            return {
                "id": i,
                "label": label,
                "bbox": [xmin, ymin, xmax, ymax],
                "center": [cx, cy]
            }

        # Execute API requests in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(process_box, i, box) for i, box in enumerate(boxes)]
            for future in concurrent.futures.as_completed(futures):
                try:
                    elements.append(future.result())
                except Exception as e:
                    logger.error("Unexpected error processing a box: %s", e)

        # ThreadPool completion order isn't guaranteed, so re-sort by the original ID
        elements.sort(key=lambda x: x["id"])

        return elements
