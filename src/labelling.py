"""
Region labelling using Florence-2 captioning.
Crops each detected region and generates a label via batched Florence inference.
"""
import os
import time
import logging
import torch
from PIL import Image

logger = logging.getLogger("omniparser")


class RegionLabeler:
    """Label detected UI regions with Florence-2 captions."""

    def __init__(self, processor, caption_model, device: str, dtype):
        self.processor = processor
        self.caption_model = caption_model
        self.device = device
        self.dtype = dtype

    def label_regions(self, image: Image.Image, boxes: list[list[float]]) -> list[dict]:
        """Crop each bounding box from the image and generate a label.

        Args:
            image: Full screenshot PIL Image.
            boxes: List of [xmin, ymin, xmax, ymax] boxes.

        Returns:
            List of dicts with keys: id, label, bbox, center.
        """
        logger.info("Florence captioning started for %d crops", len(boxes))
        florence_start = time.time()

        elements = []
        crops = []
        crop_sizes = []
        bbox_info = []

        for i, box in enumerate(boxes):
            xmin, ymin, xmax, ymax = map(int, box)
            cropped_img = image.crop((xmin, ymin, xmax, ymax))
            crops.append(cropped_img)
            crop_sizes.append((cropped_img.width, cropped_img.height))
            bbox_info.append((i, xmin, ymin, xmax, ymax))

        task_prompt = "<CAPTION>"
        FLORENCE_BATCH_SIZE = int(os.environ.get("FLORENCE_BATCH_SIZE", "1"))

        for batch_start in range(0, len(crops), FLORENCE_BATCH_SIZE):
            batch_end = min(batch_start + FLORENCE_BATCH_SIZE, len(crops))
            batch_crops = crops[batch_start:batch_end]
            batch_sizes = crop_sizes[batch_start:batch_end]
            batch_info = bbox_info[batch_start:batch_end]

            inputs = self.processor(
                text=[task_prompt] * len(batch_crops),
                images=batch_crops,
                return_tensors="pt"
            ).to(self.device)

            if self.device != "cpu":
                inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

            with torch.no_grad():
                generated_ids = self.caption_model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=int(os.environ.get("FLORENCE_MAX_TOKENS_CAPTION", "10")),
                    num_beams=1,
                    early_stopping=False
                )

            batch_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )

            for idx, generated_text in enumerate(batch_text):
                i, xmin, ymin, xmax, ymax = batch_info[idx]
                w, h = batch_sizes[idx]

                parsed_text = self.processor.post_process_generation(
                    generated_text,
                    task=task_prompt,
                    image_size=(w, h)
                )
                label = parsed_text[task_prompt].strip()

                cx = (xmin + xmax) / 2.0
                cy = (ymin + ymax) / 2.0

                elements.append({
                    "id": i,
                    "label": label,
                    "bbox": [xmin, ymin, xmax, ymax],
                    "center": [cx, cy]
                })

            # Free intermediate tensors from this sub-batch
            del inputs, generated_ids

        florence_elapsed = time.time() - florence_start
        logger.info("Florence captioning finished in %.2fs (%d elements, avg %.2fs per crop)",
                     florence_elapsed, len(elements),
                     florence_elapsed / len(elements) if elements else 0)

        return elements
