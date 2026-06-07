import os
import time

# ── Thread limits are set via docker-compose environment variables ──
import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(2)

import warnings
import logging
from PIL import Image
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForCausalLM
from thefuzz import fuzz


logger = logging.getLogger("omniparser")

# Suppress Hugging Face generation warnings about beams/early_stopping which is safe to ignore here
warnings.filterwarnings("ignore", message=".*`num_beams` is set to 1.*`early_stopping` is set to `True`.*")

class OmniParser:
    def __init__(self, weights_dir="weights"):
        # Select hardware via OMNIPARSER_DEVICE env var (cpu | cuda | mps)
        device_name = os.environ.get("OMNIPARSER_DEVICE", "cpu").lower()
        if device_name == "cuda":
            self.device = "cuda"
            self.dtype = torch.float16
        elif device_name == "mps":
            self.device = "mps"
            self.dtype = torch.float16
        else:
            self.device = "cpu"
            self.dtype = torch.float32
        logger.info("Inference device: %s (dtype: %s)", self.device.upper(), str(self.dtype).split('.')[1])


        # Florence-2 remote code compat fix for newer transformers versions
        from transformers.configuration_utils import PretrainedConfig
        if not hasattr(PretrainedConfig, "forced_bos_token_id"):
            PretrainedConfig.forced_bos_token_id = None

        logger.info("Loading YOLO detection model...")
        yolo_path = os.path.join(weights_dir, "icon_detect", "model.pt")
        self.yolo_model = YOLO(yolo_path)
        self.yolo_model.to(self.device)

        logger.info("Loading Florence captioning model...")
        florence_path = os.path.join(weights_dir, "icon_caption_florence")
        self.processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-base",
            trust_remote_code=True,
            revision="5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac",
            code_revision="5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac"
        )
        # Load directly onto the GPU in the appropriate precision
        self.caption_model = AutoModelForCausalLM.from_pretrained(
            florence_path,
            trust_remote_code=True,
            revision="f6c1a25888ffc1d945ee8a1a77ac833c7303d46e",
            code_revision="f6c1a25888ffc1d945ee8a1a77ac833c7303d46e",
            torch_dtype=self.dtype
        ).to(self.device)

        # Apply int8 dynamic quantization on CPU for 2-4x faster inference
        if self.device == "cpu":
            logger.info("Applying dynamic int8 quantization for CPU inference...")
            self.caption_model = torch.quantization.quantize_dynamic(
                self.caption_model, {torch.nn.Linear}, dtype=torch.qint8
            )

        logger.info("Models loaded successfully.")


    def parse_screen(self, image: Image.Image):
        # 1. Detect bounding boxes with YOLO
        logger.info("YOLO detection started")
        yolo_start = time.time()

        results = self.yolo_model.predict(image, conf=0.15, iou=0.1, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy().tolist()  # [xmin, ymin, xmax, ymax]

        yolo_elapsed = time.time() - yolo_start
        logger.info("YOLO detection finished in %.2fs — found %d boxes", yolo_elapsed, len(boxes))

        # 2. Crop image and caption each box in a single batched Florence forward pass
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

        # Florence batch inference in sub-batches to limit peak memory.
        # A full batch of 40+ varied-size crops creates huge padded tensors.
        # Processing 8 at a time keeps memory low without losing much throughput.
        task_prompt = "<CAPTION>"
        BATCH_SIZE = 1

        for batch_start in range(0, len(crops), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(crops))
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
                    max_new_tokens=10,
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

    def ocr_region(self, image: Image.Image, bbox: list):
        """
        Extract text from a specific region of the screen using Florence-2 OCR.

        Args:
            image: Full screenshot PIL Image
            bbox: [x1, y1, x2, y2] pixel coordinates of the region

        Returns:
            str: The extracted text from the region
        """
        x1, y1, x2, y2 = map(int, bbox)
        logger.info("OCR region: [%d, %d, %d, %d]", x1, y1, x2, y2)

        crop = image.crop((x1, y1, x2, y2))
        w, h = crop.width, crop.height

        task_prompt = "<OCR>"
        inputs = self.processor(
            text=[task_prompt],
            images=[crop],
            return_tensors="pt"
        ).to(self.device)

        if self.device != "cpu":
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

        with torch.no_grad():
            generated_ids = self.caption_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=100,
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
        text = parsed[task_prompt].strip()
        logger.info("OCR result: '%s'", text)
        return text

    def find_elements(self, image: Image.Image, prompt: str, threshold: int = 50):
        """
        Parses the screen and uses fuzzy string matching to find UI elements
        whose generated labels match the user's prompt.
        """
        logger.info("find_elements called with prompt: '%s'", prompt)
        elements = self.parse_screen(image)

        matches = []
        for el in elements:
            # Fuzzy match the predicted label against the user target prompt
            score = fuzz.token_set_ratio(prompt.lower(), el["label"].lower())
            if score >= threshold:
                el["match_score"] = score
                matches.append(el)

        # Sort by best match score descending
        matches.sort(key=lambda x: x["match_score"], reverse=True)
        logger.info("find_elements: %d matches found for '%s'", len(matches), prompt)
        return matches

# Singleton pattern for the API layer to reuse the loaded model
_parser_instance = None

def get_parser() -> OmniParser:
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = OmniParser()
    return _parser_instance
