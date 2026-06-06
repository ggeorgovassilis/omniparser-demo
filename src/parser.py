import os
import torch
import warnings
from PIL import Image
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForCausalLM
from thefuzz import fuzz

# Suppress Hugging Face generation warnings about beams/early_stopping which is safe to ignore here
warnings.filterwarnings("ignore", message=".*`num_beams` is set to 1.*`early_stopping` is set to `True`.*")

class OmniParser:
    def __init__(self, weights_dir="weights"):
        self.device = "cpu"
        
        # Florence-2 remote code compat fix for newer transformers versions
        from transformers.configuration_utils import PretrainedConfig
        if not hasattr(PretrainedConfig, "forced_bos_token_id"):
            PretrainedConfig.forced_bos_token_id = None

        print("Loading YOLO detection model...")
        yolo_path = os.path.join(weights_dir, "icon_detect", "model.pt")
        self.yolo_model = YOLO(yolo_path)
        
        print("Loading Florence captioning model...")
        florence_path = os.path.join(weights_dir, "icon_caption_florence")
        self.processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-base", 
            trust_remote_code=True,
            revision="5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac",
            code_revision="5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac"
        )
        self.caption_model = AutoModelForCausalLM.from_pretrained(
            florence_path, 
            trust_remote_code=True,
            revision="f6c1a25888ffc1d945ee8a1a77ac833c7303d46e",
            code_revision="f6c1a25888ffc1d945ee8a1a77ac833c7303d46e",
            torch_dtype=torch.float32  # CPU inference
        ).to(self.device)
        
        print("Applying dynamic quantization for faster CPU inference...")
        self.caption_model = torch.quantization.quantize_dynamic(
            self.caption_model, {torch.nn.Linear}, dtype=torch.qint8
        )
        
        print("Models loaded successfully.")
        
    def parse_screen(self, image: Image.Image):
        # 1. Detect bounding boxes with YOLO
        # Using raised detection thresholds to cull garbage boxes and reduce inference time
        results = self.yolo_model.predict(image, conf=0.15, iou=0.1, verbose=False) 
        boxes = results[0].boxes.xyxy.cpu().numpy().tolist() # [xmin, ymin, xmax, ymax]
        
        # 2. Crop image and caption each box
        elements = []
        for i, box in enumerate(boxes):
            xmin, ymin, xmax, ymax = map(int, box)
            
            # Crop the detected region.
            cropped_img = image.crop((xmin, ymin, xmax, ymax))
            
            # Generate caption using Florence (using <CAPTION> task)
            task_prompt = "<CAPTION>" 
            inputs = self.processor(text=task_prompt, images=cropped_img, return_tensors="pt").to(self.device)
            
            generated_ids = self.caption_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=20,
                num_beams=1,
                early_stopping=False
            )
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            # Clean up Florence output
            parsed_text = self.processor.post_process_generation(
                generated_text, 
                task=task_prompt, 
                image_size=(cropped_img.width, cropped_img.height)
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
            
        return elements

    def find_elements(self, image: Image.Image, prompt: str, threshold: int = 50):
        """
        Parses the screen and uses fuzzy string matching to find UI elements
        whose generated labels match the user's prompt.
        """
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
        return matches

# Singleton pattern for the API layer to reuse the loaded model
_parser_instance = None

def get_parser() -> OmniParser:
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = OmniParser()
    return _parser_instance
