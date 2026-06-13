"""
Shared model registry and loading.
Centralizes Florence-2 and YOLO model instantiation to prevent
duplicating model weights in memory when shared across components.
"""
import os
import logging

import torch
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForCausalLM

logger = logging.getLogger("omniparser")


def load_models(weights_dir: str = "weights"):
    """Load and return YOLO + Florence-2 models shared across components.

    Returns a dict with keys:
        device, dtype, yolo_model, processor, caption_model
    """
    # ── Hardware selection ──
    device_name = os.environ.get("OMNIPARSER_DEVICE", "cpu").lower()
    if device_name == "cuda":
        device = "cuda"
        dtype = torch.float16
    elif device_name == "mps":
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32
    logger.info("Inference device: %s (dtype: %s)", device.upper(), str(dtype).split('.')[1])

    # ── Florence-2 remote code compat fix ──
    from transformers.configuration_utils import PretrainedConfig
    if not hasattr(PretrainedConfig, "forced_bos_token_id"):
        PretrainedConfig.forced_bos_token_id = None

    # ── YOLO detection model ──
    logger.info("Loading YOLO detection model...")
    yolo_path = os.path.join(weights_dir, "icon_detect", "model.pt")
    yolo_model = YOLO(yolo_path)
    yolo_model.to(device)

    # ── Florence captioning / OCR model ──
    logger.info("Loading Florence captioning model...")
    florence_path = os.path.join(weights_dir, "icon_caption_florence")
    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-base",
        trust_remote_code=True,
        revision="5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac",
        code_revision="5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac"
    )
    caption_model = AutoModelForCausalLM.from_pretrained(
        florence_path,
        trust_remote_code=True,
        revision="f6c1a25888ffc1d945ee8a1a77ac833c7303d46e",
        code_revision="f6c1a25888ffc1d945ee8a1a77ac833c7303d46e",
        torch_dtype=dtype
    ).to(device)

    # Apply int8 dynamic quantization on CPU for 2-4x faster inference
    if device == "cpu":
        logger.info("Applying dynamic int8 quantization for CPU inference...")
        caption_model = torch.quantization.quantize_dynamic(
            caption_model, {torch.nn.Linear}, dtype=torch.qint8
        )

    logger.info("Models loaded successfully.")
    return {
        "device": device,
        "dtype": dtype,
        "yolo_model": yolo_model,
        "processor": processor,
        "caption_model": caption_model,
    }
