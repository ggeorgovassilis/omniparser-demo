"""
Region detection using YOLO.
Wraps the YOLO model to detect UI element bounding boxes in a screenshot.
"""
import time
import logging
from PIL import Image

logger = logging.getLogger("omniparser")


class RegionDetector:
    """Detect UI element regions in a screenshot using YOLO."""

    def __init__(self, yolo_model, device: str):
        self.yolo_model = yolo_model
        self.device = device

    def detect(self, image: Image.Image) -> list[list[float]]:
        """Run YOLO inference and return bounding boxes.

        Args:
            image: Input PIL Image (RGB screenshot).

        Returns:
            List of [xmin, ymin, xmax, ymax] boxes in pixel coordinates.
        """
        logger.info("YOLO detection started")
        yolo_start = time.time()

        results = self.yolo_model.predict(image, conf=0.15, iou=0.1, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy().tolist()

        yolo_elapsed = time.time() - yolo_start
        logger.info("YOLO detection finished in %.2fs — found %d boxes", yolo_elapsed, len(boxes))
        return boxes
