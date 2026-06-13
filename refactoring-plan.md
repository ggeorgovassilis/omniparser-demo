## Plan: Module Separation Refactoring

**Steps**
1. **Create `src/models.py` (Shared Models)**
   - Extract the Florence-2 and YOLO model loading into a shared registry, or instantiate them centrally in `OmniParser` to inject them below. This prevents duplicating model weights in memory.
2. **Create `src/detection.py`**
   - Implement a `RegionDetector` class.
   - Move YOLO `predict` logic from `parse_screen` into a method: `detect(image: Image.Image) -> list[list[float]]` (returns the bounding boxes).
3. **Create `src/labelling.py`**
   - Implement a `RegionLabeler` class taking the Florence model.
   - Move the `<CAPTION>` iteration, batched image cropping, and tensor operations from `OmniParser.parse_screen` into a method: `label_regions(image: Image.Image, boxes: list) -> list[dict]`.
4. **Create `src/ocr.py`**
   - Implement a `RegionOCR` class taking the Florence model.
   - Move `ocr_region`, `_ocr_single_tile`, and `_postprocess_ocr` from `OmniParser` to this module.
   - Core API: `extract_text(image: Image.Image, bbox: list) -> str`.
5. **Refactor parser.py (Orchestrator)**
   - `OmniParser` becomes a clean orchestration layer.
   - `__init__` sets up the models and initializes `RegionDetector`, `RegionLabeler`, and `RegionOCR` via dependency injection.
   - `parse_screen` becomes simply: `boxes = self.detector.detect(...)` followed by `elements = self.labeler.label_regions(...)`.
   - `ocr_region` directly delegates to `self.ocr.extract_text(...)`.

**Relevant files**
- New files: src/models.py, src/detection.py, src/labelling.py, src/ocr.py
- Modified file: parser.py — `OmniParser` class simplifies to orchestration.

**Verification**
1. Restart the server (`docker-compose up`) to ensure models load cleanly without Out-of-Memory (OOM) errors.
2. Hit the `/parse` endpoint with a test image and verify YOLO + Labelling pipeline successfully finishes.
3. Hit the `/ocr` endpoint with a bounding box (and without) to test Florence tiling logic.

Let me know if this looks good and you are ready to proceed with these changes!