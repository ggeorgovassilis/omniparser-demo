from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
import asyncio
import io
import time
import logging
from PIL import Image

# Import the parser
from parser import get_parser

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("omniparser")

# ── Coalescing state ──
# When a /parse arrives while another is already running, the new request
# waits for the in-flight parse to finish and returns its cached result.
# This prevents duplicate inference work when the tester agent re-submits.
_parser_busy = False
_parse_done_event = asyncio.Event()
_parse_done_event.set()  # start as "done" so the first request proceeds
_last_parse_result = None
_last_parse_error = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load the models in memory when the server starts
    get_parser()
    yield

app = FastAPI(title="OmniParser UI Element Detector", lifespan=lifespan)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"Started {request.method} {request.url.path}")
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Finished {request.method} {request.url.path} - {response.status_code} - {process_time:.2f}s")
    return response

@app.get("/health")
async def health_check():
    return {"status": "busy" if _parser_busy else "idle"}

@app.post("/parse")
async def parse_ui(
    image: UploadFile = File(...),
    prompt: Optional[str] = Form(None)
):
    """
    Provide an image screen and an optional prompt like "find the submit button".
    If prompt is provided, returns matching UI elements.
    If prompt is omitted, returns all detected UI elements.

    If a parse is already in progress, this request coalesces: it waits for
    the in-flight parse to complete and returns its cached result.
    """
    global _parser_busy, _parse_done_event, _last_parse_result, _last_parse_error

    # ── Coalesce if busy ──
    if _parser_busy:
        logger.info("Parse already in progress — coalescing request, waiting for result")
        await _parse_done_event.wait()
        if _last_parse_error is not None:
            return JSONResponse(
                status_code=500,
                content={"error": str(_last_parse_error), "coalesced": True}
            )
        return JSONResponse(content={"matches": _last_parse_result, "coalesced": True})

    # ── Normal (non-coalesced) path ──
    _parse_done_event.clear()
    _parser_busy = True

    try:
        logger.info(f"Parsing image: {image.filename} | Prompt: {prompt}")
        # Read uploaded image into memory
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Run inference
        parser = get_parser()
        if prompt:
            matches = await asyncio.to_thread(parser.find_elements, pil_image, prompt)
        else:
            matches = await asyncio.to_thread(parser.parse_screen, pil_image)

        # Cache the successful result for potential coalescers
        _last_parse_result = matches
        _last_parse_error = None

        return JSONResponse(content={"matches": matches})

    except Exception as e:
        # Cache the error for potential coalescers
        _last_parse_result = None
        _last_parse_error = e
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        _parser_busy = False
        _parse_done_event.set()  # wake all waiting coalescers

@app.post("/ocr")
async def ocr_region(
    image: UploadFile = File(...),
    bbox: Optional[str] = Form(None)
):
    """
    Extract text from a specific region of the screen using Florence-2 OCR.
    If bbox is omitted, OCR the full image.

    Args:
        image: Uploaded screenshot
        bbox: Optional comma-separated region coordinates "x1,y1,x2,y2"
    """
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

        if bbox is None:
            # Full-screen OCR
            coords = [0, 0, pil_image.width, pil_image.height]
            logger.info("OCR full screen: %dx%d", pil_image.width, pil_image.height)
        else:
            coords = [int(x.strip()) for x in bbox.split(",")]
            if len(coords) != 4:
                return JSONResponse(
                    status_code=400,
                    content={"error": "bbox must be 4 comma-separated integers: x1,y1,x2,y2"}
                )
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"error": "bbox values must be integers: x1,y1,x2,y2"}
        )

    try:
        parser = get_parser()
        text = await asyncio.to_thread(parser.ocr_region, pil_image, coords)

        return JSONResponse(content={"text": text})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
