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

# Track whether a parse task is currently running
_parser_busy = False

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
    """
    global _parser_busy
    try:
        _parser_busy = True
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
        
        return JSONResponse(content={"matches": matches})
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        _parser_busy = False

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
