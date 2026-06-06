from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import io
from PIL import Image

# Import the parser
from parser import get_parser

app = FastAPI(title="OmniParser UI Element Detector")

@app.on_event("startup")
async def startup_event():
    # Pre-load the models in memory when the server starts
    get_parser()

@app.get("/health")
async def health_check():
    return {"status": "ok"}

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
    try:
        # Read uploaded image into memory
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Run inference
        parser = get_parser()
        if prompt:
            matches = parser.find_elements(pil_image, prompt)
        else:
            matches = parser.parse_screen(pil_image)
        
        return JSONResponse(content={"matches": matches})
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
