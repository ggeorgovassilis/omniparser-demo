# AI Screen Recognition

A Dockerized API service that wraps Microsoft's [OmniParser](https://github.com/microsoft/OmniParser) to detect and locate UI elements on a screen based on text prompts. You submit a screenshot and a text prompt (e.g., "find the submit button"), and the API returns the bounding boxes of the matching UI elements.

## Architecture

The project consists of three main components running entirely inside a CPU-optimized Docker container:

1. **API Layer (FastAPI)**: An HTTP web server exposing a `POST /parse` endpoint. It accepts an image upload and a target prompt.
2. **OmniParser Engine**: 
   * **Detection**: Utilizes a YOLO model to find all interactable UI regions (bounding boxes) on the screen.
   * **Captioning**: Uses Florence-2 to generate a functional textual label (e.g., "search bar", "submit button") for each detected region.
3. **Matcher**: A lightweight fuzzy string matching utility that compares the user's prompt against all generated labels and returns the best matches along with their coordinates.

## Installation

You do not need to install Python or ML libraries on your host machine. Everything runs through Docker.

1. **Clone the repository** (if you haven't already) and navigate to the directory:
   ```bash
   cd ai-screen-recognition
   ```

2. **Download Model Weights**: 
   Run the following command to spin up a temporary Docker container that fetches the required OmniParser models (YOLO and Florence-2) from Hugging Face. The weights will be saved to a local `weights/` directory mapped as a volume.
   ```bash
   docker compose build
   docker compose run --rm omniparser python download_weights.py
   ```

## Running the Service

1. **Start the API server**:
   ```bash
   docker compose up -d
   ```
   The service will boot up and load the models into memory. 

2. **Check the status**:
   Ensure the API is ready by checking the health endpoint:
   ```bash
   curl http://localhost:8000/health
   # Expected output: {"status": "ok"}
   ```

3. **Test the API**:
   You can easily test the API using the built-in Swagger UI provided by FastAPI.
   * Open your browser and navigate to: http://localhost:8000/docs
   * Locate the `POST /parse` endpoint.
   * Click "Try it out", upload a screenshot, enter a prompt (like "submit button"), and hit Execute!

   **Testing from the command line:**
   If you have a `test.png` image in your project directory (like a Google homepage screenshot), you can use `curl` to send a request finding the coordinates for a specific element. For example, to find the search box:
   ```bash
   curl -X POST "http://localhost:8000/parse" \
        -F "prompt=search box" \
        -F "image=@test.png"
   ```
   Or to find the search button:
   ```bash
   curl -X POST "http://localhost:8000/parse" \
        -F "prompt=search button" \
        -F "image=@test.png"
   ```
