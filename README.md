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
   docker compose run --rm omniparser python src/download_weights.py
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

   If you want to get the unfiltered list of **all** detected UI elements on the screen, just omit the prompt:
   ```bash
   curl -X POST "http://localhost:8000/parse" \
        -F "image=@test-screens/test.png"
   ```

## Tools

### Drawing Bounding Boxes

You can generate a transparent image overlay containing red bounding boxes to easily visualize the results returned by the API. The `src/draw_bboxes.py` tool reads a JSON file containing the UI elements and outputs a tightly-fitted transparent `.png` with the drawn rectangles.

To run the script against an existing `result.json` (output of the curl call)   file inside `test-screens/`, mount the directory into a temporary container:

```bash
docker compose run --rm -v $(pwd)/test-screens:/test-screens omniparser python src/draw_bboxes.py /test-screens/result.json
```

(By default, this will generate a file at `test-screens/bboxes_overlay.png`)

## PoC: Visual Browser Automation

This project includes a Proof of Concept (PoC) demonstrating how a browser can be operated strictly via visual recognition without relying on its DOM. The architecture for this PoC utilizes a 2-container setup:

1. **`omniparser`**: The existing component that acts as the "eyes", converting screenshots into spatial bounding boxes and descriptive text labels.
2. **`browser-env`**: A new container based on `mcr.microsoft.com/playwright:python` which bundles Chromium and an interactive Python shell (`poc/browser_shell.py`). This acts as the "hands". 

The agent running the script only interacts via raw coordinates (e.g., clicking X, Y) and keyboard inputs, ensuring it does not "cheat" by using HTML DOM elements.

### Running the PoC

1. **Start the containers**
   ```bash
   docker compose up -d
   ```
   *This starts both the existing `omniparser` API and the new `browser-env` headless container.*

2. **Launch the interactive shell**
   Exec into the `browser-env` container to run the interactive loop:
   ```bash
   docker compose exec -it browser-env python /app/poc/browser_shell.py
   ```

3. **Operate the browser**
   Inside the `browser>` prompt, try running the following testing scenario sequence manually:
   - `goto https://google.com`
   - `observe` (This captures a screenshot, talks to OmniParser, and saves UI elements to `poc/boxes.json`)
   - Open `poc/boxes.json` locally. Find the exact matching coordinates for the Search Input field (e.g. its center point).
   - Use `click <x> <y>` to focus the input box.
   - Use `type AI visual browser automation` to write text.
   - Use `observe` again to see the updated screen and find the "Google Search" button coordinates.
   - Use `click <x> <y>` to run the search.
