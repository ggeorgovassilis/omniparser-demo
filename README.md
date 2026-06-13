# This project

TL;DR: If you hate reading long documentation, like I do, I recommend you ask your AI to explain this project to you.

A Dockerized API service that wraps Microsoft's [OmniParser](https://github.com/microsoft/OmniParser) to detect and locate UI elements on a screen based on text prompts. You submit a screenshot and a text prompt (e.g., "find the submit button"), and the API returns the bounding boxes of the matching UI elements.

As a demo of its capabilities, there is an agent which operates a chromium browser (included) only through visual parsing of the page, bypassing the DOM.

## Architecture

The project consists of four main components:

1. **API Layer (FastAPI)**: An HTTP web server exposing a `POST /parse` endpoint. It accepts an image upload and a target prompt.
2. **OmniParser Engine**: 
   * **Detection**: Utilizes a YOLO model to find all interactable UI regions (areas of interest) on the screen.
   * **Labelling**: Uses Florence-2 to generate a functional textual label (e.g., "search bar", "submit button") for each detected area.
   * **OCR**: Extracts raw text from specific bounding box areas or the entire screen using optical character recognition.
3. **Matcher**: A lightweight fuzzy string matching utility that compares the user's prompt against all generated labels and returns the best matches along with their coordinates.
4. **Visual Tester Agent**: A VS Code custom agent (configured in `.github/agents/visual_tester.agent.md`) that drives the headless browser strictly through visual recognition — without touching the DOM. It captures screenshots, calls `/parse` to detect UI elements, clicks coordinates, and extracts text via OCR. A PreToolUse guard hook (`.github/hooks/browser-guard.sh`) enforces that the agent can only interact through `browser_cli.py`, preventing DOM-based shortcuts.

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
   # Expected output: {"status": "idle"}
   ```

3. **Test the API**:
   You can easily test the API using the built-in Swagger UI provided by FastAPI.
   * Open your browser and navigate to: http://localhost:8000/docs
   * Locate the `POST /parse` endpoint.
   * Click "Try it out", upload a screenshot, enter a prompt (like "submit button"), and hit Execute!

   **Testing from the command line:**
   If you have a `test.png` image in your project directory (like a Google homepage screenshot), you can use `curl` to send a request finding the coordinates for a specific element.

   ```bash
   curl -X POST "http://localhost:8000/parse" \
        -F "image=@test-screens/test.png"
   ```

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

### Running Automated Visual Tests (CLI)

Instead of the interactive shell, you can drive the browser programmatically using `poc/browser_cli.py`. This is ideal for scripting and automating end-to-end visual tests. All commands share the same browser session (they reconnect via CDP each call).

**Available commands:**

| Command | Description |
|---------|-------------|
| `goto <url>` | Navigate the browser to a URL and wait for network idle |
| `observe` | Take a screenshot, send it to OmniParser, and list all detected UI elements with their bounding boxes and center coordinates |
| `click <x> <y>` | Click at pixel coordinates (typically the center point from `observe`) |
| `type <text>` | Type text via keyboard input into the focused element |
| `ocr` | Run OCR on the full page (requires a prior `observe`) |
| `ocr <x1>,<y1>,<x2>,<y2>` | Run OCR on a specific region of the last screenshot |

#### Walkthrough: Visual e-Commerce Price Extraction Test

This test demonstrates a full visual automation flow on a real website — dismissing a cookie banner, searching for a product, selecting a suggestion, and extracting price information via OCR — all without touching the DOM.

**Step 0 — Note the start time:**

```bash
date -u '+%Y-%m-%d %H:%M:%S'
```

**Step 1 — Navigate to the target page:**

```bash
docker compose exec browser-env python /app/poc/browser_cli.py goto https://www.telekom.de/start
```

**Step 2 — Dismiss the cookie banner:**

```bash
# Observe the page to find the cookie accept button
docker compose exec browser-env python /app/poc/browser_cli.py observe

# Click the cookie accept button (use the center coordinates from observe output)
docker compose exec browser-env python /app/poc/browser_cli.py click <cookie_accept_x> <cookie_accept_y>
```

**Step 3 — Click the search icon (magnifying glass):**

```bash
# Observe to find the search icon
docker compose exec browser-env python /app/poc/browser_cli.py observe

# Click the magnifying glass icon — a search text box will appear at the top
docker compose exec browser-env python /app/poc/browser_cli.py click <search_icon_x> <search_icon_y>
```

**Step 4 — Type the search query:**

```bash
# Observe to find the search text box that appeared
docker compose exec browser-env python /app/poc/browser_cli.py observe

# Click into the search text box
docker compose exec browser-env python /app/poc/browser_cli.py click <search_box_x> <search_box_y>

# Type the search term — this will trigger search suggestions
docker compose exec browser-env python /app/poc/browser_cli.py type galaxy
```

**Step 5 — Select a suggestion and run the search:**

```bash
# Observe to see the search suggestions list
docker compose exec browser-env python /app/poc/browser_cli.py observe

# Click "Samsung Galaxy S26" from the suggestions to execute the search
docker compose exec browser-env python /app/poc/browser_cli.py click <suggestion_x> <suggestion_y>
```

**Step 6 — Extract the price via OCR:**

```bash
# Observe the search results page
docker compose exec browser-env python /app/poc/browser_cli.py observe

# Run OCR to read the price. The bottom-right shows the price under "MONATLICH".
# Use the bounding box from observe for that region to narrow the OCR.
docker compose exec browser-env python /app/poc/browser_cli.py ocr <price_region_x1>,<price_region_y1>,<price_region_x2>,<price_region_y2>
```

**Expected result:** The OCR output should contain a price in € (e.g., "49,95 €") under the "MONATLICH" label.

**Step 7 — Note the end time and compute duration:**

```bash
date -u '+%Y-%m-%d %H:%M:%S'
```

Subtract the start time from the end time to get the total test duration in `HH:MM:SS`.

```bash
date -u '+%Y-%m-%d %H:%M:%S'
```

#### Tips for Reliable Tests

* **Always `observe` after navigation or interaction**: The visual state changes after clicks and page loads. Re-run `observe` to get fresh UI element coordinates.
* **Use center coordinates**: The `observe` command prints center points (`Center: (x, y)`) for each element — use these with `click`.
* **OCR requires a prior `observe`**: The `ocr` command operates on the last screenshot captured by `observe`, so the page content must be current.
* **OCR for specific regions**: Narrow OCR to a bounding box region (e.g., `ocr 100,200,300,400`) for more precise text extraction when the full page contains too much noise.
