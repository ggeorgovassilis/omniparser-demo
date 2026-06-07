import sys
import json
import subprocess
from playwright.sync_api import sync_playwright

def main():
    if len(sys.argv) < 2:
        print("Usage: python browser_cli.py <command> [args]")
        sys.exit(1)
        
    cmd = sys.argv[1]
    args = sys.argv[2:]

    with sync_playwright() as p:
        try:
            browser = p.chromium.connect_over_cdp("http://localhost:9222")
            context = browser.contexts[0]
            if not context.pages:
                page = context.new_page()
            else:
                page = context.pages[0] # Simply use the first page
                
            if cmd == "goto":
                url = args[0]
                print(f"Navigating to {url}...")
                page.goto(url, wait_until="networkidle")
                print("Done.")

            elif cmd == "observe":
                screenshot_path = "/app/poc/screenshot.png"
                output_json = "/app/poc/boxes.json"
                print("Capturing screenshot...")
                page.screenshot(path=screenshot_path)
                
                print("Parsing with OmniParser...")
                result = subprocess.run([
                    "curl", "-s", "-X", "POST", "http://omniparser:8000/parse",
                    "-H", "Expect:",
                    "-F", f"image=@{screenshot_path}"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    try:
                        data = json.loads(result.stdout)
                        matches = data.get("matches", [])
                        if not matches:
                            print("No UI elements detected.")
                        else:
                            print("\n--- Detected UI Elements ---")
                            for match in matches:
                                bbox = match.get("bbox", [0, 0, 0, 0])
                                # Calculate center point
                                cx = (bbox[0] + bbox[2]) / 2
                                cy = (bbox[1] + bbox[3]) / 2
                                match["center"] = [cx, cy]
                                print(f"[{match.get('id')}] {match.get('label')} -> BBox: {bbox}, Center: ({cx:.1f}, {cy:.1f})")
                            print("----------------------------\n")

                        with open(output_json, "w") as f:
                            json.dump(data, f, indent=2)
                        print(f"Success. Parsed elements saved to {output_json}")
                        
                    except json.JSONDecodeError:
                        print(f"Error parsing OmniParser JSON output. Raw output:\n{result.stdout[:200]}")
                else:
                    print(f"Error communicating with OmniParser. Return code: {result.returncode}")

            elif cmd == "click":
                x, y = float(args[0]), float(args[1])
                print(f"Clicking at ({x}, {y})...")
                page.mouse.click(x, y)
                print("Done.")

            elif cmd == "type":
                text = " ".join(args)
                print(f"Typing '{text}'...")
                page.keyboard.type(text)
                print("Done.")
                
            elif cmd == "tabs":
                for i, p in enumerate(context.pages):
                    try:
                        title = p.title() or p.url
                    except Exception:
                        title = p.url
                    print(f"[{i}] {title}")
                    
            elif cmd == "tab":
                idx = int(args[0])
                pages = context.pages
                if 0 <= idx < len(pages):
                    pages[idx].bring_to_front()
                    print(f"Switched to tab {idx}: {pages[idx].title() or pages[idx].url}")
                else:
                    print(f"Invalid index. Specify between 0 and {len(pages)-1}")

            elif cmd == "ocr":
                screenshot_path = "/app/poc/screenshot.png"

                import os as _os
                if not _os.path.exists(screenshot_path):
                    print("No screenshot found. Run 'observe' first to capture the page.")
                else:
                    curl_cmd = [
                        "curl", "-s", "-X", "POST", "http://omniparser:8000/ocr",
                        "-H", "Expect:",
                        "-F", f"image=@{screenshot_path}"
                    ]
                    if len(args) >= 1:
                        bbox = args[0]
                        curl_cmd.extend(["-F", f"bbox={bbox}"])
                        print(f"Running OCR on region {bbox} (using last observe screenshot)...")
                    else:
                        print("Running OCR on full screen (using last observe screenshot)...")

                    result = subprocess.run(curl_cmd, capture_output=True, text=True)

                    if result.returncode == 0:
                        try:
                            data = json.loads(result.stdout)
                            if "error" in data:
                                print(f"OCR error: {data['error']}")
                            else:
                                print(f"OCR text: {data.get('text', '')}")
                        except json.JSONDecodeError:
                            print(f"Error parsing OCR response. Raw output:\n{result.stdout[:200]}")
                    else:
                        # Include HTTP status and response body for debugging
                        stderr_info = f" | stderr: {result.stderr[:200]}" if result.stderr else ""
                        print(f"Error communicating with OmniParser. HTTP {result.returncode}{stderr_info}")
                        if result.stdout:
                            try:
                                err_data = json.loads(result.stdout)
                                detail = err_data.get("detail", "")
                                if detail:
                                    print(f"Server detail: {detail}")
                            except json.JSONDecodeError:
                                print(f"Response: {result.stdout[:300]}")
            else:
                print(f"Unknown command: {cmd}")
        except Exception as e:
            print(f"Failed to execute command: {e}")

if __name__ == "__main__":
    main()
