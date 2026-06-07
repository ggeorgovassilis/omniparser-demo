import os
import time
from playwright.sync_api import sync_playwright

MOBILE_MODE = os.environ.get("MOBILE_MODE", "").lower() in ("1", "true", "yes")

def main():
    # Lower process priority — Chrome yields CPU to the parser on contention
    try:
        os.nice(15)
    except AttributeError:
        pass  # os.nice only exists on Unix

    print("Starting Playwright CDP server...")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--remote-debugging-port=9222",
                "--remote-debugging-address=0.0.0.0",
                "--renderer-process-limit=1",    # Keep child process count down
                "--disable-background-networking",  # No sync/telemetry when idle
            ]
        )

        if MOBILE_MODE:
            device = p.devices["Galaxy S9+"]
            context = browser.new_context(**device)
            print(f"Running in mobile mode ({device['viewport']})")
        else:
            context = browser.new_context(viewport={"width": 1280, "height": 2000})
            print("Running in desktop mode (1280x2000)")

        page = context.new_page()
        page.goto("about:blank")
        print("CDP server running on port 9222")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping...")

if __name__ == "__main__":
    main()
