import time
from playwright.sync_api import sync_playwright

def main():
    print("Starting Playwright CDP server...")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--remote-debugging-port=9222",
                "--remote-debugging-address=0.0.0.0"
            ]
        )
        context = browser.new_context(viewport={'width': 1280, 'height': 720})
        page = context.new_page()
        page.goto("about:blank")
        print("CDP server running on port 9222 with default context/page")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping...")

if __name__ == "__main__":
    main()
