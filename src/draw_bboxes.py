import json
import argparse
from PIL import Image, ImageDraw

def main():
    parser = argparse.ArgumentParser(description="Draw bounding boxes from a JSON file onto a transparent image.")
    parser.add_argument("json_path", help="Path to the input JSON file")
    parser.add_argument("--output", "-o", help="Path to the output PNG image", default="test-screens/bboxes_overlay.png")
    args = parser.parse_args()

    json_path = args.json_path
    output_path = args.output

    # Load JSON data
    with open(json_path, "r") as f:
        data = json.load(f)

    matches = data.get("matches", [])
    if not matches:
        print("No matches found in the JSON file.")
        return

    # Determine image dimensions based on the maximum coordinates
    max_w = 0
    max_h = 0
    for match in matches:
        bbox = match.get("bbox")
        if bbox and len(bbox) == 4:
            max_w = max(max_w, int(bbox[2]))
            max_h = max(max_h, int(bbox[3]))

    # Add a tiny padding so the border doesn't get cut off exactly at the pixel limit
    max_w += 5
    max_h += 5

    # Create a completely transparent RGBA image
    img = Image.new("RGBA", (max_w, max_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw rectangles for each bounding box
    for match in matches:
        bbox = match.get("bbox")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            # Draw red rectangle with line width 3
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 255), width=3)

    img.save(output_path)
    print(f"Successfully generated transparent overlay at: {output_path}")

if __name__ == "__main__":
    main()
