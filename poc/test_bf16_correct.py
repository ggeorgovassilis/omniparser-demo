import torch
import time
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

device = "cpu"
print("Loading model float32...")
model32 = AutoModelForCausalLM.from_pretrained(
    "weights/icon_caption_florence", trust_remote_code=True, torch_dtype=torch.float32,
    revision="f6c1a25888ffc1d945ee8a1a77ac833c7303d46e", code_revision="f6c1a25888ffc1d945ee8a1a77ac833c7303d46e"
).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True, revision="5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac")

print("Loading model bfloat16...")
model16 = AutoModelForCausalLM.from_pretrained(
    "weights/icon_caption_florence", trust_remote_code=True, torch_dtype=torch.bfloat16,
    revision="f6c1a25888ffc1d945ee8a1a77ac833c7303d46e", code_revision="f6c1a25888ffc1d945ee8a1a77ac833c7303d46e"
).to(device)

image = Image.new("RGB", (32, 32), color="red")
txt = "<CAPTION>"

print("Testing FP32...")
t0 = time.time()
for _ in range(3):
    inputs = processor(text=txt, images=image, return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)
    model32.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=20, num_beams=1, early_stopping=False)
print(f"FP32: {time.time() - t0:.2f}s")

print("Testing BF16...")
t0 = time.time()
for _ in range(3):
    inputs = processor(text=txt, images=image, return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    model16.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=20, num_beams=1, early_stopping=False)
print(f"BF16: {time.time() - t0:.2f}s")
