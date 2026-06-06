import torch
import time
import os
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor

device = "cpu"
model = AutoModelForCausalLM.from_pretrained(
    "weights/icon_caption_florence", trust_remote_code=True, torch_dtype=torch.float32,
    revision="f6c1a25888ffc1d945ee8a1a77ac833c7303d46e", code_revision="f6c1a25888ffc1d945ee8a1a77ac833c7303d46e"
).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True, revision="5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac")

images = [Image.new("RGB", (32, 32), color="red") for _ in range(8)]
texts = ["<CAPTION>" for _ in range(8)]

print(f"Testing Sequential Processing (8 images)...")
t0 = time.time()
for im, txt in zip(images, texts):
    inputs = processor(text=txt, images=im, return_tensors="pt").to(device)
    model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=20,
        num_beams=1,
        early_stopping=False
    )
print(f"Sequential Time: {time.time() - t0:.2f}s")

def process_one(args):
    im, txt = args
    # We set intraop threads to 1 for the parallel workers so they don't thrash each other
    torch.set_num_threads(1)
    inputs = processor(text=txt, images=im, return_tensors="pt").to(device)
    return model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=20,
        num_beams=1,
        early_stopping=False
    )

print("Testing ThreadPoolExecutor (8 images)...")
t0 = time.time()
with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    list(executor.map(process_one, zip(images, texts)))
print(f"ThreadPool Time: {time.time() - t0:.2f}s")
