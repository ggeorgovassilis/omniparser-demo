import torch
import time
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

device = "cpu"
print("Loading model...")
processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-base", 
    trust_remote_code=True,
    revision="5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac"
)
model = AutoModelForCausalLM.from_pretrained(
    "weights/icon_caption_florence", 
    trust_remote_code=True,
    torch_dtype=torch.float32,
    revision="f6c1a25888ffc1d945ee8a1a77ac833c7303d46e",
    code_revision="f6c1a25888ffc1d945ee8a1a77ac833c7303d46e"
).to(device)

images = [Image.new("RGB", (32, 32), color="red"), Image.new("RGB", (64, 64), color="blue")]
texts = ["<CAPTION>", "<CAPTION>"]

print("Testing batch processing...")
try:
    t0 = time.time()
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)
    print("Inputs shape:", inputs["pixel_values"].shape)
    
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=20,
        num_beams=1,
        early_stopping=False
    )
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)
    print("Batch texts:", generated_texts)
    
    for i in range(len(generated_texts)):
        st = processor.post_process_generation(generated_texts[i], task="<CAPTION>", image_size=(images[i].width, images[i].height))
        print("Parsed:", st)
        
    print(f"Time: {time.time() - t0:.2f}s")
except Exception as e:
    print("Failed!", e)
