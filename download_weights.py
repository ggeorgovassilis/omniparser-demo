import os
from huggingface_hub import hf_hub_download

REPO_ID = "microsoft/OmniParser-v2.0"
LOCAL_DIR = "weights"

files_to_download = [
    "icon_detect/train_args.yaml",
    "icon_detect/model.pt",
    "icon_detect/model.yaml",
    "icon_caption/config.json",
    "icon_caption/generation_config.json",
    "icon_caption/model.safetensors",
    "icon_caption/preprocessor_config.json"
]

def download_weights():
    print(f"Downloading OmniParser weights from {REPO_ID}...")
    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    for file_path in files_to_download:
        print(f"Downloading {file_path}...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=file_path,
            local_dir=LOCAL_DIR
        )
        
    # OmniParser expects the caption folder to be named icon_caption_florence
    src = os.path.join(LOCAL_DIR, "icon_caption")
    dst = os.path.join(LOCAL_DIR, "icon_caption_florence")
    
    if os.path.exists(src) and not os.path.exists(dst):
        os.rename(src, dst)
        print(f"Renamed {src} to {dst}")
        
    print("Download complete!")

if __name__ == "__main__":
    download_weights()
