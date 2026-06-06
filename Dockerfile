FROM python:3.12-slim

WORKDIR /app

# Install system dependencies required for OpenCV and image processing
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies, forcing the CPU version of PyTorch to save space
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir "transformers<4.48.0" "timm" "einops"

# Keep container running for dev by default, or run fastapi later
CMD ["tail", "-f", "/dev/null"]
