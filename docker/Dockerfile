FROM python:3.12-slim

WORKDIR /app

# Install system dependencies required for OpenCV and image processing
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY config/requirements.txt .

# Install dependencies, locking versions using the updated requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Keep container running for dev by default, or run fastapi later
CMD ["tail", "-f", "/dev/null"]
