# Base image with CUDA support
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.5.1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Install specific versions of quantization libraries
RUN pip install "poetry==$POETRY_VERSION" \
    && pip install auto-gptq==0.4.2 \
    && pip install optimum==1.12.0 \
    && pip install bitsandbytes==0.41.1

# Copy only requirements to cache them in docker layer
WORKDIR /app
COPY poetry.lock pyproject.toml /app/

# Install project dependencies with quantization extras
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --only main --extras "quantization"

# Copy the rest of the project
COPY . /app

# Install specific versions of GPU-accelerated libraries
RUN pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118 \
    && pip install transformers==4.35.0 \
    && pip install accelerate==0.24.1

# Set up the entry point with quantization flags
EXPOSE 8000
CMD ["python", "-m", "interface.api", "--quantize", "4bit", "--device", "cuda"]
