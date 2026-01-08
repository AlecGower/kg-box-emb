# Minimal Dockerfile for kg-box-emb
# Based on Python slim image (Debian Bookworm) to keep the image small compared
# to the official devcontainer image while still providing Python 3.10 and Java 11.

FROM python:3.10-slim-bookworm

# Noninteractive apt
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/app/.local/bin:${PATH}"

# Install small set of system dependencies + OpenJDK 11
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        wget \
        ca-certificates \
        openjdk-11-jdk-headless \
        libffi-dev \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME for tools that need it
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Upgrade pip and install wheel/setuptools (helps many Python packages)
RUN python -m pip install --upgrade pip setuptools wheel

# Create an unprivileged user for running the app
RUN useradd --create-home --shell /bin/bash app
WORKDIR /home/app

# Install python requirements (kept as separate layer for caching)
# If repository includes a requirements_torch.txt use it first (helps caching large torch install)
COPY requirements_torch.txt requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ -f requirements_torch.txt ]; then python -m pip install --no-cache-dir -r requirements_torch.txt; fi \
    && if [ -f requirements.txt ]; then python -m pip install --no-cache-dir -r requirements.txt; fi

# Copy the rest of the repository
COPY --chown=app:app . .

# Switch to non-root user
USER app

# Default command opens a shell; override with a command to run your app/test
CMD ["bash"]
