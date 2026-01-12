# Minimal Dockerfile for kg-box-emb
# Based on Python slim image (Debian Bookworm) to keep the image small
# Installs OpenJDK 11 and project requirements

FROM python:3.10-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/app/.local/bin:${PATH}"

# # Ensure apt cache directories exist and are writable to avoid APT post-invoke failures
# RUN mkdir -p /var/cache/apt/archives /var/cache/apt/archives/partial \
#     && chmod -R 755 /var/cache/apt/archives
# 
# # Install small set of system dependencies + OpenJDK 11
# # Use -o flags to override any APT::Update::Post-Invoke hooks for this invocation
# RUN apt-get update -y \
#     && apt-get install -y --no-install-recommends \
#         build-essential \
#         git \
#         wget \
#         ca-certificates \
#         openjdk-11-jdk-headless \
#         libffi-dev \
#         libssl-dev \
#     && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*.deb /var/cache/apt/archives/partial/*.deb /var/cache/apt/*.bin || true

# # Set JAVA_HOME for tools that need it
# ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Upgrade pip and install wheel/setuptools (disable progress bar to avoid thread-start error in constrained build env)
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m pip install --upgrade pip setuptools wheel --progress-bar off --no-cache-dir

# Create an unprivileged user for running the app with sudo privileges
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends sudo \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash --groups sudo app \
    && echo "app ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
WORKDIR /home/app

# Copy and install python requirements (order helps cache large torch wheel)
COPY requirements_torch.txt requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ -f requirements_torch.txt ]; then python -m pip install --no-cache-dir -r requirements_torch.txt; fi \
    && if [ -f requirements.txt ]; then python -m pip install --no-cache-dir -r requirements.txt; fi

# Copy repository files
COPY --chown=app:app . .

USER app
CMD ["bash"]
