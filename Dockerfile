# Minimal Dockerfile for kg-box-emb
# Based on Python slim image (Debian Bookworm) to keep the image small
# Installs OpenJDK 11 and project requirements

FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/app/.local/bin:${PATH}"

# Ensure apt cache directories exist and are writable to avoid APT post-invoke failures
RUN mkdir -p /var/cache/apt/archives /var/cache/apt/archives/partial \
    && chmod -R 755 /var/cache/apt/archives \
    # Temporarily neutralize any APT::Update::Post-Invoke hooks that may fail in minimal environments
    && printf 'APT::Update::Post-Invoke { "true"; };
APT::Update::Post-Invoke-Success { "true"; };
' > /etc/apt/apt.conf.d/99no-postinvoke

# Install small set of system dependencies + OpenJDK 11
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        wget \
        ca-certificates \
        openjdk-11-jdk-headless \
        libffi-dev \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*.deb /var/cache/apt/archives/partial/*.deb /var/cache/apt/*.bin || true \
    && rm -f /etc/apt/apt.conf.d/99no-postinvoke || true

# Set JAVA_HOME for tools that need it
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Upgrade pip and install wheel/setuptools (disable progress bar to avoid thread-start error in constrained build env)
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m pip install --upgrade pip setuptools wheel --progress-bar off --no-cache-dir

# Create an unprivileged user for running the app
RUN useradd --create-home --shell /bin/bash app
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
