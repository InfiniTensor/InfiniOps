ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG http_proxy
ARG https_proxy
ARG no_proxy

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        cmake \
        ninja-build \
        coreutils \
        curl \
        libclang-dev \
    && rm -rf /var/lib/apt/lists/*

ARG PRIVATE_SDK_URL
RUN if [ -n "$PRIVATE_SDK_URL" ]; then \
        curl -fSL "$PRIVATE_SDK_URL" -o /tmp/sdk.run && \
        chmod +x /tmp/sdk.run && /tmp/sdk.run --quiet && \
        rm /tmp/sdk.run; \
    fi

RUN pip install --no-cache-dir \
    scikit-build-core \
    pybind11 \
    libclang \
    pytest \
    pytest-cov \
    pytest-xdist \
    pyyaml

WORKDIR /workspace
