ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

ARG http_proxy
ARG https_proxy

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        cmake \
        ninja-build \
        libclang-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    scikit-build-core \
    pybind11 \
    libclang \
    pytest \
    pytest-cov \
    pytest-xdist \
    pyyaml

WORKDIR /workspace
