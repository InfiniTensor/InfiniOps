ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

USER root

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG http_proxy
ARG https_proxy
ARG no_proxy

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
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

ARG PIP_INDEX_URL
RUN pip install --no-cache-dir \
    ${PIP_INDEX_URL:+--index-url "$PIP_INDEX_URL"} \
    cmake \
    scikit-build-core \
    pybind11 \
    libclang \
    pytest \
    pytest-cov \
    pytest-xdist \
    pyyaml \
    ruff==0.15.7

# Pin pre-installed torch to prevent pip from replacing it.
RUN pip show torch >/dev/null 2>&1 && \
    echo "torch==$(pip show torch | grep '^Version:' | awk '{print $2}')" > /etc/pip-constraints.txt || \
    touch /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt

WORKDIR /workspace
