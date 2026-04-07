ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# `MUSA_HOME`, `PATH`, `LD_LIBRARY_PATH` already set by base image.

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG http_proxy
ARG https_proxy
ARG no_proxy

ARG APT_MIRROR
RUN if [ -n "$APT_MIRROR" ]; then \
        sed -i "s|http://[^/]*/ubuntu|${APT_MIRROR}|g" /etc/apt/sources.list; \
    fi && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        ninja-build \
        libclang-dev \
    && rm -rf /var/lib/apt/lists/*

ARG PIP_INDEX_URL
RUN pip install --no-cache-dir \
    ${PIP_INDEX_URL:+--index-url "$PIP_INDEX_URL"} \
    scikit-build-core \
    libclang \
    pytest-cov \
    pytest-xdist \
    ruff==0.15.7

# Pin pre-installed `torch` to prevent `pip` from replacing it with upstream version.
RUN echo "torch==$(pip show torch | grep '^Version:' | awk '{print $2}')" > /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt

WORKDIR /workspace
