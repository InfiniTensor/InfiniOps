ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG http_proxy
ARG https_proxy
ARG no_proxy

ARG PIP_INDEX_URL=https://pypi.org/simple

RUN pip install --no-cache-dir --progress off \
    scikit-build-core \
    libclang \
    pytest \
    pytest-cov \
    pytest-xdist \
    ruff

WORKDIR /workspace
