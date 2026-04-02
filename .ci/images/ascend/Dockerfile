ARG BASE_IMAGE
FROM ${BASE_IMAGE}

USER root

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG http_proxy
ARG https_proxy
ARG no_proxy

ARG PIP_INDEX_URL
RUN pip install --no-cache-dir --progress-bar off \
    ${PIP_INDEX_URL:+--index-url "$PIP_INDEX_URL"} \
    ninja \
    scikit-build-core \
    pybind11 \
    pytest \
    pytest-cov \
    pytest-xdist \
    ruff==0.15.7

# Pin pre-installed torch to prevent pip from replacing it.
RUN pip show torch >/dev/null 2>&1 && \
    echo "torch==$(pip show torch | grep '^Version:' | awk '{print $2}')" > /etc/pip-constraints.txt || \
    touch /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt

WORKDIR /workspace
