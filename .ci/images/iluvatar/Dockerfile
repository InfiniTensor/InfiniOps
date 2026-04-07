ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# CoreX runtime environment (base image sets these in `/etc/bash.bashrc`,
# but `docker build` `RUN` uses `/bin/sh` which doesn't source it).
ENV PATH=/usr/local/corex/bin:/usr/local/corex-4.3.0/corex-toolbox-1.0.0/bin:/usr/local/corex/lib64/python3/dist-packages/bin:/usr/local/openmpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages
ENV LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/lib:/usr/local/openmpi/lib

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
        git \
        ninja-build \
        coreutils \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf $(which python3) /usr/local/bin/python 2>/dev/null || true

ARG PIP_INDEX_URL
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    ${PIP_INDEX_URL:+--index-url "$PIP_INDEX_URL"} \
    scikit-build-core \
    pybind11 \
    libclang \
    pytest \
    pytest-cov \
    pytest-xdist \
    pyyaml \
    ruff==0.15.7

RUN pip config set global.index-url https://pypi.org/simple

# Pin pre-installed CoreX `torch` to prevent `pip` from replacing it with upstream version.
RUN pip show torch >/dev/null 2>&1 && \
    echo "torch==$(pip show torch | grep '^Version:' | awk '{print $2}')" > /etc/pip-constraints.txt || \
    touch /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt

WORKDIR /workspace
