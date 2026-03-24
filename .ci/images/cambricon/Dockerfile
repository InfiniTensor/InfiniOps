ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Python 3.10 executables (pip-installed tools) live under /usr/local/python3.10/bin.
ENV PATH=/usr/local/python3.10/bin:${PATH}

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG http_proxy
ARG https_proxy
ARG no_proxy

# git and cmake are pre-installed; coreutils-single covers coreutils needs.
RUN dnf install -y ninja-build && dnf clean all

ARG PIP_INDEX_URL
RUN pip install --no-cache-dir \
    ${PIP_INDEX_URL:+--index-url "$PIP_INDEX_URL"} \
    scikit-build-core \
    libclang \
    pytest \
    pytest-cov \
    pytest-xdist \
    ruff==0.15.7

# Pin pre-installed Cambricon torch to prevent pip from replacing it with upstream version.
RUN pip show torch >/dev/null 2>&1 && \
    echo "torch==$(pip show torch | grep '^Version:' | awk '{print $2}')" > /etc/pip-constraints.txt || \
    touch /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt

WORKDIR /workspace
