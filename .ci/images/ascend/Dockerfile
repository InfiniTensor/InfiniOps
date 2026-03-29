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

# Bake Ascend toolkit environment into the image so every container
# has PATH / LD_LIBRARY_PATH / PYTHONPATH set without sourcing set_env.sh.
# Keep in sync with /usr/local/Ascend/ascend-toolkit/set_env.sh.
ENV ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest \
    ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest \
    ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest \
    ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp \
    TBE_IMPL_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe \
    TOOLCHAIN_HOME=/usr/local/Ascend/ascend-toolkit/latest/toolkit
ENV PATH=/usr/local/Ascend/ascend-toolkit/latest/bin:/usr/local/Ascend/ascend-toolkit/latest/compiler/ccec_compiler/bin:/usr/local/Ascend/ascend-toolkit/latest/tools/ccec_compiler/bin:/usr/local/Ascend/ascend-toolkit/latest/ccec_compiler/bin:/usr/local/python3.9.2/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/tools/aml/lib64:/usr/local/Ascend/ascend-toolkit/latest/tools/aml/lib64/plugin:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/nnengine:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/tools/hccn_tool/:/usr/lib/aarch64-linux-gnu/hdf5/serial:/usr/local/python3.9.2/lib:
ENV PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe:/usr/local/Ascend/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe:

WORKDIR /workspace
