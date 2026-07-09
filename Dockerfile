FROM python:3.14-slim-trixie

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    cmake \
    git \
    curl \
    libhdf5-dev \
    hdf5-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ARG PROJECT_TAG=v2.9.0

RUN git clone --recurse-submodules --shallow-submodules --depth=1 --branch "$PROJECT_TAG" \
    https://github.com/roualdes/bridgestan.git

COPY . klhr/

RUN cd /app/bridgestan \
    && make -j4 test_models \
    && make -j4 /app/klhr/stan/normal_model.so \
    && make -j4 /app/klhr/stan/garch_model.so \
    && make -j4 /app/klhr/stan/arma_model.so \
    && make -j4 /app/klhr/stan/ar1_model.so \
    && make -j4 /app/klhr/stan/earnings_model.so \
    && make -j4 /app/klhr/stan/funnel_model.so \
    && make -j4 /app/klhr/stan/corr-normal_model.so \
    && make -j4 /app/klhr/stan/ill-normal_model.so \
    && make -j4 /app/klhr/stan/arK_model.so \
    && make -j4 /app/klhr/stan/glmm-poisson_model.so \
    && make -j4 /app/klhr/stan/rosenbrock_model.so

RUN cd /app/klhr \
    && cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build