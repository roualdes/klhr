# syntax=docker/dockerfile:1

# builder
FROM debian:trixie-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive

ARG BRIDGESTAN_TAG=v2.9.0
ARG BUILD_JOBS=4
ARG KLHR_TARGET=klhr-experiment

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone \
        --recurse-submodules \
        --shallow-submodules \
        --depth=1 \
        --branch="${BRIDGESTAN_TAG}" \
        https://github.com/roualdes/bridgestan.git \
        /app/bridgestan

COPY stan/*.stan /app/klhr/stan/

RUN make \
        -C /app/bridgestan \
        -j"${BUILD_JOBS}" \
        /app/klhr/stan/normal_model.so \
        /app/klhr/stan/garch_model.so \
        /app/klhr/stan/arma_model.so \
        /app/klhr/stan/ar1_model.so \
        /app/klhr/stan/earnings_model.so \
        /app/klhr/stan/funnel_model.so \
        /app/klhr/stan/corr-normal_model.so \
        /app/klhr/stan/ill-normal_model.so \
        /app/klhr/stan/arK_model.so \
        /app/klhr/stan/glmm-poisson_model.so \
        /app/klhr/stan/rosenbrock_model.so \
        /app/klhr/stan/ssp3nc3r_2_model.so \
        /app/klhr/stan/ssp3nc3r_model.so

COPY . /app/klhr

# Collect BridgeStan's matching TBB runtime library.
RUN set -eux; \
    mkdir -p /opt/klhr-runtime/lib; \
    tbb_library="$(find \
        /app/bridgestan/stan/lib/stan_math/lib \
        \( -type f -o -type l \) \
        -name 'libtbb.so.2' \
        -print \
        -quit)"; \
    test -n "${tbb_library}"; \
    echo "Using BridgeStan TBB library: ${tbb_library}"; \
    cp -L "${tbb_library}" /opt/klhr-runtime/lib/libtbb.so.2

RUN cmake \
        -S /app/klhr \
        -B /app/klhr/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/klhr \
        -DKLHR_BUILD_EXAMPLES=ON

RUN cmake \
        --build /app/klhr/build \
        --target "${KLHR_TARGET}" \
        --parallel "${BUILD_JOBS}"

RUN cmake \
        --install /app/klhr/build

RUN test -x "/opt/klhr/bin/klhr-experiment"


# runtime
FROM debian:trixie-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive

ARG KLHR_TARGET=klhr-experiment

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libhdf5-310 \
        libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/klhr

COPY --from=builder \
    /opt/klhr/bin/${KLHR_TARGET} \
    /usr/local/bin/klhr-experiment

COPY --from=builder \
    /app/klhr/stan/ \
    /app/klhr/stan/

COPY --from=builder \
    /opt/klhr-runtime/lib/libtbb.so.2 \
    /usr/local/lib/libtbb.so.2

RUN ldconfig

RUN set -eux; \
    for file in /usr/local/bin/klhr-experiment /app/klhr/stan/*_model.so; do \
        echo "Checking ${file}"; \
        ldd "${file}"; \
        if ldd "${file}" | grep -q "not found"; then \
            echo "Missing shared-library dependency for ${file}" >&2; \
            exit 1; \
        fi; \
    done

RUN chmod -R a+rX /app/klhr/stan \
    && mkdir -p /app/klhr/output \
    && chown 10001:10001 /app/klhr/output

# Safe default for standalone container execution.
USER 10001:10001

ENTRYPOINT ["/usr/local/bin/klhr-experiment"]
