# llama-server (+ llama-quantize) mit CUDA, gebaut für genau eine GPU-Architektur.
# Build (macht scripts/setup_gpu.sh bzw. docker compose --profile gpu build):
#   docker build -f docker/llama.Dockerfile --build-arg CUDA_ARCH=90 --build-arg LLAMA_CPP_REF=b7xxx -t arena-llama:b7xxx .
# CUDA_ARCH: 86 = RTX 3090 / A6000, 90 = H100 / H200, 120 = RTX PRO 6000 Blackwell (CUDA >= 12.8), 87 = Jetson
ARG CUDA_VERSION=12.8.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS build
ARG CUDA_ARCH=90
ARG LLAMA_CPP_REF=master
RUN apt-get update -qq && apt-get install -y -qq git cmake build-essential libcurl4-openssl-dev && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/ggml-org/llama.cpp /llama.cpp && cd /llama.cpp && git checkout -q "$LLAMA_CPP_REF" \
 && cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_CURL=ON -DGGML_NATIVE=OFF \
 && cmake --build build -j"$(nproc)" --target llama-server llama-quantize \
 && git rev-parse HEAD > /llama.cpp/COMMIT

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04
RUN apt-get update -qq && apt-get install -y -qq libcurl4 libgomp1 curl && rm -rf /var/lib/apt/lists/*
COPY --from=build /llama.cpp/build/bin/ /usr/local/bin/
COPY --from=build /llama.cpp/COMMIT /COMMIT
ENV LD_LIBRARY_PATH=/usr/local/bin
HEALTHCHECK CMD curl -sf http://127.0.0.1:${PORT:-8081}/health || exit 1
ENTRYPOINT ["llama-server"]
