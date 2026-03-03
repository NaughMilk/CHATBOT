#!/usr/bin/env bash
set -e

# Install all dependencies (including langchain-google-vertexai with its deps)
pip install -r requirements.txt
pip install langchain-google-vertexai

# Remove heavy packages not needed for API (torch ~700MB, CUDA libs)
pip uninstall -y torch triton nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-cufile-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 nvidia-nvjitlink-cu12 nvidia-nvshmem-cu12 nvidia-nvtx-cu12 cuda-bindings cuda-pathfinder 2>/dev/null || true
pip uninstall -y sentence-transformers transformers tokenizers 2>/dev/null || true
