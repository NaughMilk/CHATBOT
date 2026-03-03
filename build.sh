#!/usr/bin/env bash
set -e

# Install regular dependencies
pip install -r requirements.txt

# Install langchain-google-vertexai WITHOUT its heavy deps (torch, sentence-transformers)
pip install langchain-google-vertexai --no-deps
