#!/bin/sh
# Source from the project root before running Python commands.
export HF_HOME="$PWD/.cache/huggingface"
export XDG_CACHE_HOME="$PWD/.cache"
export PIP_CACHE_DIR="$PWD/.cache/pip"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export PYTHONPATH="$PWD/src:$PWD/upstream/eecs183-hw4${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=30
