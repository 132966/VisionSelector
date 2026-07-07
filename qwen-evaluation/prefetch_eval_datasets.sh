#!/usr/bin/env bash
set -euo pipefail

export http_proxy=http://10.229.18.27:8412
export https_proxy=http://10.229.18.27:8412
export HTTP_PROXY=http://10.229.18.27:8412
export HTTPS_PROXY=http://10.229.18.27:8412

HF_CACHE_ROOT="/tmp/zh/work/VisionSelector/.cache/huggingface"
export HF_HOME="$HF_CACHE_ROOT"
export HF_HUB_CACHE="$HF_CACHE_ROOT/hub"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_ROOT/hub"
export HF_DATASETS_CACHE="$HF_CACHE_ROOT/datasets"
export TRANSFORMERS_CACHE="$HF_CACHE_ROOT/hub"
export TMPDIR="$HF_CACHE_ROOT/tmp"
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TMPDIR"

# wait for cache migration if still running
while [ ! -f /tmp/zh/work/VisionSelector/.cache/hf_migrate.done ]; do
    sleep 5
done

python3 <<'PY'
from datasets import load_dataset

prefetch = [
    ("lmms-lab/DocVQA", "DocVQA"),
    ("lmms-lab/ChartQA", None),
    ("lmms-lab/textvqa", None),
    ("echo840/OCRBench", None),
    ("lmms-lab/ScienceQA", "ScienceQA-IMG"),
    ("Efficient-Large-Model/ai2d-no-mask", None),
    ("lmms-lab/MMMU", None),
    ("lmms-lab/MME", None),
    ("lmms-lab/POPE", None),
]

for path, name in prefetch:
    print(f"prefetch: {path} ({name})", flush=True)
    kwargs = {"trust_remote_code": True}
    if name:
        load_dataset(path, name, **kwargs)
    else:
        load_dataset(path, **kwargs)
    print(f"ok: {path}", flush=True)
PY
