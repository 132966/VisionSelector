#!/usr/bin/env bash
# Quick smoke: 1 sample, 1 GPU. Use _lite tasks where full dataset is too slow.
set -uo pipefail

ROOT_DIR="./result/eval_smoke"
SAMPLE_LIMIT=1
MODEL_PATH="../output_ckpt/VisionSelector-Qwen2.5-VL-3B"
MODEL_NAME="VisionSelector-Qwen2.5-VL-3B"
METHOD="selector"
BUDGET="0.2"
FILENAME="3b"

export OPENAI_API_URL=""
export OPENAI_API_KEY="dummy"
export http_proxy=http://10.229.18.27:8412
export https_proxy=http://10.229.18.27:8412
export HTTP_PROXY=http://10.229.18.27:8412
export HTTPS_PROXY=http://10.229.18.27:8412

HF_CACHE_ROOT="/tmp/zh/work/VisionSelector/.cache/huggingface"
DEFAULT_HF_CACHE="${HOME}/.cache/huggingface"
mkdir -p "$HF_CACHE_ROOT"/{hub,datasets,tmp}
[ -e "$DEFAULT_HF_CACHE" ] || ln -sf "$HF_CACHE_ROOT" "$DEFAULT_HF_CACHE"
export HF_HOME="$HF_CACHE_ROOT"
export HF_HUB_CACHE="$HF_CACHE_ROOT/hub"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_ROOT/hub"
export HF_DATASETS_CACHE="$HF_CACHE_ROOT/datasets"
export TMPDIR="$HF_CACHE_ROOT/tmp"

# task_name:lmms_eval_task
TASKS=(
    "docvqa_val:docvqa_val"
    "chartqa:chartqa"
    "textvqa_val:textvqa_val_lite"
    "ocrbench:ocrbench"
    "scienceqa_img:scienceqa_img"
    "ai2d_no_mask:ai2d_no_mask"
    "mmmu_val:mmmu_val"
    "mme:mme"
    "pope:pope"
)

has_results() {
    local output_path="$1"
    find "$output_path" -name '*results*.json' -print -quit | grep -q .
}

run_task() {
    local label="$1"
    local task="$2"
    local output_path="$ROOT_DIR/${label}/${BUDGET}/${MODEL_NAME}_${METHOD}_${BUDGET}_${FILENAME}"
    local log_file="${output_path}/quick.log"

    if has_results "$output_path"; then
        echo "[SKIP] $label"
        return 0
    fi

    rm -rf "$output_path"
    mkdir -p "$output_path"

    local limit="$SAMPLE_LIMIT"
    if [ "$label" = "mme" ]; then
        limit=2
    fi

    echo "[RUN] $label via $task (limit=${limit}) ..."
    if CUDA_VISIBLE_DEVICES=0 python3 -m lmms_eval \
        --model qwen2_5_vl_with_token_compression \
        --batch_size 1 \
        --limit "$limit" \
        --tasks "$task" \
        --output_path "$output_path" \
        --log_samples \
        --log_samples_suffix "$label" \
        --model_args "pretrained=${MODEL_PATH},method=${METHOD},budgets=${BUDGET},attn_implementation=flash_attention_2" \
        > "$log_file" 2>&1; then
        if has_results "$output_path"; then
            echo "[OK] $label"
            return 0
        fi
    fi

    echo "[FAIL] $label"
    tail -8 "$log_file" || true
    return 1
}

mkdir -p "$ROOT_DIR"
failed=()
for entry in "${TASKS[@]}"; do
    label="${entry%%:*}"
    task="${entry##*:}"
    if ! run_task "$label" "$task"; then
        failed+=("$label")
    fi
done

echo "===== smoke summary ====="
find "$ROOT_DIR" -name '*results*.json' | sort
if [ "${#failed[@]}" -gt 0 ]; then
    echo "failed: ${failed[*]}"
    exit 1
fi
echo "all tasks passed"
