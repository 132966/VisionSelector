cd /tmp/zh/work/VisionSelector/qwen-vl-finetune
bash scripts/sft_3b.sh # for VisionSelector-Qwen2.5-VL-3B
cd /tmp/zh/work/VisionSelector/qwen-evaluation
bash run_selector.sh
cd /tmp/zh/work/VisionSelector/qwen-evaluation
bash run_token_compression.sh

cd /tmp/zh/work/VisionSelector/qwen-vl-finetune-learnable-budget
bash scripts/sft_3b.sh
cd /tmp/zh/work/VisionSelector/qwen-vl-finetune-learnable-budget
bash scripts/sft_3b.sh --disable_scorer True
bash scripts/sft_3b.sh --scorer_ckpt ../output_ckpt/VisionSelector-Qwen2.5-VL-3B-train-Layer-Attn-10epoch/scorer_weights.pt --disable_scorer True
