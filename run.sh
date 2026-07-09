cd /tmp/zh/work/VisionSelector/qwen-vl-finetune
bash scripts/sft_3b.sh # for VisionSelector-Qwen2.5-VL-3B

cd /tmp/zh/work/VisionSelector/qwen-evaluation
bash run_selector.sh

cd /tmp/zh/work/VisionSelector/qwen-evaluation
bash run_token_compression.sh
