"""
Standalone script to extract LLM middle layer weights for scorer initialization.
Run once: python3 extract_scorer_weights.py --model_path pretrained/Qwen2.5-VL-3B-Instruct --output_path compression_method/scorer_init_3b.pt
"""
import argparse
import torch
from transformers import Qwen2_5_VLForConditionalGeneration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    print(f"Loading model from {args.model_path} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
    )

    middle_layer_idx = len(model.model.layers) // 2
    llm_layer = model.model.layers[middle_layer_idx]
    print(f"Extracting weights from LLM layer {middle_layer_idx} (total {len(model.model.layers)} layers)")

    state_dict = {
        "q_proj.weight": llm_layer.self_attn.q_proj.weight.clone(),
        "q_proj.bias": llm_layer.self_attn.q_proj.bias.clone(),
        "k_proj.weight": llm_layer.self_attn.k_proj.weight.clone(),
        "k_proj.bias": llm_layer.self_attn.k_proj.bias.clone(),
        "input_layernorm.weight": llm_layer.input_layernorm.weight.clone(),
    }

    torch.save(state_dict, args.output_path)
    print(f"Saved scorer init weights to {args.output_path}")
    for k, v in state_dict.items():
        print(f"  {k}: {v.shape}")


if __name__ == "__main__":
    main()
