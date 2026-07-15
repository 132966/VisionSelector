"""Extract scorer weights from a trained checkpoint (safetensors format)."""
import torch
from safetensors import safe_open
import glob
import sys

ckpt_dir = sys.argv[1]
output_path = sys.argv[2] if len(sys.argv) > 2 else ckpt_dir.rstrip('/') + '/scorer_weights.pt'

scorer_keys = [
    'visual.importance_scorer.q_proj.weight',
    'visual.importance_scorer.q_proj.bias',
    'visual.importance_scorer.k_proj.weight',
    'visual.importance_scorer.k_proj.bias',
    'visual.importance_scorer.input_layernorm.weight',
]

state_dict = {}
for shard in sorted(glob.glob(f'{ckpt_dir}/model-*.safetensors')):
    f = safe_open(shard, framework='pt')
    for k in scorer_keys:
        if k in f.keys():
            state_dict[k.replace('visual.importance_scorer.', '')] = f.get_tensor(k)

torch.save(state_dict, output_path)
print(f'Saved scorer weights to {output_path}')
for k, v in state_dict.items():
    print(f'  {k}: {v.shape}')
