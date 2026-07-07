from datasets import load_dataset

missing = [
    ("lmms-lab/ScienceQA", "ScienceQA-IMG"),
    ("Efficient-Large-Model/ai2d-no-mask", None),
    ("lmms-lab/MMMU", None),
    ("lmms-lab/MME", None),
    ("lmms-lab/POPE", None),
]

for path, name in missing:
    print(f"prefetch: {path} ({name})", flush=True)
    if name:
        load_dataset(path, name)
    else:
        load_dataset(path)
    print(f"ok: {path}", flush=True)

print("missing datasets ready", flush=True)
