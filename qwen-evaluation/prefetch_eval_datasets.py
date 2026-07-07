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
    if name:
        load_dataset(path, name, trust_remote_code=True)
    else:
        load_dataset(path, trust_remote_code=True)
    print(f"ok: {path}", flush=True)

print("all datasets ready", flush=True)
