from datasets import load_dataset
import json, re

ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")

# Grab 5 examples from different parts of the dataset
for i in [0, 1000, 5000, 50000, 100000]:
    ex = ds[i]
    chat = ex.get("chat", "")
    # Check if our split would find role markers
    parts = re.split(r'(USER:|ASSISTANT:|FUNCTION RESPONSE:)', chat)
    has_markers = any(p.strip() in ("USER:", "ASSISTANT:", "FUNCTION RESPONSE:") for p in parts)
    print(f"--- Example {i} (markers={has_markers}) ---")
    print(f"CHAT (first 400 chars): {repr(chat[:400])}")
    print()