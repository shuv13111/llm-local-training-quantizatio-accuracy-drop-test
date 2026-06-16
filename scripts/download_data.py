#!/usr/bin/env python3
"""
Step 2: Download and Prepare Training Data
============================================
Downloads Glaive and xLAM datasets, normalizes both to Qwen3's Hermes-style
tool-calling chat format, merges, shuffles, splits 90/10, and reports stats.

Why this step matters:
    Glaive uses raw string markup (USER:/ASSISTANT:/<functioncall> tags).
    xLAM uses structured JSON (query/tools/answers fields).
    The model needs ONE consistent format. We normalize both into Qwen3's
    Hermes-style chat template: tools in <tools> XML, calls in <tool_call> XML.
    Garbage in, garbage out — this step determines training quality.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --sample 1000   # quick test with 1k rows
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import Counter

import yaml
import numpy as np
from datasets import load_dataset, Dataset, concatenate_datasets


# ── Load project config ──────────────────────────────────────
def load_config():
    config_path = Path(__file__).parent.parent / "configs" / "project_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Qwen3 Hermes-style system prompt builder ────────────────
def build_tool_system_prompt(tools_list: list[dict]) -> str:
    """
    Build the system prompt that Qwen3's chat template expects for tool use.
    Tools go inside <tools></tools> XML, model responds with <tool_call> XML.
    """
    tools_json = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools_list)
    return (
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        f"<tools>\n{tools_json}\n</tools>\n\n"
        "For each function call, return a json object with function name and "
        "arguments within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call>"
    )


def format_tool_call(name: str, arguments: dict) -> str:
    """Format a single tool call in the <tool_call> XML style."""
    call = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    return f"<tool_call>\n{call}\n</tool_call>"


# ── Glaive Parser ────────────────────────────────────────────
def extract_json_objects(text: str) -> list[dict]:
    """
    Extract top-level JSON objects from a string using bracket counting.
    Handles arbitrarily nested braces — unlike regex, this works on
    deeply nested 'parameters' objects that Glaive's system prompts contain.
    """
    objects = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            depth = 0
            start = i
            in_string = False
            escape_next = False
            while i < len(text):
                ch = text[i]
                if escape_next:
                    escape_next = False
                elif ch == '\\' and in_string:
                    escape_next = True
                elif ch == '"' and not escape_next:
                    in_string = not in_string
                elif not in_string:
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            candidate = text[start:i + 1]
                            try:
                                objects.append(json.loads(candidate))
                            except json.JSONDecodeError:
                                pass
                            break
                i += 1
        i += 1
    return objects


def parse_glaive_system(system_str: str) -> list[dict] | None:
    """
    Parse Glaive's system field to extract function definitions.
    Glaive stores them as JSON objects after a preamble like
    "You are a helpful assistant with access to the following functions."
    Returns list of tool dicts in OpenAI-style format, or None if no tools.
    """
    if not system_str:
        return None

    # Extract all JSON objects from the system string
    json_objects = extract_json_objects(system_str)

    tools = []
    for obj in json_objects:
        # Check if this looks like a function definition (has "name" and either
        # "parameters" or "description" — Glaive uses several slight variations)
        if isinstance(obj, dict) and "name" in obj and (
            "parameters" in obj or "description" in obj
        ):
            tool = {
                "type": "function",
                "function": {
                    "name": obj.get("name", ""),
                    "description": obj.get("description", ""),
                    "parameters": obj.get("parameters", {}),
                },
            }
            tools.append(tool)

    return tools if tools else None


def parse_functioncall(text: str) -> dict | None:
    """
    Parse a <functioncall> JSON that might have single-quoted arguments.
    Glaive frequently uses: {"name": "func", "arguments": '{"key": "val"}'}
    which is Python dict syntax, not valid JSON.
    """
    # Strategy 1: Try standard JSON parsing
    objects = extract_json_objects(text)
    for obj in objects:
        if isinstance(obj, dict) and "name" in obj:
            args = obj.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            return {"name": obj["name"], "arguments": args}

    # Strategy 2: Single-quoted arguments — extract name via regex,
    # then find the arguments JSON object inside the single quotes
    # using bracket counting (which ignores the quote wrapper)
    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', text)
    if not name_match:
        return None
    name = name_match.group(1)

    # Look for arguments value starting with '{ (single-quoted JSON)
    args_sq = re.search(r'"arguments"\s*:\s*\'\s*(\{)', text)
    if args_sq:
        inner_objects = extract_json_objects(text[args_sq.start(1):])
        if inner_objects:
            return {"name": name, "arguments": inner_objects[0]}

    # Strategy 3: arguments as a raw JSON object (no quotes)
    args_raw = re.search(r'"arguments"\s*:\s*(\{)', text)
    if args_raw:
        inner_objects = extract_json_objects(text[args_raw.start(1):])
        if inner_objects:
            return {"name": name, "arguments": inner_objects[0]}

    # Got a name but couldn't parse arguments — return with empty args
    return {"name": name, "arguments": {}}


def parse_glaive_chat(chat_str: str) -> list[dict] | None:
    """
    Parse Glaive's chat field into a list of message dicts.
    Glaive format:
        USER: question <|endoftext|>
        ASSISTANT: text or <functioncall> {"name":..., "arguments":...} <|endoftext|>
        FUNCTION RESPONSE: {"result": ...}
        ASSISTANT: final answer <|endoftext|>
    """
    if not chat_str:
        return None

    messages = []
    # Split on role markers
    parts = re.split(r'(USER:|ASSISTANT:|FUNCTION RESPONSE:)', chat_str)

    current_role = None
    for part in parts:
        part = part.strip()
        if part == "USER:":
            current_role = "user"
        elif part == "ASSISTANT:":
            current_role = "assistant"
        elif part == "FUNCTION RESPONSE:":
            current_role = "tool"
        elif current_role and part:
            # Clean up content
            content = part.replace("<|endoftext|>", "").strip()
            if not content:
                continue

            if current_role == "assistant" and "<functioncall>" in content:
                # Split out any pre-function-call text
                pre_text = content.split("<functioncall>")[0].strip()
                if pre_text:
                    messages.append({"role": "assistant", "content": pre_text})

                # Extract the function call
                fc_match = re.search(r'<functioncall>\s*(.*)', content, re.DOTALL)
                if fc_match:
                    fc = parse_functioncall(fc_match.group(1))
                    if fc:
                        messages.append({
                            "role": "assistant",
                            "content": format_tool_call(fc["name"], fc["arguments"]),
                        })
                    else:
                        # Couldn't parse function call — skip this example
                        return None
                else:
                    return None
            else:
                messages.append({"role": current_role, "content": content})

    return messages if messages else None


def convert_glaive_example(example: dict) -> dict | None:
    """Convert a single Glaive example to normalized format."""
    tools = parse_glaive_system(example.get("system", ""))
    chat_messages = parse_glaive_chat(example.get("chat", ""))

    if chat_messages is None:
        return None

    messages = []

    # Add system message with tools if available
    if tools:
        messages.append({
            "role": "system",
            "content": build_tool_system_prompt(tools),
        })
    else:
        # Some Glaive examples are plain conversations without tools
        system_text = example.get("system", "").strip()
        if system_text:
            messages.append({"role": "system", "content": system_text})

    messages.extend(chat_messages)

    # Validate: must have at least user + assistant
    roles = [m["role"] for m in messages]
    if "user" not in roles or "assistant" not in roles:
        return None

    return {
        "messages": messages,
        "source": "glaive",
        "has_tool_call": tools is not None,
    }


# ── xLAM Parser ─────────────────────────────────────────────
def convert_xlam_example(example: dict) -> dict | None:
    """
    Convert a single xLAM example to normalized format.
    xLAM fields: query (str), tools (JSON str), answers (JSON str)
    """
    query = example.get("query", "").strip()
    tools_str = example.get("tools", "")
    answers_str = example.get("answers", "")

    if not query or not tools_str or not answers_str:
        return None

    # Parse tools
    try:
        tools_raw = json.loads(tools_str) if isinstance(tools_str, str) else tools_str
    except json.JSONDecodeError:
        return None

    # Normalize tool format — xLAM may use slightly different schema
    tools = []
    for t in tools_raw:
        if isinstance(t, dict):
            # xLAM might already be in OpenAI format or might need wrapping
            if "type" in t and t["type"] == "function":
                tools.append(t)
            elif "name" in t:
                tool = {
                    "type": "function",
                    "function": {
                        "name": t.get("name", ""),
                        "description": t.get("description", ""),
                        "parameters": t.get("parameters", {}),
                    },
                }
                tools.append(tool)

    if not tools:
        return None

    # Parse answers (function calls)
    try:
        answers = json.loads(answers_str) if isinstance(answers_str, str) else answers_str
    except json.JSONDecodeError:
        return None

    # Build the tool call response
    tool_calls = []
    for ans in answers:
        if isinstance(ans, dict) and "name" in ans:
            args = ans.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tool_calls.append(format_tool_call(ans["name"], args))

    if not tool_calls:
        return None

    # Assemble messages
    messages = [
        {"role": "system", "content": build_tool_system_prompt(tools)},
        {"role": "user", "content": query},
        {"role": "assistant", "content": "\n".join(tool_calls)},
    ]

    return {
        "messages": messages,
        "source": "xlam",
        "has_tool_call": True,
    }


# ── Main Pipeline ────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Download and prepare training data")
    parser.add_argument("--sample", type=int, default=None,
                        help="Use N samples from each dataset (for quick testing)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to project config YAML")
    args = parser.parse_args()

    config = load_config()
    project_root = Path(__file__).parent.parent

    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    seed = config["data_processing"]["seed"]
    split_ratio = config["data_processing"]["train_val_split"]

    # ── Step 1: Download ─────────────────────────────────────
    print("=" * 60)
    print("  Step 2a: Downloading Datasets")
    print("=" * 60)

    print(f"\n📥 Downloading Glaive: {config['datasets']['glaive']['hf_repo']}")
    glaive_ds = load_dataset(
        config["datasets"]["glaive"]["hf_repo"],
        split=config["datasets"]["glaive"]["split"],
    )
    print(f"   Loaded {len(glaive_ds):,} examples")
    print(f"   Columns: {glaive_ds.column_names}")

    print(f"\n📥 Downloading xLAM: {config['datasets']['xlam']['hf_repo']}")
    xlam_ds = load_dataset(
        config["datasets"]["xlam"]["hf_repo"],
        split=config["datasets"]["xlam"]["split"],
    )
    print(f"   Loaded {len(xlam_ds):,} examples")
    print(f"   Columns: {xlam_ds.column_names}")

    # ── Step 2: Inspect raw samples ──────────────────────────
    print("\n" + "=" * 60)
    print("  Step 2b: Raw Data Inspection")
    print("=" * 60)

    print("\n── Glaive sample (raw) ──")
    sample = glaive_ds[0]
    for key, val in sample.items():
        preview = str(val)[:300]
        print(f"   {key}: {preview}...")

    print("\n── xLAM sample (raw) ──")
    sample = xlam_ds[0]
    for key, val in sample.items():
        preview = str(val)[:300]
        print(f"   {key}: {preview}...")

    # ── Step 3: Convert to normalized format ─────────────────
    print("\n" + "=" * 60)
    print("  Step 2c: Normalizing to Qwen3 Hermes Format")
    print("=" * 60)

    # Optionally sample for quick testing
    if args.sample:
        glaive_ds = glaive_ds.select(range(min(args.sample, len(glaive_ds))))
        xlam_ds = xlam_ds.select(range(min(args.sample, len(xlam_ds))))
        print(f"\n⚡ Using --sample {args.sample}: {len(glaive_ds)} Glaive, {len(xlam_ds)} xLAM")

    # Convert Glaive
    print(f"\n🔄 Converting Glaive ({len(glaive_ds):,} examples)...")
    glaive_converted = []
    glaive_skip_reasons = Counter()
    for i, example in enumerate(glaive_ds):
        result = convert_glaive_example(example)
        if result:
            glaive_converted.append(result)
        else:
            # Diagnose why it failed
            system = example.get("system", "")
            chat = example.get("chat", "")
            tools = parse_glaive_system(system)
            msgs = parse_glaive_chat(chat)
            if not chat:
                glaive_skip_reasons["empty_chat"] += 1
            elif msgs is None:
                glaive_skip_reasons["chat_parse_failed"] += 1
            else:
                roles = [m["role"] for m in msgs]
                if "user" not in roles:
                    glaive_skip_reasons["no_user_message"] += 1
                elif "assistant" not in roles:
                    glaive_skip_reasons["no_assistant_message"] += 1
                else:
                    glaive_skip_reasons["unknown"] += 1
        if (i + 1) % 10000 == 0:
            print(f"   Processed {i+1:,}...")

    glaive_skipped = sum(glaive_skip_reasons.values())
    print(f"   ✅ Converted: {len(glaive_converted):,}")
    print(f"   ⚠️  Skipped: {glaive_skipped:,}")
    if glaive_skip_reasons:
        for reason, count in glaive_skip_reasons.most_common():
            print(f"      - {reason}: {count:,}")

    print(f"   ✅ Converted: {len(glaive_converted):,}")
    print(f"   ⚠️  Skipped (unparseable): {glaive_skipped:,}")

    # Convert xLAM
    print(f"\n🔄 Converting xLAM ({len(xlam_ds):,} examples)...")
    xlam_converted = []
    xlam_skipped = 0
    for i, example in enumerate(xlam_ds):
        result = convert_xlam_example(example)
        if result:
            xlam_converted.append(result)
        else:
            xlam_skipped += 1
        if (i + 1) % 10000 == 0:
            print(f"   Processed {i+1:,}...")

    print(f"   ✅ Converted: {len(xlam_converted):,}")
    print(f"   ⚠️  Skipped (unparseable): {xlam_skipped:,}")

    # ── Step 4: Show converted samples ───────────────────────
    print("\n" + "=" * 60)
    print("  Step 2d: Converted Sample Preview")
    print("=" * 60)

    print("\n── Glaive → Qwen3 format ──")
    if glaive_converted:
        sample = glaive_converted[0]
        for msg in sample["messages"]:
            role = msg["role"].upper()
            content = msg["content"][:200]
            print(f"   [{role}]: {content}...")
        print(f"   source={sample['source']}, has_tool_call={sample['has_tool_call']}")

    print("\n── xLAM → Qwen3 format ──")
    if xlam_converted:
        sample = xlam_converted[0]
        for msg in sample["messages"]:
            role = msg["role"].upper()
            content = msg["content"][:200]
            print(f"   [{role}]: {content}...")
        print(f"   source={sample['source']}, has_tool_call={sample['has_tool_call']}")

    # ── Step 5: Merge, shuffle, split ────────────────────────
    print("\n" + "=" * 60)
    print("  Step 2e: Merge, Shuffle, Split")
    print("=" * 60)

    all_examples = glaive_converted + xlam_converted
    print(f"\n📊 Total examples before split: {len(all_examples):,}")
    print(f"   From Glaive: {len(glaive_converted):,}")
    print(f"   From xLAM:   {len(xlam_converted):,}")

    # Count tool call vs plain conversation
    tool_count = sum(1 for e in all_examples if e["has_tool_call"])
    plain_count = len(all_examples) - tool_count
    print(f"   With tool calls: {tool_count:,}")
    print(f"   Plain conversations: {plain_count:,}")

    # Shuffle
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(all_examples))
    all_examples = [all_examples[i] for i in indices]

    # Split
    split_idx = int(len(all_examples) * split_ratio)
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]

    print(f"\n   Train: {len(train_examples):,}")
    print(f"   Val:   {len(val_examples):,}")

    # Verify source distribution in splits
    for split_name, split_data in [("Train", train_examples), ("Val", val_examples)]:
        source_counts = Counter(e["source"] for e in split_data)
        print(f"   {split_name} sources: {dict(source_counts)}")

    # ── Step 6: Compute token length stats ───────────────────
    print("\n" + "=" * 60)
    print("  Step 2f: Token Length Analysis")
    print("=" * 60)

    # Estimate token counts using character-based heuristic (4 chars ≈ 1 token)
    # More accurate tokenization happens during training; this is for sizing
    def estimate_tokens(messages):
        total_chars = sum(len(m["content"]) for m in messages)
        return total_chars // 4  # rough estimate

    train_lengths = [estimate_tokens(e["messages"]) for e in train_examples]
    train_lengths = np.array(train_lengths)

    print(f"\n   Estimated token lengths (train set):")
    print(f"   Mean:    {train_lengths.mean():.0f}")
    print(f"   Median:  {np.median(train_lengths):.0f}")
    print(f"   P90:     {np.percentile(train_lengths, 90):.0f}")
    print(f"   P95:     {np.percentile(train_lengths, 95):.0f}")
    print(f"   P99:     {np.percentile(train_lengths, 99):.0f}")
    print(f"   Max:     {train_lengths.max():.0f}")

    # Suggest max_seq_length
    p95 = np.percentile(train_lengths, 95)
    if p95 <= 1024:
        suggested = 1024
    elif p95 <= 2048:
        suggested = 2048
    elif p95 <= 4096:
        suggested = 4096
    else:
        suggested = 8192
    print(f"\n   💡 Suggested max_seq_length: {suggested}")
    print(f"      (covers P95={p95:.0f} estimated tokens)")

    # ── Step 7: Save processed data ──────────────────────────
    print("\n" + "=" * 60)
    print("  Step 2g: Saving Processed Data")
    print("=" * 60)

    # Save as JSONL — each line is {"messages": [...], "source": "...", ...}
    train_path = processed_dir / "train.jsonl"
    val_path = processed_dir / "val.jsonl"

    for path, data, name in [
        (train_path, train_examples, "train"),
        (val_path, val_examples, "val"),
    ]:
        with open(path, "w") as f:
            for example in data:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"   ✅ Saved {name}: {path} ({len(data):,} examples, {size_mb:.1f} MB)")

    # Save metadata
    metadata = {
        "total_examples": len(all_examples),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "glaive_converted": len(glaive_converted),
        "glaive_skipped": glaive_skipped,
        "glaive_skip_reasons": dict(glaive_skip_reasons),
        "xlam_converted": len(xlam_converted),
        "xlam_skipped": xlam_skipped,
        "with_tool_calls": tool_count,
        "plain_conversations": plain_count,
        "seed": seed,
        "split_ratio": split_ratio,
        "suggested_max_seq_length": suggested,
        "token_stats": {
            "mean": float(train_lengths.mean()),
            "median": float(np.median(train_lengths)),
            "p90": float(np.percentile(train_lengths, 90)),
            "p95": float(np.percentile(train_lengths, 95)),
            "p99": float(np.percentile(train_lengths, 99)),
            "max": int(train_lengths.max()),
        },
    }

    meta_path = processed_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✅ Saved metadata: {meta_path}")

    # ── Done ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✅ Step 2 Complete!")
    print("=" * 60)
    print(f"\n   Next: python scripts/baseline_eval.py")
    print(f"   (downloads base Qwen3 8B and runs BFCL pre-fine-tune baseline)\n")


if __name__ == "__main__":
    main()