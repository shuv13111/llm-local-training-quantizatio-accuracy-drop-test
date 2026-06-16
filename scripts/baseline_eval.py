#!/usr/bin/env python3
"""
Step 3: Download Base Model & Run Baseline Evaluation
=======================================================
Downloads Qwen3-8B, runs a quick function-calling sanity check to confirm
it works on the Spark, then runs a BFCL-style baseline evaluation on a
curated subset so you have pre-fine-tune numbers to compare against.

Why this step matters:
    You need to know how good the model ALREADY is at function calling
    before you fine-tune it. These "before" numbers are what make your
    "after" numbers meaningful. If the base model scores 85%, your fine-tune
    needs to beat that. If it scores 40%, there's a lot of room to improve.

Usage:
    python scripts/baseline_eval.py                  # full run
    python scripts/baseline_eval.py --skip-download   # if model already downloaded
"""

import argparse
import json
import sys
import time
from pathlib import Path
from collections import Counter

import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Load config ──────────────────────────────────────────────
def load_config():
    config_path = Path(__file__).parent.parent / "configs" / "project_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Test Cases ───────────────────────────────────────────────
# These are hand-crafted function-calling scenarios that test different
# capabilities. We use these for a quick sanity check AND as a mini
# benchmark to get pre-fine-tune accuracy numbers.

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and state/country"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search for products in a catalog",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_price": {"type": "number", "description": "Maximum price filter"},
                    "category": {"type": "string", "enum": ["electronics", "clothing", "food", "books"]},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email to a recipient",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email address"},
                    "subject": {"type": "string", "description": "Email subject line"},
                    "body": {"type": "string", "description": "Email body text"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_mortgage",
            "description": "Calculate monthly mortgage payment",
            "parameters": {
                "type": "object",
                "properties": {
                    "principal": {"type": "number", "description": "Loan amount in dollars"},
                    "annual_rate": {"type": "number", "description": "Annual interest rate as percentage"},
                    "term_years": {"type": "integer", "description": "Loan term in years"},
                },
                "required": ["principal", "annual_rate", "term_years"],
            },
        },
    },
]

TEST_CASES = [
    # ── Simple: single function, clear match ──
    {
        "id": "simple_01",
        "category": "simple",
        "query": "What's the weather in Tampa, Florida?",
        "expected_function": "get_weather",
        "expected_params": {"location": "Tampa, Florida"},
    },
    {
        "id": "simple_02",
        "category": "simple",
        "query": "Find me books under $20 about machine learning",
        "expected_function": "search_products",
        "expected_params": {"query": "machine learning", "max_price": 20, "category": "books"},
    },
    {
        "id": "simple_03",
        "category": "simple",
        "query": "Send an email to bob@example.com with subject 'Meeting Tomorrow' saying 'Can we reschedule to 3pm?'",
        "expected_function": "send_email",
        "expected_params": {"to": "bob@example.com", "subject": "Meeting Tomorrow", "body": "Can we reschedule to 3pm?"},
    },
    {
        "id": "simple_04",
        "category": "simple",
        "query": "Calculate the monthly payment for a $350,000 mortgage at 6.5% over 30 years",
        "expected_function": "calculate_mortgage",
        "expected_params": {"principal": 350000, "annual_rate": 6.5, "term_years": 30},
    },
    # ── Param extraction: tricky values, optional params ──
    {
        "id": "param_01",
        "category": "param_extraction",
        "query": "What's the temperature in Tokyo in fahrenheit?",
        "expected_function": "get_weather",
        "expected_params": {"location": "Tokyo", "unit": "fahrenheit"},
    },
    {
        "id": "param_02",
        "category": "param_extraction",
        "query": "Search for electronics",
        "expected_function": "search_products",
        "expected_params": {"query": "electronics"},
    },
    # ── Refusal: no function matches the request ──
    {
        "id": "refusal_01",
        "category": "refusal",
        "query": "What's the capital of France?",
        "expected_function": None,  # should NOT call any function
        "expected_params": None,
    },
    {
        "id": "refusal_02",
        "category": "refusal",
        "query": "Tell me a joke",
        "expected_function": None,
        "expected_params": None,
    },
    # ── Ambiguity: could match multiple functions ──
    {
        "id": "ambig_01",
        "category": "ambiguous",
        "query": "I need to know about the weather for my trip to London and also find waterproof jackets under $100",
        "expected_function": "multi",  # should ideally call both get_weather AND search_products
        "expected_params": None,
    },
]


# ── Response Parser ──────────────────────────────────────────
def parse_tool_calls(response_text: str) -> list[dict]:
    """
    Extract tool calls from model response.
    Qwen3 uses <tool_call>{"name": ..., "arguments": ...}</tool_call> format.
    """
    import re
    tool_calls = []

    # Find all <tool_call> blocks
    pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    matches = re.findall(pattern, response_text, re.DOTALL)

    for match in matches:
        try:
            call = json.loads(match)
            tool_calls.append(call)
        except json.JSONDecodeError:
            # Try to salvage — sometimes there's extra whitespace or formatting
            try:
                cleaned = match.strip()
                call = json.loads(cleaned)
                tool_calls.append(call)
            except json.JSONDecodeError:
                tool_calls.append({"_parse_error": True, "raw": match})

    return tool_calls


# ── Evaluation Logic ─────────────────────────────────────────
def evaluate_response(test_case: dict, tool_calls: list[dict], raw_response: str) -> dict:
    """Score a single model response against expected output."""
    result = {
        "id": test_case["id"],
        "category": test_case["category"],
        "query": test_case["query"],
        "raw_response": raw_response[:500],
        "tool_calls": tool_calls,
        "scores": {},
    }

    expected_fn = test_case["expected_function"]

    # ── Refusal case: model should NOT call any function ──
    if expected_fn is None:
        no_call = len(tool_calls) == 0
        result["scores"]["refusal_correct"] = no_call
        result["scores"]["overall"] = 1.0 if no_call else 0.0
        return result

    # ── Multi-call case ──
    if expected_fn == "multi":
        result["scores"]["made_calls"] = len(tool_calls) > 0
        result["scores"]["multiple_calls"] = len(tool_calls) > 1
        result["scores"]["overall"] = 1.0 if len(tool_calls) > 1 else 0.5 if len(tool_calls) == 1 else 0.0
        return result

    # ── Standard case: expect a specific function call ──
    if len(tool_calls) == 0:
        result["scores"]["format_valid"] = False
        result["scores"]["overall"] = 0.0
        return result

    call = tool_calls[0]

    # Format validity
    format_valid = "name" in call and "arguments" in call and "_parse_error" not in call
    result["scores"]["format_valid"] = format_valid

    if not format_valid:
        result["scores"]["overall"] = 0.0
        return result

    # Function name accuracy
    name_correct = call.get("name") == expected_fn
    result["scores"]["name_correct"] = name_correct

    if not name_correct:
        result["scores"]["overall"] = 0.25  # got format right but wrong function
        return result

    # Parameter accuracy
    expected_params = test_case.get("expected_params", {})
    actual_params = call.get("arguments", {})

    if expected_params:
        param_scores = []
        for key, expected_val in expected_params.items():
            if key in actual_params:
                actual_val = actual_params[key]
                # Flexible comparison — allow minor differences
                if isinstance(expected_val, str):
                    match = expected_val.lower() in str(actual_val).lower() or str(actual_val).lower() in expected_val.lower()
                elif isinstance(expected_val, (int, float)):
                    match = abs(float(actual_val) - float(expected_val)) < 0.01
                else:
                    match = actual_val == expected_val
                param_scores.append(1.0 if match else 0.5)
            else:
                param_scores.append(0.0)

        param_accuracy = sum(param_scores) / len(param_scores) if param_scores else 1.0
    else:
        param_accuracy = 1.0

    result["scores"]["param_accuracy"] = param_accuracy

    # Hallucinated params (extra params not in expected)
    if expected_params:
        extra = set(actual_params.keys()) - set(expected_params.keys())
        result["scores"]["hallucinated_params"] = len(extra)
    else:
        result["scores"]["hallucinated_params"] = 0

    # Overall score
    result["scores"]["overall"] = (1.0 + param_accuracy) / 2.0  # average of name + params

    return result


# ── Main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip model download (use if already cached)")
    args = parser.parse_args()

    config = load_config()
    project_root = Path(__file__).parent.parent
    model_name = config["models"]["dense"]["name"]
    model_dir = project_root / config["models"]["dense"]["base_dir"]
    results_dir = project_root / "eval" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 3a: Download Model ──────────────────────────────
    print("=" * 60)
    print("  Step 3a: Loading Base Model")
    print("=" * 60)

    print(f"\n📥 Model: {model_name}")
    print(f"   This will download ~16GB of weights on first run.")
    print(f"   Loading in BF16 (native precision)...\n")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    load_time = time.time() - t0
    print(f"   ✅ Model loaded in {load_time:.1f}s")

    # Memory stats
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"   GPU memory allocated: {allocated:.1f} GB")
        print(f"   GPU memory reserved:  {reserved:.1f} GB")

    # ── Step 3b: Sanity Check ────────────────────────────────
    print("\n" + "=" * 60)
    print("  Step 3b: Sanity Check — Single Function Call")
    print("=" * 60)

    # Build a simple test prompt using Qwen3's native tool calling
    # (the chat template handles tool formatting when you pass tools= param)

    messages = [
        {"role": "user", "content": "What's the weather in Tampa, Florida?"},
    ]

    # Let Qwen3's chat template handle tool formatting natively
    # (passing tools= param auto-generates the system prompt with tool definitions)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # disable thinking mode for tool calling
        tools=TOOLS,
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("\n🧪 Test prompt: 'What's the weather in Tampa, Florida?'")
    print("   Generating response...\n")

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,  # greedy for reproducibility
            temperature=None,
            top_p=None,
        )
    gen_time = time.time() - t0

    # Decode only the generated tokens (not the prompt)
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(f"   Response ({gen_time:.1f}s):")
    print(f"   {response[:300]}")

    tool_calls = parse_tool_calls(response)
    if tool_calls:
        print(f"\n   ✅ Parsed tool call(s):")
        for tc in tool_calls:
            print(f"      {json.dumps(tc, indent=2)}")
    else:
        print(f"\n   ⚠️  No tool calls parsed from response.")
        print(f"      The model may use a different format. Check raw output above.")

    # ── Step 3c: Run All Test Cases ──────────────────────────
    print("\n" + "=" * 60)
    print("  Step 3c: Baseline Evaluation (10 test cases)")
    print("=" * 60)

    all_results = []
    for i, test_case in enumerate(TEST_CASES):
        messages = [
            {"role": "user", "content": test_case["query"]},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            tools=TOOLS,
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        gen_time = time.time() - t0

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        tool_calls = parse_tool_calls(response)
        result = evaluate_response(test_case, tool_calls, response)
        result["gen_time_s"] = gen_time
        all_results.append(result)

        # Print per-case result
        overall = result["scores"]["overall"]
        icon = "✅" if overall >= 0.9 else "⚠️" if overall >= 0.5 else "❌"
        fn_called = tool_calls[0]["name"] if tool_calls and "name" in tool_calls[0] else "none"
        expected = test_case["expected_function"] or "none (refusal)"
        print(f"   {icon} [{result['category']}] {test_case['id']}: "
              f"expected={expected}, got={fn_called}, "
              f"score={overall:.2f}, time={gen_time:.1f}s")

    # ── Step 3d: Summary ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Step 3d: Baseline Results Summary")
    print("=" * 60)

    # Overall accuracy
    overall_scores = [r["scores"]["overall"] for r in all_results]
    avg_overall = sum(overall_scores) / len(overall_scores)
    print(f"\n   Overall accuracy: {avg_overall:.1%}")

    # Per-category breakdown
    categories = set(r["category"] for r in all_results)
    for cat in sorted(categories):
        cat_results = [r for r in all_results if r["category"] == cat]
        cat_avg = sum(r["scores"]["overall"] for r in cat_results) / len(cat_results)
        print(f"   {cat}: {cat_avg:.1%} ({len(cat_results)} cases)")

    # Timing
    total_gen_time = sum(r["gen_time_s"] for r in all_results)
    avg_gen_time = total_gen_time / len(all_results)
    print(f"\n   Avg generation time: {avg_gen_time:.1f}s per query")
    print(f"   Total eval time: {total_gen_time:.1f}s")

    # ── Save results ─────────────────────────────────────────
    output = {
        "model": model_name,
        "stage": "baseline_pre_finetune",
        "precision": "bf16",
        "overall_accuracy": avg_overall,
        "per_category": {
            cat: sum(r["scores"]["overall"] for r in all_results if r["category"] == cat) /
                 len([r for r in all_results if r["category"] == cat])
            for cat in categories
        },
        "results": all_results,
    }

    output_path = results_dir / "baseline_qwen3_8b_bf16.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n   ✅ Results saved: {output_path}")

    # ── Next steps ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✅ Step 3 Complete!")
    print("=" * 60)
    print(f"\n   Baseline accuracy: {avg_overall:.1%}")
    print(f"   Next: python scripts/train.py")
    print(f"   (fine-tune Qwen3 8B with LoRA on your prepared dataset)\n")


if __name__ == "__main__":
    main()