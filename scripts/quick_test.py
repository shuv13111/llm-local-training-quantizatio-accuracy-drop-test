#!/usr/bin/env python3
"""Quick manual test — eyeball the model's tool calls vs expected."""

import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-8B"

TOOLS = [
    {"type":"function","function":{"name":"get_weather","description":"Get current weather","parameters":{"type":"object","properties":{"location":{"type":"string"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location"]}}},
    {"type":"function","function":{"name":"search_products","description":"Search product catalog","parameters":{"type":"object","properties":{"query":{"type":"string"},"max_price":{"type":"number"},"category":{"type":"string","enum":["electronics","clothing","food","books"]}},"required":["query"]}}},
    {"type":"function","function":{"name":"send_email","description":"Send an email","parameters":{"type":"object","properties":{"to":{"type":"string"},"subject":{"type":"string"},"body":{"type":"string"}},"required":["to","subject","body"]}}},
    {"type":"function","function":{"name":"calculate_mortgage","description":"Calculate monthly mortgage payment","parameters":{"type":"object","properties":{"principal":{"type":"number"},"annual_rate":{"type":"number"},"term_years":{"type":"integer"}},"required":["principal","annual_rate","term_years"]}}},
]

TESTS = [
    ("How cold is it in Reykjavik in celsius?",
     "get_weather → location=Reykjavik, unit=celsius"),

    ("Find me cheap electronics under 50 bucks",
     "search_products → query=electronics, max_price=50, category=electronics"),

    ("Email alice@corp.com subject 'Q3 Report' body 'Attached the draft, please review by Friday'",
     "send_email → to=alice@corp.com, subject=Q3 Report, body=...review by Friday"),

    ("What's the monthly on a 500k loan at 7.25% for 15 years?",
     "calculate_mortgage → principal=500000, annual_rate=7.25, term_years=15"),

    ("Who won the 1998 World Cup?",
     "NO tool call — should answer directly or decline"),

    ("Check the weather in Berlin and also find me books about German history under $30",
     "MULTI: get_weather(Berlin) AND search_products(German history, max_price=30, category=books)"),
]

print("Loading model...")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

for query, expected in TESTS:
    msgs = [{"role": "user", "content": query}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False, tools=TOOLS)
    ids = tok([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=300, do_sample=False, temperature=None, top_p=None)

    resp = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

    print(f"\n{'='*60}")
    print(f"Q: {query}")
    print(f"EXPECTED: {expected}")
    print(f"GOT:      {resp[:300]}")
    print(f"{'='*60}")