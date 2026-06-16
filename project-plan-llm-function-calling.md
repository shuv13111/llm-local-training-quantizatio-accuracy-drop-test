# LLM Function Calling: Fine-Tune, Quantize, Benchmark

## The Big Idea

Fine-tune open-source LLMs for structured function calling / tool use, then systematically quantize across multiple precision levels and measure exactly where and how tool-use accuracy degrades. Run the study on two architectures — a dense model (Qwen3 8B) and a Mixture-of-Experts model (Gemma 4 26B A4B) — to compare how each responds to quantization pressure. The published result fills a real gap — most quantization benchmarks focus on perplexity and chat quality, but almost nobody has published detailed breakdowns of how quantization affects structured output reliability, and nobody's compared dense vs MoE resilience for tool-use tasks.

---

## Hardware

**NVIDIA DGX Spark**

- Grace Blackwell Superchip (GB10)
- 128 GB unified CPU+GPU memory
- 1 PFLOP AI performance (FP4 sparse)
- Native NVFP4 support via fifth-gen Tensor Cores
- DGX OS (Ubuntu-based) with full NVIDIA AI stack
- CUDA 13, TensorRT-LLM, NeMo preinstalled
- Can fine-tune models up to 70B parameters locally

---

## Base Model Selection

Run two models — one dense, one MoE — to compare how quantization affects each architecture differently. This is the angle that makes the study stand out.

### Primary: Dense Model

| Model | Params | Architecture | License | Why |
|---|---|---|---|---|
| **Qwen3 8B** (recommended) | 8B | Dense | Apache 2.0 | Already has strong tool-calling ability out of the box. Fine-tuning an already-capable model lets you measure improvement *and* quantization degradation from a higher baseline — more interesting story than starting from zero. Huge community, well-documented fine-tuning recipes. Fits in 128GB at full BF16 with room to spare. |
| Gemma 4 8B | 8B | Dense | Apache 2.0 | Google put explicit tool-call training into this family. Good alternative if you want a Google-lineage comparison point. |
| Qwen3 32B | 32B | Dense | Apache 2.0 | The well-rounded tool-calling model in the 30B range. Fits comfortably on 128GB. Use this if you want a bigger dense model without jumping to 70B. |
| Gemma 4 31B | 31B | Dense | Apache 2.0 | Strong on math and coding benchmarks, inherits Gemma's tool-call training. Easier to fine-tune than a 70B since it's still dense. |

### Secondary: MoE Model

| Model | Total Params | Active Params | License | Why |
|---|---|---|---|---|
| **Gemma 4 26B A4B** (recommended) | 26B | 3.8B | Apache 2.0 | Only activates 3.8B parameters per forward pass yet punches way above its weight. The quantization story becomes *much* more interesting because MoE expert routing is sensitive to precision loss. If you can show that NVFP4 handles expert routing better (or worse) than GGUF Q4, that's a genuinely novel finding nobody's published. |
| Qwen3.5 27B | 27B | ~3B (MoE) | Apache 2.0 | Successor to Qwen3, just dropped. Edges out Gemma 4 31B on MMLU Pro and GPQA Diamond. Freshest option if it's stable enough when you start. |

### Why Two Models Matters for the Publish

The dense model (Qwen3 8B) gives you the clean, controlled baseline study: "here's how function-calling accuracy degrades across quantization levels." The MoE model (Gemma 4 26B A4B) gives you the novel angle: "does quantization break expert routing before it breaks output quality?" Dense vs MoE under quantization pressure is a comparison the community actually needs and nobody's done thoroughly yet.

Both models are Apache 2.0, so you can release fine-tuned weights and quantized variants without licensing headaches.

**Decision:** Start with Qwen3 8B (dense) to validate the full pipeline end-to-end. Then run Gemma 4 26B A4B (MoE) through the same pipeline and compare results.

---

## Datasets

### Primary: Glaive Function Calling v2

- **Source:** `glaive-ai/glaive-function-calling-v2` on HuggingFace
- **Size:** ~113K examples
- **Format:** System prompt with available functions → user query → assistant response with function call → function result → final assistant response
- **Why:** Large, clean, covers diverse function schemas. Good for teaching the model the mechanics of structured tool calling.

### Secondary: Salesforce xLAM

- **Source:** `Salesforce/xlam-function-calling-60k` on HuggingFace
- **Size:** ~60K examples
- **Format:** Function definitions + queries → structured JSON function calls
- **Why:** Higher quality curation, more complex multi-turn and parallel function calling scenarios. Good for pushing accuracy on harder cases.
- **Licensing caveat:** xLAM is a **gated** dataset (you must accept terms on HuggingFace) and is released **for research purposes only**. This matters for a publish project — if you release a model trained on it, check that your model's license/usage statement is consistent with xLAM's research-only terms. Glaive (Apache-2.0) is safer if you want a cleanly redistributable model.

### Evaluation: Berkeley Function Calling Leaderboard (BFCL)

- **Source:** The `bfcl` evaluation harness lives in the `ShishirPatil/gorilla` GitHub repo (install via pip as an editable package); the raw data is mirrored at `gorilla-llm/Berkeley-Function-Calling-Leaderboard` on HuggingFace.
- **Important correction:** BFCL is **not** a simple "load as a test set" dataset. It is a full evaluation framework with its own CLI. The current version is **V4 (agentic evaluation)**; earlier versions are V1 (AST-based metric), V2 (live/enterprise functions), V3 (multi-turn). You run it in two phases: (1) generate model responses across test categories, then (2) run the AST-based evaluator, which produces per-category accuracy CSVs.
- **Why:** Industry-standard benchmark with a published leaderboard, so your numbers are directly comparable to other models. The AST evaluation method means you're scoring whether the *function call structure* is correct, not just string-matching — which is exactly what makes it sensitive to quantization damage.
- **Categories to expect:** simple, multiple, parallel, parallel-multiple, plus live and multi-turn/agentic splits. Report per-category so you can show which call types degrade first.

### Data Prep Checklist

- [ ] Download Glaive + xLAM for training; set up the BFCL harness separately (it's code, not just data)
- [ ] Inspect schema formats — normalize Glaive and xLAM to a single chat-template format compatible with the base model's expected input structure
- [ ] Split Glaive + xLAM into train/validation (90/10)
- [ ] Use BFCL purely for final evaluation — never train on it
- [ ] Check for overlap/leakage between the training sets and BFCL's functions
- [ ] Tokenize and compute sequence length distributions to set max_seq_length appropriately

---

## Fine-Tuning Plan

### Method: LoRA (8B) / QLoRA (70B)

Full fine-tuning of 8B is feasible on 128GB but LoRA is more practical, reproducible, and what most readers will actually use. Keeps the study grounded.

### Key Hyperparameters to Document

| Parameter | Starting Value | Notes |
|---|---|---|
| LoRA rank (r) | 64 | Higher than typical chat fine-tunes because structured output needs precise token-level control |
| LoRA alpha | 128 | Standard 2x rank |
| LoRA target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | These names work for Qwen3. For Gemma 4 MoE, verify the expert layer naming — MoE models have router/gate weights inside expert blocks that you may or may not want to target. Start by targeting the same attention + FFN projections and only add expert-specific modules if baseline results are weak. |
| Learning rate | 2e-4 | With cosine schedule + warmup |
| Warmup steps | 100 | |
| Batch size | 4 (effective 32 with gradient accumulation) | Tune based on memory |
| Epochs | 3 | Monitor eval loss for early stopping |
| Max sequence length | 2048-4096 | Based on dataset token distribution |
| Optimizer | AdamW (8-bit via bitsandbytes) | Fallback: standard AdamW if bitsandbytes doesn't work on aarch64 (see open questions) |
| Weight decay | 0.01 | |

**MoE-specific note:** When fine-tuning Gemma 4 26B A4B, LoRA adapters only touch a subset of experts per training example (since only ~3.8B activate per forward pass). This means training may need more steps/epochs to reach all experts sufficiently. Monitor per-expert activation coverage if your tooling supports it.

### Tooling

- **Framework:** HuggingFace Transformers + PEFT + TRL (SFTTrainer)
- **Alternative:** NVIDIA NeMo (native on DGX Spark, may offer better Blackwell optimization)
- **Logging:** Weights & Biases or TensorBoard

### Training Checkpoints

- [ ] Verify base model loads and runs inference correctly on DGX Spark
- [ ] Run base model against BFCL to establish pre-fine-tune baseline
- [ ] **Mix Glaive + xLAM into a single shuffled training set** (preferred over training sequentially — sequential fine-tuning risks catastrophic forgetting of the first dataset's patterns). If you do want a curriculum (simpler → harder), treat it as an experiment to measure, not the default.
- [ ] Evaluate on validation set — check function call format accuracy, not just loss
- [ ] Final evaluation on BFCL hold-out
- [ ] Save the merged LoRA + base model as the "full-precision fine-tuned" checkpoint

---

## Quantization Plan

This is the core contribution. Quantize the fine-tuned model across every practical precision level and benchmark each one.

### Quantization Levels to Test

| Level | Format | Tool | Notes |
|---|---|---|---|
| BF16 | BrainFloat 16 | Baseline | Full fine-tuned model, no quantization |
| FP8 | Float 8 (E4M3) | TensorRT Model Optimizer + TRT-LLM | Native Blackwell support |
| **NVFP4** | **NVIDIA Float 4** | **TensorRT Model Optimizer (inside TRT-LLM container)** | **Unique angle — hardware-native FP4 on Blackwell Tensor Cores. Most people can't test this. Documented to cut memory ~3.5x vs FP16 and keep accuracy within ~1% of FP8.** |
| MXFP4 (optional) | Microscaling FP4 | TRT-LLM | Other 4-bit float format TRT-LLM supports; nice extra comparison point against NVFP4 |
| GPTQ Q4 | 4-bit grouped | GPTQModel | Community standard. Note: AutoGPTQ is effectively unmaintained — use its successor **GPTQModel** |
| AWQ Q4 | 4-bit activation-aware | AutoAWQ | Often better than GPTQ for structured tasks (verify the package still installs cleanly on aarch64) |
| GGUF Q4_K_M | 4-bit k-quant mixed | llama.cpp | Most popular consumer format |
| GGUF Q3_K_M | 3-bit k-quant mixed | llama.cpp | Where things start to get interesting |
| GGUF Q2_K | 2-bit k-quant | llama.cpp | Expect significant degradation — document exactly what breaks |

**The exact NVFP4 path on Spark:** run NVIDIA's TensorRT Model Optimizer (`modelopt`) inside the `nvcr.io/nvidia/tensorrt-llm` container, using the `huggingface_example.sh` script with `--quant nvfp4 --export_fmt hf`. NVIDIA publishes a ready-made playbook for exactly this at `github.com/NVIDIA/dgx-spark-playbooks` (the `nvfp4-quantization` folder) — start there rather than from scratch.

**Methodology caveat that matters for your write-up:** GGUF runs through llama.cpp while NVFP4/FP8 run through TensorRT-LLM. So a throughput comparison between, say, GGUF Q4 and NVFP4 is really measuring *format + runtime together*, not the format in isolation. Be explicit about this in the publish — for a clean format-vs-format accuracy comparison, hold the runtime fixed where you can; treat cross-runtime throughput numbers as "real-world deployment" comparisons rather than controlled ones.

### Quantization Checklist

- [ ] Install GPTQModel, AutoAWQ, llama.cpp, and the TRT-LLM container (modelopt comes inside it)
- [ ] Quantize model to each level
- [ ] Verify each quantized model loads and generates coherent output
- [ ] Record model file sizes at each level
- [ ] Record VRAM usage at inference time for each level
- [ ] Note any quantization failures or format-specific issues

---

## Benchmarking Plan

### Metrics to Capture Per Quantization Level

**Accuracy Metrics (the interesting stuff):**

- Function call format validity rate — does the output parse as valid JSON/function call?
- Function name accuracy — did it pick the right function?
- Parameter extraction accuracy — did it fill in the right arguments?
- Parameter type accuracy — did string/int/float types survive quantization?
- Multi-function call accuracy — for queries requiring parallel or sequential calls
- Hallucinated parameter rate — did it invent arguments not in the schema?
- Refusal accuracy — when no function matches, does it correctly decline?

**Performance Metrics:**

- Tokens per second (inference throughput)
- Time to first token (latency)
- Peak memory usage during inference
- Model file size on disk

**Qualitative Analysis:**

- Categorize failure modes at each quantization level
- Identify which function-calling patterns degrade first
- Document specific examples of "interesting" failures (e.g., Q3 starts confusing similar parameter names, Q2 hallucinates extra parameters)

**Dense vs MoE Comparison (the novel angle):**

- At each quantization level, compare Qwen3 8B (dense) vs Gemma 4 26B A4B (MoE) on the same BFCL categories
- Track whether the MoE model shows a sharper accuracy cliff at some precision level (hypothesis: expert routing precision loss may cause a sudden drop rather than gradual degradation)
- Compare the "active parameter efficiency" — the MoE only uses 3.8B params per inference, so at Q4 it's effectively running ~1.9GB of active weights vs the dense model's ~4GB. Does the MoE's smaller active footprint make it *more* or *less* tolerant of quantization?
- If possible, log which experts get activated per call and whether quantization changes routing decisions

### Benchmark Execution

- [ ] Write a standardized inference harness that runs BFCL eval across all quantized models
- [ ] Run each model 3 times minimum for variance estimation
- [ ] Log raw predictions alongside ground truth for error analysis
- [ ] Generate comparison tables and charts

---

## Publishing Plan

### Deliverables

1. **Blog post / write-up** — narrative walkthrough of the full process, aimed at practitioners
2. **HuggingFace model card** — publish the fine-tuned model (and optionally quantized variants)
3. **GitHub repo** — all training scripts, quantization scripts, eval harness, raw results
4. **Results dashboard** — interactive charts showing accuracy vs. quantization level tradeoffs

### Key Findings to Highlight

- The "sweet spot" quantization level where you get max compression with minimal accuracy loss
- NVFP4 vs community Q4 formats — does hardware-native quantization actually win?
- **Dense vs MoE under quantization pressure** — does expert routing in Gemma 4 26B A4B break at a different precision threshold than dense Qwen3 8B? Which architecture is more quantization-resilient for structured output?
- Which function-calling subtasks are most sensitive to quantization (probably: numerical parameter extraction, complex nested schemas, parallel/multi-function calls)
- Practical recommendations: "if you're deploying a tool-use model locally, here's what precision to use"

### Where to Publish

- HuggingFace (model + dataset + model card)
- GitHub (code + reproducibility)
- Blog (Medium, Substack, or personal site)
- Reddit (r/LocalLLaMA — this audience will love it)
- Twitter/X thread with key charts

---

## Rough Order of Operations

1. Environment setup — verify DGX Spark stack, install dependencies, confirm bitsandbytes/aarch64 compatibility
2. Download and prep datasets (Glaive + xLAM merged, BFCL harness installed)
3. Baseline eval — run base Qwen3 8B and Gemma 4 26B A4B against BFCL (pre-fine-tune numbers)
4. Fine-tune Qwen3 8B with LoRA on merged Glaive + xLAM dataset
5. Post-fine-tune eval — run fine-tuned Qwen3 8B against BFCL
6. Quantize fine-tuned Qwen3 8B to all target levels, benchmark each
7. Repeat steps 4-6 for Gemma 4 26B A4B
8. Cross-model analysis — compare dense vs MoE quantization curves
9. Generate charts, write up findings
10. Publish model weights, code, and write-up
9. Write up findings
10. Publish model, code, and write-up

---

## Dependencies to Install

```
transformers
peft
trl
datasets
bitsandbytes      # verify aarch64/Grace support — see open questions
gptqmodel         # successor to the now-unmaintained auto-gptq
autoawq
llama-cpp-python
wandb
accelerate
scipy
sentencepiece
protobuf
```

NVFP4/FP8 quantization and TRT-LLM serving come from the `nvcr.io/nvidia/tensorrt-llm` container rather than pip — don't try to pip-install those.

---

## Open Questions to Resolve Before Starting

- **bitsandbytes on ARM/Grace:** The Spark's CPU is ARM (aarch64). bitsandbytes' CUDA kernels (8-bit optimizer, 4-bit QLoRA) have historically had patchy aarch64 support. Confirm the current build works on DGX OS before committing to a bitsandbytes-dependent training recipe — if it doesn't, fall back to a non-bnb optimizer or use NeMo's quantized training path. Verify the same for AutoAWQ and GPTQModel wheels on aarch64.
- Does NeMo offer meaningful speedups over HuggingFace Transformers for LoRA on Blackwell, or is the ecosystem friction not worth it?
- Is the BFCL harness runnable as-is against a locally served model (e.g. via `trtllm-serve` / vLLM OpenAI-compatible endpoint), or does it need a custom model handler? BFCL has a "Prompt" vs "FC" (native function-calling) mode — decide which you're scoring, and be consistent.
- **MoE LoRA coverage:** When fine-tuning Gemma 4 26B A4B, only a subset of experts activate per example. Investigate whether PEFT/TRL handles this gracefully or whether you need extra epochs / specific expert-targeting strategies to ensure all experts get touched.
- **Gemma 4 MoE quantization support:** Verify that GPTQModel, AutoAWQ, and llama.cpp all handle the Gemma 4 MoE architecture cleanly. Some quantization tools have historically choked on MoE routing layers — test this early, not after you've already fine-tuned.
- For the quantization comparison: can you hold the inference runtime fixed across at least some formats, so at least part of your accuracy comparison is runtime-controlled rather than confounded?
