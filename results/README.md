# Experimental Results & Summary Tables

This directory contains structured CSV result tables (`results/tables/`) and aggregated campaign-level summary JSONs (`results/summaries/`). Per-episode and step-level logs (`episodes.jsonl`, `transitions.jsonl`, `api.jsonl`) are omitted to keep the repository lightweight.

## Summary CSV Tables (`results/tables/`)

- `main_benchmark_results.csv`: Six-backbone main benchmark performance (ALFWorld, ScienceWorld, Combined Pooled SR, Macro Retrieval Tokens, ScienceWorld mean terminal Score, paired McNemar discordant counts, and exploratory 6- / 7-comparison Holm-adjusted $p$-values).
- `ablation_results.csv`: Eight-condition component ablation and estimator comparison on `DeepSeek-V4.1-Flash` ($50$ episodes $\times 2$ seeds per environment), including mechanism activation telemetry and cache/retry accounting.
- `counterfactual_2x2_results.csv`: Zero-overlap $2 \times 2$ counterfactual rule-pair intervention results across `DeepSeek-V4.1-Flash` and `Qwen3-8B`.
- `synthetic_diagnostics.csv`: Five-seed performance across the six controlled synthetic mechanism diagnostic scenarios.
- `token_efficiency_results.csv`: Price-agnostic end-to-end finalized logical LLM tokens per win across `DeepSeek-V4.1-Flash`, `GLM-5.3-Flash`, and `GPT-6-Luna`.
- `routing_latency_results.csv`: Per-episode elapsed routing latency (mean, median, P95, max in milliseconds).

## Aggregated Campaign Summary JSONs (`results/summaries/`)

- `main_deepseek_v41_flash.json`, `main_glm53_flash.json`, `main_gpt6_luna.json`, `main_qwen3_14b_awq.json`, `main_gemma4_12b_it.json`, `main_qwen3_8b.json`: Main-benchmark aggregated summaries per backbone.
- `ablation_deepseek_v41_flash.json`: Eight-condition ablation aggregated summary.
- `counterfactual_2x2_deepseek_v41_flash.json`, `counterfactual_2x2_qwen3_8b.json`: $2\times 2$ counterfactual summaries.
- `synthetic_diagnostics/*.json`: Six synthetic diagnostic scenario summaries.
