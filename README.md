# Combinatorial Credit Assignment and Synergy-Aware Memory Pruning for Language Agents (CoMeP)

**CoMeP** is a zero-manager-LLM-call online memory routing and pruning framework for language agents operating under strict retrieval and storage token budgets.

It provides a modular codebase for evaluating and developing online memory managers across two interactive embodied benchmarks—**ALFWorld** (household manipulation) and **ScienceWorld** (elementary science experimentation)—by jointly estimating context intercepts, relevance-weighted main effects, and sparse pairwise rule interactions via regularized second-order logistic regression.

🚩 **News (2026.11)** We release the CoMeP implementation, benchmark split manifests, train-derived warm-start rule pools, offline verification suites, and campaign runners.

---

## Benchmark Summary

Across a selected six-backbone subset on held-out **ALFWorld** (`valid_unseen`) and **ScienceWorld** (`test`), the main benchmark covers **17,325 displayed episodes** across 7 main-table methods $\times$ 3 seeds (from a **19,800-episode** 8-method campaign that also runs the internal `Additive Laplace UCB` ablation), complemented by **1,600 ablation episodes** and **248 zero-overlap $2 \times 2$ counterfactual rollouts**:

| Backbone Configuration | Displayed Eps / Method | Extra Fmt. Retries | Per-Replicate Cache | ALFWorld SR (%) | ScienceWorld SR / Late (%) | Combined Pooled SR (%) |
|---|---:|---|---|---|---|---|
| **DeepSeek-V4.1-Flash** (Hosted API) | 495 ($90\times 3 + 75\times 3$) | 0 | Disabled | **88.9 ± 3.8** | 57.8 ± 8.6 / **66.7** | **74.7%** |
| **GLM-5.3-Flash** (Hosted API) | 495 ($90\times 3 + 75\times 3$) | Up to 2 | Enabled | 89.6 ± 1.3 | 41.3 ± 3.5 / 41.2 | 67.7% |
| **GPT-6-Luna** (Hosted API) | 495 ($90\times 3 + 75\times 3$) | Up to 2 | Enabled | **75.9 ± 2.3** | 52.0 ± 4.6 / 49.1 | **65.1%** |
| **Qwen3-14B-AWQ** (4-bit AWQ, L4) | 330 ($60\times 3 + 50\times 3$) | Up to 2 | Enabled | 60.6 ± 5.9 | **12.0 ± 5.3** / **13.3** | **38.5%** |
| **Gemma-4-12B-it** (W4A16 QAT + Logit Bias, L4) | 330 ($60\times 3 + 50\times 3$) | Up to 2 | Enabled | 17.2 ± 7.5 | **30.7 ± 6.1** / **37.3** | **23.3%** |
| **Qwen3-8B** (BF16, L4) | 330 ($60\times 3 + 50\times 3$) | 0 | Disabled | **46.7 ± 5.0** | **12.7 ± 5.0** / **13.3** | **31.2%** |

> **Evidentiary Boundaries:** CoMeP achieves the highest descriptive episode-weighted aggregate success rate (**53.90%**) and equal-backbone macro rate (**50.08%**), leads main-table pooled point estimates on **5/6 backbones** (4/6 when including `Additive Laplace UCB`, which reaches **67.3%** vs. **65.1%** on `GPT-6-Luna`, non-significant at $p_{\text{Holm}}=0.2664$), and uses fewer environment-averaged retrieval tokens than unpruned `Full-Memory (Top-k)` on all six. Across all six backbones, pooled differences vs. `Additive Laplace UCB` remain non-significant; the six-backbone subset is retrospectively selected excluding an earlier 48-run campaign (`34.9%` pooled); and paired McNemar, 2-seed ablation, and small-tail $2\times 2$ tests are exploratory indicators.

**Implemented methods and baselines in this repo.** ☑ indicates implementations included under `comep/`:

- ☑ **CoMeP (Ours)** — Context-stratified second-order Laplace credit assignment, exposure-gated lookahead routing, and coalition-aware pruning (`comep/core.py`).
- ☑ **Additive Laplace UCB** — First-order additive Laplace ablation with $s_{ij} \equiv 0$ (`comep/core.py`).
- ☑ **EvoMemGuard-style** — Author-designed rule-level DeBERTa-v3 NLI + utility adaptation inspired by Wang et al. (EECCT 2026) (`comep/evomemguard.py`).
- ☑ **ExpeL-derived Online** — Matched-budget online trajectory-critic baseline adapted from Zhao et al. (AAAI 2024) (`comep/critic.py`).
- ☑ **MMR-style Diversity** — Cosine-redundancy diversity reranking and eviction inspired by Carbonell & Goldstein (SIGIR 1998) (`comep/core.py`).
- ☑ **Individual Credit** — Independent $+1/-1$ per-rule outcome counters (`comep/core.py`).
- ☑ **Full-Memory (Top-k)** & ☑ **No-Memory** — Unpruned top-$k$ relevance retrieval and memoryless baselines (`comep/core.py`).

---

## Usage

1. **Install Python >= 3.11 and dependencies.**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[plots,semantic,tokens]'
```

2. **Prepare split configs and pretrained embedding weights.** Disjoint train/dev/test manifests and curated 18-rule warm-start pools are included under `./configs/`. To run semantic-embedding regression tests locally, download `BAAI/bge-small-en-v1.5`:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-small-en-v1.5', local_dir='models/bge-small-en-v1.5')"
```

3. **Verify and run experiments.**

```bash
# Clean-clone offline verification (no local model weights or external API calls)
python scripts/verify.py all

# Revised-protocol regression suite (requires local BGE weights, zero external LLM calls)
python scripts/verify_revised_protocol.py

# Early synthetic diagnostic simulation & plotting
python -m comep synthetic --config configs/synthetic.toml --output outputs/demo
python -m comep report outputs/demo

# Full benchmark suite runner (requires ALFWorld / ScienceWorld and configured LLM API keys)
python scripts/run_benchmark_suite.py --help
```

4. **Repository structure.**

- `./comep/`: Core memory managers, Laplace solver, routing/pruning logic, LLM client adapters, and environment wrappers.
- `./configs/`: Split manifests (`alfworld_*.json`, `scienceworld_*.json`), warm-start pools (`*_warmstart_pool.json`), and TOML/JSON campaign configs.
- `./scripts/`: Verification suites (`verify*.py`), campaign runners (`run_benchmark_suite.py`, `run_campaign.py`), and analysis/export scripts.
- `./results/`: Structured CSV result tables (`results/tables/`) and aggregated campaign summary JSONs (`results/summaries/`).

---

## Acknowledgement

This codebase builds on publicly released environments and pretrained encoder ecosystems:

- **ALFWorld**: https://github.com/alfworld/alfworld
- **ScienceWorld**: https://github.com/allenai/ScienceWorld
- **BGE (`BAAI/bge-small-en-v1.5`)**: https://huggingface.co/BAAI/bge-small-en-v1.5
- **DeBERTa-v3 NLI (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`)**: https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli

---

## Citation

If you find this repository helpful in your research, please cite:

```bibtex
@article{liu2026comep,
  title={Combinatorial Credit Assignment and Synergy-Aware Memory Pruning for Language Agents},
  author={Liu, Yueyu and Jiao, Minghai},
  year={2026},
  url={https://github.com/Mufumoca/comep}
}
```
