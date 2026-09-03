# Warehouse long-horizon benchmark

*Based on the paper: Badhe, Tiwari, Chung — SKILL.state: Scalable Long-Horizon Agent Skills, [arXiv:2608.26263](https://arxiv.org/abs/2608.26263) (SkillExecBench Warehouse).*

`src/mnestic/benchmarks/warehouse.py`, `scripts/warehouse_bench.py`. A deterministic environment hands the agent one
order per step — *store N item* (pick a shelf with room), *ship 1 item* (pick a shelf that has it), *count item* — and
never restates the inventory. Every order is scored against the simulation. The same environment and tool are driven
two ways with the same model: **skillstate** (this runtime: skill + state + newest observation per call) and **react**
(a plain PydanticAI agent whose tool loop accumulates the whole transcript).

## Results — 60 orders, 12 shelves, seed 7, via 9Router

Token figures are what the provider reported through 9Router (`prompt_tokens` includes cache hits), which is the
like-for-like measure; the runtime's own `input_tokens` column only counts uncached tokens.

| model | mode | score | prompt tokens (all calls) | of which cached | per call, first → last | max transcript/context |
|---|---|---:|---:|---:|---|---:|
| Claude Sonnet 5 | skillstate | **0.98** | 706,598 | 88% | 11.5K → 11.6K (flat) | 5,438 chars |
| Claude Sonnet 5 | react | **0.98** | 431,579 | 97% | 2.2K → 11.9K (linear) | 132,441 chars |
| Claude Sonnet 5 | skillstate, **slim schema + feedback fix** | **1.00** | **365,200** | 76% | 5.9K → 6.0K (flat) | 5,501 chars |
| Claude Haiku 4.5 | skillstate | 0.80 | 645,996 | 80% | ~9.5K flat | 6,940 chars |
| Claude Haiku 4.5 | react | **1.00** | 781,952 | 94% | ~2.2K → 24K (linear) | 165,255 chars |
| scripted optimal policy (no model) | skillstate, 300 orders | 1.00 | — | — | context 5,486 chars flat | 5,486 chars |

### What this says, honestly

1. **Accuracy at 60 orders: no SKILL.state advantage.** Sonnet ties (one arithmetic slip each); Haiku is *worse*
   under SKILL.state (0.80 vs 1.00). At this horizon a 165K-char transcript still fits comfortably and the model can
   re-derive the inventory from the full history at every step; under SKILL.state a single bad patch corrupts the
   table with no second chance. Haiku's misses were (a) losing the pending order when a patch was rejected — a runtime
   gap, fixed since (feedback observations now repeat the input they correct) — and (b) inventing plan-step ids.
2. **Tokens at 60 orders: ReAct is cheaper on this route.** SKILL.state's per-call cost is *flat but high*: on the
   Claude Code route each call carries ≈ 5.7K tokens of that route's own system prompt + ≈ 4.4K tokens of the
   `AgentDecision` output schema + ≈ 1.4K of actual context. The ReAct transcript only overtakes that at ~order 60. And
   prompt caching favours append-only transcripts (97% cache hits) over a mutable state block (88%).
3. **Where the architecture wins is beyond the crossover.** ReAct grows ~160 prompt tokens per order; SKILL.state does
   not grow. Extrapolating the measured slopes: at 300 orders ReAct ≈ 7.8M prompt tokens vs 3.5M; at 1,000 orders
   ≈ 82M vs 11.5M — and past the context window ReAct cannot run at all. The scripted 300-order run and the Kimi
   300-order run (below) are the empirical part of that claim.
4. **The fixed per-call overhead was the thing to fix, and fixing it flips the result at 60 orders.** Pruning the
   output schema to the five patch ops the skill declares (`allowed_ops`; 18K → 5.8K chars) halved the per-call
   cost (11.5K → 5.9K prompt tokens). Rerun: Sonnet **1.00 with 365K prompt tokens** — fewer tokens than the ReAct
   run (432K) *and* a perfect score, with the context still flat at 5.5K chars. The remaining fixed cost on this
   route is the Claude Code system prompt (~4.4K of the 5.9K), which is not ours.

Caveat on cache columns: through 9Router's OpenAI-compatible endpoint the cache hit counts arrive in a non-standard
field, so the runtime's `cache_read_tokens` reads 0 for these runs; the percentages above come from 9Router's own log.

## Gemma 4 12B (local llama.cpp, Q8, RTX 4090) — 30 orders, `native` (grammar) mode

| setting | score | calls | s/call | prompt tokens | notes |
|---|---:|---:|---:|---:|---|
| `reasoning_effort: low`, max_tokens 4000 | (8/8 then failed) | 11 | 9–30 | — | three consecutive steps spent the whole 4,000-token budget thinking about inventory arithmetic and emitted no JSON |
| `reasoning_effort: none`, max_tokens 1500 | **0.87** (26/30) | 31 | **2.6** | 80,697 (31% cached) | zero malformed decisions (grammar), zero rejected patches; all 4 misses = trying to store on a shelf it believed had room (S03, actually full) |

A local 12B model runs the loop end to end at 2.6 s per step with a flat 5.6K-char context; its misses are arithmetic,
not format. Compared with the same model's earlier attempts, the schema fixes (string bounds, required discriminators)
and the cached op reference are what made grammar mode usable at all.

## 300 orders — SKILL.state vs ReAct, same model, same seed

Provider-side prompt tokens (9Router / llama.cpp logs; includes cache hits). "SKILL.state" rows run on the current build
(slim schema, seeded Σ₀, `inspect` verification, tools-only actions) unless marked *old*; the Sonnet/Kimi ReAct runs
started before `inspect` existed and could not verify.

| model | mode | score | prompt tokens, all calls | per call, first → last | output tokens | wall | notes |
|---|---|---:|---:|---|---:|---:|---|
| **Sonnet 5** | SKILL.state + inspect | **0.95** | ≈ 1.03M | 6.0K → 6.2K, **flat** | 99K | **22 min** | 17 inspections; 12 count / 3 store / 1 ship misses |
| Sonnet 5 | SKILL.state, no inspect | 0.85 | ≈ 1.12M | flat | 108K | 24 min | drift never corrected |
| Sonnet 5 | ReAct | 0.96 | ≈ 8.5M (92% cache hits) | 2.2K → **51K** | 417K | 97 min | transcript 659K chars at the end |
| **Kimi K3** | ReAct | **0.98** | 7.7M (no caching) | 1.0K → **51K** | 34K | 87 min | strongest raw result; expensive |
| Kimi K3 | SKILL.state (*old build*) | 0.88 | 1.76M | 5.6K → 5.7K, flat | 94K | 2.9 h (NVIDIA stalls) | before inspect/seed/slim; matched rerun pending |
| Gemma 4 12B (local) | ReAct | 0.83 | 6.7M (50% KV-cache reuse) | ~1K → 22K | 9K | 12 min | 131K window, no truncation |
| Gemma 4 12B | SKILL.state, seeded + inspect | 0.48 | 0.80M | flat 5.4K chars | 62K | 13 min | 82% in the first 50 orders → 23% in the last: arithmetic drift, 8 inspections |
| scripted optimal policy | SKILL.state | 1.00 | — | flat 5.5K chars | — | 1 s | harness ceiling |

### Reading

- **Tokens: decisive.** SKILL.state's per-call cost is the same at order 300 as at order 1; ReAct's grows ~165 tokens per
  order to 51K. Over 300 orders that is 7–8× fewer prompt tokens for Sonnet/Kimi and 4× fewer wall-clock minutes for
  Sonnet (22 vs 97), and the gap widens with every additional order. Caching softens the *bill* for ReAct on routes
  that have it (92% hits) but not the latency: the model still reads 51K tokens per order.
- **Accuracy: tied for a strong model with verification, not better.** Sonnet 0.95 vs 0.96. Without the `inspect`
  tool the state-based agent drifts (0.85) because a wrong shelf belief is never contradicted; with it, it verifies
  ~once per 18 orders and recovers. The paper's accuracy gain did not reproduce here; parity at 1/8 the cost did.
- **Small models: the transcript wins.** Gemma 12B cannot maintain a 12-line inventory through 300 arithmetic
  updates (0.48) but can re-derive answers from raw history (0.83). This is the paper's own small-model finding
  (state overwrite errors dominate) seen from the other side: a curated memory is only as good as the curator.
- **What would move the small-model number** (not run, since it changes the task): the environment echoing the
  shelf's new contents after each store/ship, so updates are copies instead of arithmetic — a real WMS does that —
  or a skill rule forcing an inspect before every count and after every rejection.

## Kimi K3 (NVIDIA route, no prompt caching, no route prefix) — 300 orders, old build

| score | correct | steps | calls | prompt tokens (provider, no caching) | per call first → last | context chars first → last | wall |
|---:|---:|---:|---:|---:|---|---|---:|
| **0.88** | 265/300 | 341 | 342 | 1,761,269 (≈ 5.9K/call) | 5,580 → 5,698 (**flat**) | 5,177 → 5,782 (**flat**) | 2.9 h |

Run on the pre-slim schema (30 ops) and before the feedback fix; NVIDIA's endpoint has no prompt caching, so these are
full-price tokens. The 35 misses are count arithmetic (e.g. "clamp total is 4") and stale shelf beliefs, not format
errors (1 rejected patch in 342 calls). The per-call cost at order 300 is the same as at order 1 — the property the
architecture exists for. For comparison, ReAct's measured slope (~160 tokens/order) puts a 300-order transcript agent at
≈ 50K tokens per call by the end and ≈ 7.8M cumulative, 4.4× this run — and that is before the transcript starts
colliding with context limits. With the slim schema the same run would cost ≈ 2.5K tokens/call.
