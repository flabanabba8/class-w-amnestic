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

## Kimi K3 (NVIDIA route, no prompt caching, no route prefix) — 300 orders

_pending; per-call prompt tokens observed so far: ≈ 5.5K flat (1.4K context + 4.4K output schema), 0 wrong through order 14._
