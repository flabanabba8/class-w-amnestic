# Service-registry long-horizon benchmark

*Based on the paper: Badhe, Tiwari, Chung — SKILL.state: Scalable Long-Horizon Agent Skills, [arXiv:2608.26263](https://arxiv.org/abs/2608.26263).*

`src/mnestic/benchmarks/registry.py`, `scripts/registry_bench.py`. A second long-horizon task, built after the warehouse
benchmark exposed that the runtime was making the model keep the books. A registry of 12 services (port, status,
dependencies) receives 200 sequential orders: mutations (`set_port`, `set_status`, `add_dep`, `remove_dep`) and queries
(`port`, `status`, `deps`, and `dependents` — a reverse-graph question that needs aggregation across every record).
The `registry` tool behaves like any API: it returns the resource it touched as `facts`, and the **generic** tool-fact
ledger keeps the latest record per service under `domain.registry.<service>`. There is no registry-specific code in the
runtime; the same mechanism stores a file's hash after `read_text_file` or a shelf's contents in the warehouse.

## Results — 200 orders, 12 services, seed 7

| model | mode | score | prompt tokens (provider) | per call | wall | notes |
|---|---|---:|---:|---|---:|---|
| Sonnet 5 | SKILL.state | 0.97 | 1.26M (69% cached) | 6.0K → 6.4K, flat | 10 min | all 6 misses = `dependents` aggregation |
| Sonnet 5 | ReAct | 0.98 | 3.25M (99% cached) | 2.8K → 29.7K | 5 min | 3 misses (1 wrong action, 2 dependents) |
| Gemma 4 12B | ReAct | **0.99** | 5.5M (50% KV reuse) | up to ~22K | 9 min | 2 dependents misses |
| Gemma 4 12B | SKILL.state, untyped tool args | 0.54 | 0.88M | flat 8K chars | 17 min, failed | wrote `answer="port"` — the query's name, not its value |
| Gemma 4 12B | SKILL.state, typed tool args | 0.54 | 0.81M | flat | 12 min, failed | schema can't help: `"port"` is a valid string |
| Gemma 4 12B | SKILL.state, + example values in the tool description, effort none | **0.91** | 0.74M | flat | 9 min | 10 dependents misses, 7 wrong actions |
| Gemma 4 12B | SKILL.state, same, effort low | **1.00 over 76** | 0.25M | flat | then stalled | thinking budget exhaustion ended the run (see warehouse notes on llama.cpp reasoning) |
| scripted optimal policy | SKILL.state | 1.00 | — | flat | 1 s | harness ceiling |

## Reading

- **Strong model: parity again** (0.97 vs 0.98) at 2.6× fewer processed tokens — but *not* faster this time: a 99%-cached
  transcript at 200 orders is cheap for the provider to re-read, while our per-call prompt has a larger uncached share.
- **Small model: the interface, not the memory, was the problem.** With the ledger correct in its state, Gemma still answered
  queries with the query's *name* until the tool's argument description carried example values — then 0.54 → 0.91 at zero
  reasoning effort, and 100% (over the 76 orders it completed) with a little thinking. As a ReAct agent it scored 0.99 with
  the same tool: re-reading a transcript full of its own earlier correct answers is a form of in-context example that our
  bounded prompt does not provide. Lesson for tool authors on this runtime: argument descriptions need example values;
  that is the only in-context example a stateless step gets.
- **Remaining structural gap:** `dependents` needs aggregation over 12 ledger entries; every model misses some of these in
  both modes. A derived/indexed view (the runtime maintaining a reverse index from tool facts) would remove the reasoning
  step, but it is task-specific unless expressed as a generic "index facts by field" option — not done.

## Warehouse, generic ledger (no task-specific bookkeeping code)

With the hand-written `adjust` effects removed and the warehouse tool simply reporting the shelf it touched, Gemma 4 12B
scores **0.99 (298/300)** on the 300-order warehouse — identical to the hand-written version. See `WAREHOUSE_BENCHMARK.md`.
