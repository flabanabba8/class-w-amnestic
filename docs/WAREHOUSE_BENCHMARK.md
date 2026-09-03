# Warehouse long-horizon benchmark

*Based on the paper: Badhe, Tiwari, Chung — SKILL.state, [arXiv:2608.26263](https://arxiv.org/abs/2608.26263) (SkillExecBench Warehouse).*

One order per step; the environment never restates inventory; score = correct orders / orders.


## mock — 300 orders, 12 shelves, seed 7

| mode | model | orders | score | correct | steps | model calls | input tok | output tok | max ctx chars | wall s | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| skillstate | mock | 300 | 1.00 | 300 | 300 | 301 | 407,867 | 28,778 | 5,486 | 1 | completed |
