# GNN vs. Tabular Baselines for UPI-Style Fraud Ring Detection

Investigating whether Graph Neural Networks (GCN, GraphSAGE) meaningfully outperform
tabular machine learning (XGBoost) at detecting fraud rings and mule accounts in
payment networks — and, if so, under what conditions that advantage actually holds up.

## The problem

UPI processes billions of transactions monthly. Traditional rule-based fraud systems
evaluate each transaction in isolation, which means they miss coordinated fraud rings:
a mule account receiving many small, individually unremarkable payments from unrelated
"source" accounts (a **fan-in** pattern). This kind of fraud is only visible when you
look at the *shape* of the transaction network, not any single transaction.

## Research questions

1. Does graph structure meaningfully outperform tabular-only fraud detection?
2. Does GraphSAGE's inductive capability (generalizing to unseen accounts) hold up in
   a realistic scenario where new accounts constantly appear?
3. How does detection performance change as fraud rings become smaller, sparser, or
   more disguised (camouflaged among legitimate transactions)?

## Approach

Four detection approaches were compared, head-to-head, on identical data:

- **XGBoost, tabular-only** — account age, transaction amount stats. No graph
  information at all.
- **XGBoost + graph statistics** — the above, plus in-degree, out-degree, and
  PageRank computed from the transaction graph.
- **GCN** — a 2–3 layer Graph Convolutional Network, learning its own neighborhood
  aggregation directly from the graph.
- **GraphSAGE** — same idea as GCN, but using neighbor-sampling and feature
  concatenation instead of averaging, specifically designed to generalize to nodes
  never seen during training (inductive).

Each was tested on two datasets:

- **Synthetic UPI-style data** — a Barabási-Albert (hub-heavy, realistic) base
  network of 10,000 accounts, with 250 fraud rings (fan-in patterns) injected,
  each with randomized size and transaction-amount range.
- **Elliptic** — a real, publicly available, labeled Bitcoin transaction dataset
  (203,769 transactions, 2.2% illicit), used as an independent validation benchmark
  against published results (Weber et al., 2019).

A **robustness sweep** further tested all four models across three difficulty levels
(easy / smaller rings / camouflaged amounts) on the synthetic data, to see not just
which model wins on "easy" data, but which degrades most gracefully as fraud gets
harder to detect.

## Results

### Synthetic data — robustness sweep (F1 score)

| Model | Easy | Medium (smaller rings) | Hard (camouflaged amounts) |
|---|---|---|---|
| XGBoost, tabular-only | 0.750 | 0.044 | 0.234 |
| XGBoost + graph stats | 0.970 | 0.849 | 0.849 |
| GCN | 0.758 | 0.610 | 0.694 |
| GraphSAGE | 0.987 (inductive test) | 0.831 | 0.848 |

### Elliptic (real data)

| Model | F1 | PR-AUC |
|---|---|---|
| XGBoost, tabular-only | 0.795 | 0.803 |
| XGBoost + graph stats | 0.783 | 0.797 |
| GCN | 0.492 | 0.478 |
| GraphSAGE | 0.675 | 0.675 |

Published reference (Weber et al., 2019): plain GCN F1=0.628, Skip-GCN F1=0.705,
Random Forest F1=0.788.

## Key findings

1. **The value of graph statistics depends on what's already in the base features.**
   On synthetic data (deliberately structure-free tabular features), adding degree
   and PageRank caused a dramatic jump (F1 0.75 → 0.97). On Elliptic, where roughly
   half the given features already encode aggregated neighborhood information, adding
   more graph statistics provided no measurable benefit (F1 0.795 → 0.783).

2. **GraphSAGE consistently outperformed GCN**, on both datasets and at every
   difficulty level. This traces to a real architectural cause: GCN's neighbor
   *averaging* dilutes the sharp in-degree signal that defines fan-in fraud, while
   GraphSAGE's *concatenation* of a node's own features with its neighbor summary
   preserves that signal.

3. **GraphSAGE generalizes to entirely unseen accounts.** In a strict inductive test
   — 1,500 nodes and their edges fully removed during training, reintroduced only at
   test time — GraphSAGE scored F1=0.987. GCN and XGBoost+graphstats cannot be
   meaningfully tested this way at all: both require the complete graph (or
   precomputed graph statistics over it) to score any node.

4. **Simple graph statistics are a very strong, cheap baseline** — nearly matching
   full GNNs on "easy" fraud patterns. The real GNN advantage isn't raw accuracy on
   easy data; it's robustness under harder conditions and the ability to generalize
   to new accounts without retraining or recomputing statistics over the whole graph.

5. **Tabular, amount-based detection is fragile.** Its performance varied wildly
   (F1 0.13–0.75) across different random draws of the same generation parameters,
   depending entirely on incidental overlap between fraud and legitimate transaction
   amounts. Graph-structural methods remained stable regardless, since they don't
   depend on amount separability at all.

6. **Consistent with the original Elliptic paper's own finding** that tree-based
   methods (Random Forest) outperformed GCN — our results independently reproduce
   this pattern on both a synthetic and a real dataset.

## Repository structure

```
gnn-upi/
├── notebooks/
│   ├── 01_synthetic_graph_gen.ipynb       — builds the synthetic UPI graph + fraud rings
│   ├── 02_baseline_xgboost_tabular.ipynb  — XGBoost, no graph information
│   ├── 03_baseline_xgboost_graphstats.ipynb — XGBoost + degree/PageRank
│   ├── 04_gcn.ipynb                       — GCN (transductive)
│   ├── 05_graphsage.ipynb                 — GraphSAGE (inductive test on unseen nodes)
│   ├── 06_elliptic_validation.ipynb       — all four models re-run on Elliptic
│   └── 07_robustness_sweep.ipynb          — all four models across 3 difficulty levels
├── src/
│   └── graph_gen.py                       — reusable synthetic graph generator
├── data/                                  — saved graphs, features, and results (JSON)
└── paper/                                 — writeup and figures
```

## Limitations & honest caveats

- Synthetic fraud amounts were, in some configurations, drawn from a narrower range
  than legitimate transactions — this gave the tabular-only baseline an incidental
  amount-based shortcut in some runs, which is exactly what the "camouflaged" sweep
  config was designed to remove.
- Elliptic's test window (timesteps 35–49) includes a documented dark-market shutdown
  event, which the original paper's authors note degrades all models' performance —
  likely explains part of the gap between our GCN result and the original paper's.
- GCN's neighbor-averaging is a poor architectural fit for anomaly patterns defined by
  a node looking *different* from its neighbors (heterophily) — a known, documented
  limitation of vanilla GCN for fraud/anomaly detection tasks.

## Future work

- Elliptic inductive test (mirroring notebook 05's approach on real data)
- A fourth, more extreme sweep config combining smallest ring size + camouflage +
  sparsity simultaneously
- Full-scale production/MLOps extension: Kafka-based real-time ingestion, precomputed
  GNN embeddings served via Redis for sub-50ms live scoring, periodic offline
  retraining — deliberately scoped out of this research phase

## References

- Weber, M. et al. (2019). *Anti-Money Laundering in Bitcoin: Experimenting with
  Graph Convolutional Networks for Financial Forensics.* [arXiv:1908.02591](https://arxiv.org/abs/1908.02591)
- Kipf, T. & Welling, M. (2017). *Semi-Supervised Classification with Graph
  Convolutional Networks.*
- Hamilton, W., Ying, R., & Leskovec, J. (2017). *Inductive Representation Learning
  on Large Graphs.*
