# GNN vs. Tabular Baselines for UPI-Style Fraud Ring Detection

Investigating whether Graph Neural Networks (GCN, GraphSAGE) practically outperform tabular ML (XGBoost) at spotting mule accounts and fraud rings in payment networks, and pinning down when that extra engineering overhead is actually worth it.

## The Core Problem

UPI moves billions of transactions every month. Rule-based engines evaluate transactions one by one, completely missing coordinated rings: a single mule account quietly collecting small, routine transfers from dozens of unrelated accounts (fan-in pattern). Spotting this requires looking directly at the topology of the payment network rather than individual transaction amounts.

## Research Questions

- Does graph structure beat tabular-only baselines when identifying ring patterns?
- Does GraphSAGE generalize cleanly to completely new accounts created after training?
- How far does detection drop when rings shrink, turn sparse, or blend their amounts into normal traffic?

## Benchmarked Approaches

Four models tested on identical splits:

- **XGBoost (Tabular Only)** — Baseline using only raw features like account age and transaction amount aggregates.
- **XGBoost + Graph Stats** — The tabular features plus computed in-degree, out-degree, and PageRank.
- **GCN** — 2–3 layer Graph Convolutional Network learning representations across neighbors via normalized averaging.
- **GraphSAGE** — Inductive graph neural network using neighbor sampling and feature concatenation to handle unseen nodes.

## Datasets Used

- **Synthetic UPI Network** — Barabási-Albert scale-free network (10,000 accounts) injected with 250 fan-in fraud rings varying in size and transaction distributions.
- **Elliptic Benchmark** — Public Bitcoin dataset (203,769 transactions, 2.2% illicit) used to check behavior against Weber et al. (2019).

## Benchmark Results

### Synthetic Data: Robustness Sweep (F1 Score)

| Model | Easy Setup | Medium (Smaller Rings) | Hard (Camouflaged Amounts) |
|---|---|---|---|
| XGBoost (Tabular Only) | 0.750 | 0.044 | 0.234 |
| XGBoost + Graph Stats | 0.970 | 0.849 | 0.849 |
| GCN | 0.758 | 0.610 | 0.694 |
| GraphSAGE | 0.987 (Inductive) | 0.831 | 0.848 |

### Elliptic Data (Real-World Baseline)

| Model | F1 Score | PR-AUC |
|---|---|---|
| XGBoost (Tabular Only) | 0.795 | 0.803 |
| XGBoost + Graph Stats | 0.783 | 0.797 |
| GCN | 0.492 | 0.478 |
| GraphSAGE | 0.675 | 0.675 |

Published reference numbers (Weber et al., 2019): Random Forest F1=0.788, Skip-GCN F1=0.705, Standard GCN F1=0.628.

## Technical Insights

- **Feature saturation dictates graph utility.** Adding explicit graph stats caused a jump on synthetic data (F1 0.75 to 0.97) because the tabular features had zero structural context. On Elliptic, where the raw features already bundle local neighborhood aggregations, adding basic graph metrics did nothing (F1 0.795 to 0.783).
- **GraphSAGE preserves the fan-in signal better than GCN.** GCN's neighbor averaging smooths out and drowns the sudden spike in incoming edges. GraphSAGE concatenates a node's own vector with its aggregated neighbors, preserving the anomaly signal.
- **GraphSAGE handles strict inductive splits.** When 1,500 nodes and their edges were completely hidden during training and introduced only at inference, GraphSAGE hit an F1 of 0.987. Neither standard GCN nor XGBoost with precomputed graph stats can score new nodes without recalculating metrics across the full graph.
- **Amount-only tabular models break easily.** Tabular XGBoost scores swung wildly (F1 between 0.13 and 0.75) depending entirely on whether the random data generator picked overlapping amount ranges for normal vs. fraud traffic. Structural methods avoided this dependency entirely.
- **Tree models remain tough baselines on pre-aggregated data.** Replicating the Elliptic paper showed that classical tree-based models beat standard GCNs when features already capture local graph context.

## Repository Layout

```
gnn-upi/
├── notebooks/
│   ├── 01_synthetic_graph_gen.ipynb          Synthetic UPI graph and ring injection
│   ├── 02_baseline_xgboost_tabular.ipynb     Tabular-only XGBoost baseline
│   ├── 03_baseline_xgboost_graphstats.ipynb  XGBoost with Degree and PageRank
│   ├── 04_gcn.ipynb                          Transductive GCN implementation
│   ├── 05_graphsage.ipynb                    Inductive GraphSAGE on unseen nodes
│   ├── 06_elliptic_validation.ipynb          Elliptic benchmark replication
│   └── 07_robustness_sweep.ipynb             3-tier difficulty comparison
├── src/
│   └── graph_gen.py                          Reusable graph generation script
├── data/                                     Saved graphs, features, and run logs
└── paper/                                    Notes and evaluation plots
```

## Constraints & Realities

- **Synthetic shortcuts:** Some baseline configurations allowed tabular models to separate classes simply by amount thresholds. The camouflaged sweep fixed this, and tabular F1 immediately collapsed to 0.234.
- **Elliptic distribution shift:** The test split (timesteps 35 to 49) spans a real-world darknet market shutdown, degrading all models and explaining the gap between standard results and our GCN run.
- **Heterophily penalty:** Vanilla GCN assumes connected nodes share labels (homophily). Fraud rings represent heterophilous anomalies where mules connect to regular victims, making standard graph convolutions a poor mechanical fit.

## Next Steps

- Run a strict inductive test on Elliptic to match the notebook 05 setup on real-world data.
- Build a stress-test scenario that combines minimal ring sizes, camouflaged amounts, and extreme edge sparsity simultaneously.
- Prototype production serving: stream raw events through Kafka, cache node embeddings in Redis for sub-50ms inference, and handle batch GNN graph retraining on a separate offline schedule.

## References

- Weber, M. et al. (2019). *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics.* arXiv:1908.02591
- Kipf, T. & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks.*
- Hamilton, W., Ying, R., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs.*
