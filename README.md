# GNN vs. Tabular Baselines for UPI-Style Fraud Ring Detection

This project checks whether Graph Neural Networks (GCN, GraphSAGE) actually beat plain
tabular machine learning (XGBoost) at spotting fraud rings and mule accounts in payment
networks. And if they do, under what conditions that advantage holds up.

## The problem

UPI processes billions of transactions a month. Rule-based fraud systems check each
transaction on its own, so they miss coordinated rings: a mule account collects many
small payments from unrelated "source" accounts, and each individual payment looks
fine. This is a fan-in pattern. You can only see it by looking at the shape of the
transaction network, not any single transaction in isolation.

## Research questions

1. Does graph structure actually improve fraud detection over tabular-only methods?
2. Does GraphSAGE generalize to accounts it never saw during training? New accounts
   show up on UPI constantly, so this matters for any real deployment.
3. How does detection performance change as fraud rings get smaller, sparser, or
   camouflaged among normal-looking transactions?

## Approach

Four models, tested head to head on the same data:

- **XGBoost, tabular-only.** Account age and transaction amount stats. No graph
  information.
- **XGBoost + graph statistics.** Same as above, plus in-degree, out-degree, and
  PageRank computed from the transaction graph.
- **GCN.** A 2-3 layer Graph Convolutional Network. Learns its own neighborhood
  aggregation directly from the graph structure.
- **GraphSAGE.** Same general idea, but uses neighbor sampling and concatenates a
  node's own features with its neighbor summary instead of averaging them. Built
  specifically to generalize to nodes it never trained on.

Two datasets:

- **Synthetic UPI-style data.** A Barabási-Albert base network of 10,000 accounts,
  with 250 fraud rings injected. Each ring has a randomized number of source accounts
  and its own transaction-amount range.
- **Elliptic.** A real, public, labeled Bitcoin transaction dataset (203,769
  transactions, 2.2% illicit), used to check results against a real benchmark
  (Weber et al., 2019).

A robustness sweep tested all four models across three difficulty levels (easy,
smaller rings, camouflaged amounts) to see which model holds up as the fraud pattern
gets harder to detect, not just which one wins on the easy case.

A 5-seed variance run was added after the first pass through the sweep showed the
tabular baseline swinging wildly between runs. This is discussed below.

## Results

### Synthetic data, first run (notebooks 02-05)

The original run on the synthetic graph: 10,000 accounts, 250 fraud rings, 15-40
sources per ring, a tight fraud-amount range.

| Model | F1 | PR-AUC | Notes |
|---|---|---|---|
| XGBoost, tabular-only | 0.750 | 0.854 | No graph information |
| XGBoost + graph stats | 0.970 | 0.999 | in_degree alone accounted for 95.8% of feature importance |
| GCN | 0.758 | 0.623 | 3-layer, tuned class weights and decision threshold |
| GraphSAGE (inductive) | 0.987 | 0.999 | Tested on 1,500 nodes fully unseen during training |

### Seed variance (5 seeds, easy config)

The single-run numbers above turned out not to tell the full story. Rerunning the
easy config across 5 different random seeds gave this:

| Model | Mean F1 | Std dev |
|---|---|---|
| XGBoost, tabular-only | 0.106 | 0.036 |
| XGBoost + graph stats | 0.970 | 0.015 |
| GCN | 0.855 | 0.064 |
| GraphSAGE | 0.965 | 0.049 |

The tabular-only baseline's original F1=0.750 was, in hindsight, closer to a lucky
outlier than a typical result. Its real average across seeds is much lower, and it
swings a lot. XGBoost with graph stats and GraphSAGE both stay strong and comparatively
stable. GCN is solid but the most variable of the graph-aware methods.

A paired t-test across the same 5 seeds:

- GraphSAGE vs. XGBoost + graph stats: t = -0.314, p = 0.769. Not significant. On the
  easy synthetic case, these two are statistically indistinguishable, so we can't
  claim GraphSAGE beats XGBoost + graph stats here.
- GraphSAGE vs. GCN: t = 3.003, p = 0.040. Significant. GraphSAGE does reliably
  outperform GCN.

### Robustness sweep (F1, single seed per config)

Same generation parameters, rebuilt through a reusable function for this experiment.
A different random draw of the underlying numbers meant this run's "easy" config
came out weaker for tabular-only than the original run above, purely by chance (see
Limitations). Read this table for the trend across difficulty levels, not as a
second version of the numbers above.

| Model | Easy | Medium (smaller rings) | Hard (camouflaged amounts) |
|---|---|---|---|
| XGBoost, tabular-only | 0.134 | 0.044 | 0.234 |
| XGBoost + graph stats | 0.970 | 0.849 | 0.849 |
| GCN | 0.959 | 0.610 | 0.694 |
| GraphSAGE | 1.000 | 0.831 | 0.848 |

### Elliptic (real data)

| Model | F1 | PR-AUC |
|---|---|---|
| XGBoost, tabular-only | 0.795 | 0.803 |
| XGBoost + graph stats | 0.783 | 0.797 |
| GCN | 0.492 | 0.478 |
| GraphSAGE | 0.675 | 0.675 |

For reference, Weber et al.'s original paper reports: plain GCN F1=0.628, Skip-GCN
F1=0.705, Random Forest F1=0.788.

## What the results actually show

**Graph statistics only help when the base features don't already carry structural
information.** On the synthetic data, the tabular features (account age, transaction
amounts) were deliberately built with zero structural information, so adding degree
and PageRank caused a large jump, F1 going from around 0.75-0.11 up to 0.97. On
Elliptic, roughly half of the given 165 features are already aggregated statistics
computed from each transaction's neighbors. Adding more graph statistics on top of
that gave nothing back, F1 went from 0.795 to 0.783, basically flat.

**GraphSAGE beats GCN, consistently, and this time we checked it wasn't noise.**
It won on both datasets, at every difficulty level in the sweep, and the difference
against GCN specifically passed a paired t-test (p = 0.040). The likely reason is
architectural: GCN averages a node's features together with its neighbors', which
dilutes the sharp in-degree spike that defines a fan-in pattern. GraphSAGE keeps a
node's own features and its neighbor summary as two separate pieces, concatenated
rather than blended, so that spike survives into the final prediction.

**GraphSAGE and XGBoost with graph stats are not distinguishable on easy data.**
This came out of the 5-seed variance run and it's worth stating plainly: the original
single-run comparison suggested GraphSAGE was ahead (0.987 vs 0.970), but across 5
seeds the gap disappeared (p = 0.769). GraphSAGE's real advantage isn't raw accuracy
here.

**Where GraphSAGE actually earns its complexity is generalization to new accounts.**
In a strict test, 1,500 nodes and every edge touching them removed entirely from
training, added back only at test time, GraphSAGE scored F1 = 0.987. Neither GCN nor
XGBoost with graph stats can be tested this way at all. Both need the full graph, or
statistics computed over it, before they can score anyone. On a live UPI network where
new accounts open constantly, that's a real, structural limitation for both of them,
not a tuning problem.

**Tabular, amount-based detection is genuinely fragile.** Its F1 ranged from 0.106 to
0.750 depending on how the random fraud-amount draw happened to overlap with normal
transaction amounts. Graph-based methods stayed steady across the same variation,
since they never depended on amount separability to begin with.

**Tree-based methods beating GCN isn't unique to this project.** Weber et al. found
the same thing in their original paper: Random Forest outperformed GCN. Our results
on two separate datasets land on the same conclusion independently.

## Repository structure

```
gnn-upi/
├── notebooks/
│   ├── gnn-upi-graph.ipynb        builds the synthetic UPI graph and fraud rings
│   ├── gnn-upi-xgb.ipynb   XGBoost, no graph information
│   ├── gnn-upi-xgb-graphf.ipynb XGBoost plus degree and PageRank
│   ├── gnn-upi-gcn.ipynb                        GCN, transductive
│   ├── gnn-upi-graphsage.ipynb                  GraphSAGE, inductive test on unseen nodes
│   ├── gnn-upi-elliptic(1).ipynb        all four models rerun on Elliptic
│   └── gnn-upi-final.ipynb           difficulty sweep, seed variance, t-tests
├── src/
│   └── graph_gen.py                        reusable synthetic graph generator
├── data/                                   saved graphs, features, and results (JSON)
└── paper/                                  writeup and figures
```

## Tuning notes

XGBoost used n_estimators=200, max_depth=4, and scale_pos_weight set to match the
class imbalance in each dataset. These are reasonable defaults with imbalance
correction, not the result of an exhaustive search. GCN and GraphSAGE used
hidden_channels=64, learning rate 0.005, 200-400 training epochs, and a softened
class weight (square root of the imbalance ratio rather than the raw ratio). That
softening wasn't arbitrary: an early attempt using the raw imbalance ratio directly
caused the model to collapse into predicting fraud for almost every node, which we
caught by inspecting precision and recall separately rather than trusting F1 alone.
Neither the tree-based nor the graph-based models went through a full hyperparameter
search. Results should be read with that in mind.

## Limitations

The synthetic generator is a controlled stress test, not a simulation of real UPI
traffic. The Barabási-Albert base graph and the fan-in injection give a setting where
the ground truth is known exactly, which is what makes the model comparison possible
in the first place. Real payment networks likely have structure this doesn't capture:
merchant categories, geography, time-of-day patterns. The Elliptic results are what
support that these findings generalize past this one synthetic setup.

Some synthetic runs gave the tabular-only baseline an amount-based shortcut it
shouldn't get credit for, purely by how the random amount ranges happened to land
relative to normal transactions. This is exactly why the seed-variance run and the
camouflaged sweep config exist: to check whether a result depended on that kind of
luck.

Elliptic's test window, timesteps 35 through 49, includes a documented dark market
shutdown event. Weber et al. note this degrades every model's performance in their
own experiments, which is a plausible part of why our GCN result sits below their
reported number even after we matched their architecture and training setup as
closely as we could.

GCN's neighbor-averaging is a bad fit for anomaly patterns where a node is defined
by looking different from its neighbors, not similar to them. This is a known
limitation of plain GCN for fraud and anomaly detection generally, not something
specific to this implementation.

## Future work

- Run the inductive test on Elliptic the same way it was done on the synthetic data.
- Add a harder sweep config that combines the smallest ring size, camouflage, and
  sparsity all at once.
- A full production system: Kafka for real-time transaction ingestion, GNN
  embeddings precomputed and served from Redis for sub-50ms scoring, periodic
  offline retraining. Deliberately left out of this project to keep the research
  question answerable in a few weeks.

## References

- Weber, M. et al. (2019). Anti-Money Laundering in Bitcoin: Experimenting with
  Graph Convolutional Networks for Financial Forensics. arXiv:1908.02591
- Kipf, T. & Welling, M. (2017). Semi-Supervised Classification with Graph
  Convolutional Networks.
- Hamilton, W., Ying, R., & Leskovec, J. (2017). Inductive Representation Learning
  on Large Graphs.
