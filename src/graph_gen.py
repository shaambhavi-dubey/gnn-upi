
import networkx as nx
import random
import numpy as np
import pandas as pd
import pickle

random.seed(42)
np.random.seed(42)

# --- base graph ---
n_legit = 10000
n_attach = 3
Gr = nx.barabasi_albert_graph(n=n_legit, m=n_attach, seed=42)

DirGr = nx.DiGraph()
DirGr.add_nodes_from(Gr.nodes())
for u, v in Gr.edges():
    if random.random() < 0.5:
        DirGr.add_edge(u, v)
    else:
        DirGr.add_edge(v, u)

for u, v in DirGr.edges():
    amount = np.random.lognormal(mean=4, sigma=1.2)
    DirGr[u][v]["amount"] = round(amount, 2)

account_age_days = {n: random.randint(30, 2000) for n in DirGr.nodes()}
nx.set_node_attributes(DirGr, account_age_days, "account_age_days")
nx.set_node_attributes(DirGr, 0, "label")

# --- fraud ring injection ---
def inject_fraud_ring(DirGr, mule_node, source_nodes, min_amt, max_amt):
    for sorc in source_nodes:
        if sorc == mule_node:
            continue
        amt = round(random.uniform(min_amt, max_amt), 2)
        DirGr.add_edge(sorc, mule_node, amount=amt)
    DirGr.nodes[mule_node]["label"] = 1
    return DirGr

n_rings = 250
all_nodes = list(DirGr.nodes())
random.shuffle(all_nodes)
mule_nodes = all_nodes[:n_rings]
remaining_pool = all_nodes[n_rings:]

for mule in mule_nodes:
    sources_per_ring = random.randint(15, 40)
    sources = random.sample(remaining_pool, sources_per_ring)
    min_amt = random.uniform(5, 20)
    max_amt = min_amt + random.uniform(50, 200)
    DirGr = inject_fraud_ring(DirGr, mule, sources, min_amt, max_amt)

# --- save ---
with open("data/synthetic_graph.pkl", "wb") as f:
    pickle.dump(DirGr, f)

node_rows = []
for n in DirGr.nodes():
    node_rows.append({
        "node_id": n,
        "account_age_days": DirGr.nodes[n]["account_age_days"],
        "in_degree": DirGr.in_degree(n),
        "out_degree": DirGr.out_degree(n),
        "label": DirGr.nodes[n]["label"]
    })
pd.DataFrame(node_rows).to_csv("data/node_features.csv", index=False)
