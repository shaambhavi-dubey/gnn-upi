# finally we are going to create a more complex synthetic graph and see ow the models perform on this new graph, generated ourselves and also labelled 
# we will make 3 new graphs -> easy,medium and hard which basically ranges number of sources and changes the fraud rings and run all 4 onthe 3 graphs
# this is the robustness sweep
def generate_synthetic_graph(n_legit=10000, n_rings=250,
                               sources_range=(15, 40),
                               fraud_amt_range=None,   # None = original tight range; or overlap with legit
                               seed=42):
    random.seed(seed)
    np.random.seed(seed)

    Gr = nx.barabasi_albert_graph(n=n_legit, m=3, seed=seed)
    DirGr = nx.DiGraph()
    DirGr.add_nodes_from(Gr.nodes())
    for u, v in Gr.edges():
        if random.random() < 0.5:
          DirGr.add_edge(u, v)
        else:
            DirGr.add_edge(v, u)

    for u, v in DirGr.edges():
        amount = np.random.lognormal(mean=4, sigma=1.2)
        DirGr[u][v]['amount'] = round(amount, 2)

    account_age_days = {n: random.randint(30, 2000) for n in DirGr.nodes()}
    nx.set_node_attributes(DirGr, account_age_days, 'account_age_days')
    nx.set_node_attributes(DirGr, 0, 'label')

    all_nodes = list(DirGr.nodes())
    random.shuffle(all_nodes)
    mule_nodes = all_nodes[:n_rings]
    remaining_pool = all_nodes[n_rings:]

    for mule in mule_nodes:
        n_sources = random.randint(*sources_range)
        sources = random.sample(remaining_pool, n_sources)

        if fraud_amt_range == "camouflaged":
            # draw fraud amounts from the SAME lognormal distribution as legit transactions
            for src in sources:
                if src == mule:
                    continue
                amt = round(np.random.lognormal(mean=4, sigma=1.2), 2)
                DirGr.add_edge(src, mule, amount=amt)
        else:
            min_amt = random.uniform(5, 20)
            max_amt = min_amt + random.uniform(50, 200)
            for src in sources:
                if src == mule:
                    continue
                  amt = round(random.uniform(min_amt, max_amt), 2)
                DirGr.add_edge(src, mule, amount=amt)

        DirGr.nodes[mule]['label'] = 1

    return DirGr
configs = {
    "easy (original)": {"sources_range": (15, 40), "fraud_amt_range": None},
    "medium (smaller rings)": {"sources_range": (8, 20), "fraud_amt_range": None},
    "hard (camouflaged amounts)": {"sources_range": (8, 20), "fraud_amt_range": "camouflaged"},
}
