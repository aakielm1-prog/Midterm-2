import matplotlib.pyplot as plt
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import NetworkGrid
from agents import axl_agent

import numpy as np
import networkx as nx
import random
from tqdm.notebook import tqdm

# ── Model-level reporter ──────────────────────────────────────────────────────
# Axelrod measures cultural heterogeneity by counting how many features differ
# across neighbouring pairs.  Because traits are *categorical* (a trait value of
# 3 is not "closer to" 4 than to 9), the right comparison is a binary mismatch:
# two sites either share a feature or they do not.
# This function sums, over every edge in the graph, the number of features on
# which the two endpoint agents differ.  np.sum(a != b) counts the mismatches
# in one feature vector comparison without treating the magnitude of any
# difference as meaningful — consistent with Axelrod's similarity definition
# (1997, p. 208: similarity = proportion of features with *identical* traits).
# A return value of 0 means every neighbouring pair is culturally identical or
# completely isolated (the two absorbing conditions), so no further change is
# possible.  Higher values indicate more cultural diversity remains in the
# territory.

def f_difference(model):
    diff = 0
    conn = nx.edges(model.graph)
    for edge in conn:
        diff += np.sum(model.agents_list[edge[0]].feature != model.agents_list[edge[1]].feature)
    return diff

class axl_model(Model):
    def __init__(self, N=10, features=5, traits=10, seed=None, Graph=None):
        super().__init__(seed=seed)
         # ── Build the geographic grid ─────────────────────────────────────────
        # Axelrod places agents on a 2-D lattice where each interior site has
        # exactly four cardinal neighbors (N/E/S/W).  NetworkX's, which I imported, grid_2d_graph
        # produces this topology; nodes are then relabelled 0…N²-1 so they can
        # serve as integer indices into agents_list.
        # If a custom graph is passed (e.g. from the __main__ experiments) it is
        # used directly, with N interpreted as the total agent count rather than
        # the side-length.
        if Graph is None:
            Graph = nx.grid_2d_graph(N, N)
            Graph = nx.relabel_nodes(Graph, dict(zip(Graph.nodes, range(Graph.number_of_nodes()))))
            self.no_agents = N * N
        else:
            self.no_agents = N  # old calling convention passes total agents as N
        self.graph = Graph
         # ── Store cultural parameters ─────────────────────────────────────────
        # F (features) and q (traits) are the two key cultural-complexity
        # parameters in Axelrod's Table 2.  More features cause easier convergence;
        # more traits cause harder convergence (more possible cultures, lower chance
        # two neighbors share any given feature).
        self.no_features = features
        self.no_traits = traits
         # This wraps the NetworkX graph in Mesa's NetworkGrid so the framework can
        # place and retrieve agents by node id (not used for movement since
        # Axelrod's model has fixed sites, but required for Mesa compatibility).
        self.G = NetworkGrid(Graph)
         # ── Initialise agents with random cultures ────────────────────────────
        # Each site receives a random culture vector of length F drawn uniformly
        # from {0, …, q-1}.  This mirrors Table 1 in the Axelrod's paper: cultures are
        # assigned at random so that most neighboring pairs initially share few
        # or no features (low cultural similarity makes it unlikely to interact).
        self.agents_list = []
        
        for i in range(self.no_agents):
            a = axl_agent(i, features, traits, self)
            self.agents_list.append(a)
          # ── Data collection ───────────────────────────────────────────────────
        # Records the f_difference metric at every step so we can plot how
        # cultural difference evolves over simulated time (events per site).
        self.datacollector = DataCollector(model_reporters={"Difference": f_difference})
    
    # ── Core simulation step (one "event" in Axelrod's terminology) ───────────
    def step(self):
        # Collect the current cultural-difference metric before applying change,
        # giving us a snapshot of the landscape at this point in time.
        self.datacollector.collect(self)
          # Step 1 (reflects Axelrod p. 208): choose a random active site and one of its
        # neighbors.  Using the neighbor list from the graph respects the local
        # interaction constraint — agents can only influence their immediate
        # geographic neighbors, not the whole population.
        agent = random.choice(self.agents_list)
        neighbors = list(self.graph.neighbors(agent.unique_id))
        if not neighbors:
            return
        neigh = self.agents_list[random.choice(neighbors)]
        
         # Step 2a is to compute cultural similarity (proportion of shared features).
        # This is the interaction probability: the more features two sites already
        # share, the more likely they are to interact and become even more similar
        # ("similarity breeds interaction, interaction breeds similarity").
        prob = np.count_nonzero(agent.feature == neigh.feature) / self.no_features
         # Identify the features on which the two sites currently differ. These
        # are the candidates for cultural transmission if interaction occurs.
        diff_indices = np.nonzero(agent.feature - neigh.feature)[0]

         # If the pair is already identical there is nothing to transmit; skip.
        # This is one of the two absorbing conditions: identical neighbors can
        # interact but produce no change.
        if len(diff_indices) == 0:
            return
        
          # Step 2b – interact with probability equal to cultural similarity.
        # If the pair shares no features (prob = 0) they never interact — the
        # second absorbing condition that locks in stable cultural boundaries.
        # If they do interact, one randomly chosen differing feature of the
        # active site is overwritten with the neighbor's trait, nudging the two
        # sites one step closer to full cultural agreement.
        if np.random.rand() < prob:
            index = np.random.choice(diff_indices)
            agent.feature[index] = neigh.feature[index]

if __name__ == "__main__":
     # ── Experiment 1: random (Erdős–Rényi) network ───────────────────────────
    # Tests the model on a non-lattice topology to explore how the range of
    # interaction (here determined by edge probability P) affects convergence.
    # Axelrod notes that larger neighborhoods produce fewer stable regions.
    features = 5
    traits = 6
    N = 100
    P = 0.1
    time = 3000
    graph = nx.fast_gnp_random_graph(N, P)

    model = axl_model(N=N, Graph=graph, features=features, traits=traits)
    for i in range(time):
        model.step()
    data = model.datacollector.get_model_vars_dataframe()
    data.plot()

# ── Helper: grid layout for visualisation ──────
    # Maps integer node ids back to (row, col) positions so NetworkX draws the
    # graph as a spatial 2-D grid, matching Axelrod's Figure 1 presentation.
    def grid_layout(g):
        pos = {}
        L = np.sqrt(g.number_of_nodes())
        for n in g.nodes():
            pos[n] = [int(n/L), n%L]
        return pos

 # ── Experiment 2: 2-D lattice with a single cultural feature ─────────────
    # With F=1, each site's entire culture is a single digit (one of q traits).
    # This makes cultural regions easy to visualize as color maps and
    # replicates the snapshots in Axelrod's Figure 1: distinct homogeneous
    # patches emerge and then consolidate over time.
    features = 1
    traits = 5
    N = 20
    P = 0.1
    time = 10000
    ws = {}

     # Build the canonical N×N lattice and relabel nodes to integer indices.
    graph = nx.grid_2d_graph(N, N)
    graph = nx.relabel_nodes(graph, dict(zip(graph.nodes, range(graph.number_of_nodes()))))
    
     # N is passed as N² because a custom graph is provided; the model treats N
    # as the total number of agents when Graph is not None (see __init__).
    model = axl_model(N=N**2, Graph=graph, features=features, traits=traits)
    for i in tqdm(range(time)):
        model.step()
        # Record a spatial snapshot every 1000 events to track how cultural
        # regions grow and consolidate — analogous to the panels in Figure 1.
        if i % 1000 == 0:
            ws[i] = []
            for j, agent in enumerate(model.agents_list):
                ws[i].append(agent.feature[0])

    data = model.datacollector.get_model_vars_dataframe()
    data.plot()
    pos = grid_layout(graph)

    # ── Visualise cultural snapshots over time ────────────────────────────────
    # Each subplot colors nodes by their single-feature trait value, showing
    # how homogeneous cultural regions emerge and enlarge as the simulation
    # progresses (local convergence → fewer, larger regions).
    fig = plt.figure(figsize=(20, 8))

    for i, k in enumerate(ws.keys()):
        plt.subplot(2, 5, i+1)
        plt.title('Time: ' + str(i*1000))
        nx.draw_networkx_nodes(graph, pos, node_size=10, node_color=[ws[k][x] for x in graph.nodes()])

# ── Track the Largest Cultural Region over time ───────────────────────────
    # For each snapshot, find the largest connected component that shares the
    # same trait value.  Watching this value grow reflects the consolidation
    # dynamic: large regions tend to "eat" smaller dialects (Axelrod p. 216),
    # eventually dominating the territory.
    GCS = []
    for t in ws:
        gcs = []
        for attr in np.unique(ws[t]):
              # Subgraph containing only sites with trait == attr
            attr_subnodes = [i for i, k in enumerate(ws[t]) if k == attr]
            attr_subgraph = nx.subgraph(graph, attr_subnodes)
            # Size of the largest connected component for this trait value
            gcs.append(len(list(nx.components.connected_components(attr_subgraph))[0]))
        GCS.append(np.max(gcs))
    plt.plot(GCS)
    plt.show()