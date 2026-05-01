import matplotlib.pyplot as plt
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import NetworkGrid
from agents import axl_agent

import numpy as np
import networkx as nx
import random
from tqdm.notebook import tqdm

def f_difference(model):
    diff = 0
    conn = nx.edges(model.graph)
    for edge in conn:
        diff += np.mean(abs(model.agents_list[edge[0]].feature - model.agents_list[edge[1]].feature))
    return diff

class axl_model(Model):
    def __init__(self, N=10, features=5, traits=10, seed=None, Graph=None):
        super().__init__(seed=seed)
        # Build graph internally if not provided (needed for SolaraViz)
        if Graph is None:
            Graph = nx.grid_2d_graph(N, N)
            Graph = nx.relabel_nodes(Graph, dict(zip(Graph.nodes, range(Graph.number_of_nodes()))))
            self.no_agents = N * N
        else:
            self.no_agents = N  # old calling convention passes total agents as N
        self.graph = Graph
        self.no_features = features
        self.no_traits = traits
        self.G = NetworkGrid(Graph)
        self.agents_list = []
        
        for i in range(self.no_agents):
            a = axl_agent(i, features, traits, self)
            self.agents_list.append(a)
        
        self.datacollector = DataCollector(model_reporters={"Difference": f_difference})
    
    def step(self):
        self.datacollector.collect(self)
        agent = random.choice(self.agents_list)
        neighbors = list(self.graph.neighbors(agent.unique_id))
        if not neighbors:
            return
        neigh = self.agents_list[random.choice(neighbors)]
        prob = np.count_nonzero(agent.feature == neigh.feature) / self.no_features
        diff_indices = np.nonzero(agent.feature - neigh.feature)[0]
        if len(diff_indices) == 0:
            return
        if np.random.rand() < prob:
            index = np.random.choice(diff_indices)
            agent.feature[index] = neigh.feature[index]

if __name__ == "__main__":
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

    def grid_layout(g):
        pos = {}
        L = np.sqrt(g.number_of_nodes())
        for n in g.nodes():
            pos[n] = [int(n/L), n%L]
        return pos

    features = 1
    traits = 5
    N = 20
    P = 0.1
    time = 10000
    ws = {}
    graph = nx.grid_2d_graph(N, N)
    graph = nx.relabel_nodes(graph, dict(zip(graph.nodes, range(graph.number_of_nodes()))))
    model = axl_model(N=N**2, Graph=graph, features=features, traits=traits)
    for i in tqdm(range(time)):
        model.step()
        if i % 1000 == 0:
            ws[i] = []
            for j, agent in enumerate(model.agents_list):
                ws[i].append(agent.feature[0])

    data = model.datacollector.get_model_vars_dataframe()
    data.plot()
    pos = grid_layout(graph)
    fig = plt.figure(figsize=(20, 8))

    for i, k in enumerate(ws.keys()):
        plt.subplot(2, 5, i+1)
        plt.title('Time: ' + str(i*1000))
        nx.draw_networkx_nodes(graph, pos, node_size=10, node_color=[ws[k][x] for x in graph.nodes()])

    GCS = []
    for t in ws:
        gcs = []
        for attr in np.unique(ws[t]):
            attr_subnodes = [i for i, k in enumerate(ws[t]) if k == attr]
            attr_subgraph = nx.subgraph(graph, attr_subnodes)
            gcs.append(len(list(nx.components.connected_components(attr_subgraph))[0]))
        GCS.append(np.max(gcs))
    plt.plot(GCS)
    plt.show()