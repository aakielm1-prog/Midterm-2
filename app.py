import solara
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import networkx as nx
from mesa.visualization import SolaraViz, make_plot_component
from model import axl_model
from mesa.visualization.utils import update_counter


def similarity(a1, a2):
    return np.mean(a1.feature == a2.feature)

def grid_layout(graph, N):
    return {node: (node % N, N - 1 - node // N) for node in graph.nodes()}


@solara.component
def CulturalSimilarityMap(model):
    update_counter.get()
    N = int(np.sqrt(model.no_agents))
    graph = model.graph
    pos = grid_layout(graph, N)

    edge_colors = [
        similarity(model.agents_list[e[0]], model.agents_list[e[1]])
        for e in graph.edges()
    ]

    fig, ax = plt.subplots(figsize=(5, 5))
    nx.draw_networkx(
        graph, pos, ax=ax,
        node_size=18,
        node_color="#888888",
        edge_color=edge_colors,
        edge_cmap=plt.cm.RdBu,
        edge_vmin=0, edge_vmax=1,
        width=3.5,
        with_labels=False,
    )
    sm = cm.ScalarMappable(cmap=plt.cm.RdBu, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Cultural similarity (0 = none, 1 = identical)", fontsize=8)
    ax.set_title("Map of Cultural Similarities", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    solara.FigureMatplotlib(fig)
    plt.close(fig)


DifferencePlot = make_plot_component({"Difference": "steelblue"})

model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "N": {
        "type": "SliderInt",
        "value": 10,
        "label": "Grid size N (N × N agents)",
        "min": 5,
        "max": 20,
        "step": 1,
    },
    "features": {
        "type": "SliderInt",
        "value": 5,
        "label": "Cultural features (F)",
        "min": 1,
        "max": 15,
        "step": 1,
    },
    "traits": {
        "type": "SliderInt",
        "value": 10,
        "label": "Traits per feature (q)",
        "min": 2,
        "max": 20,
        "step": 1,
    },
}

axelrod_model = axl_model()

page = SolaraViz(
    axelrod_model,
    components=[CulturalSimilarityMap, DifferencePlot],
    model_params=model_params,
    name="Axelrod (1997) – Dissemination of Culture",
)

page