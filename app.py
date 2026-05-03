import solara
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import networkx as nx
from mesa.visualization import SolaraViz
from model import axl_model
from mesa.visualization.utils import update_counter

# ── Helper: cultural similarity between two agents ────────────────────────────
def similarity(a1, a2):
    # np.mean over a boolean array gives the proportion of True values,
    # producing Axelrod's (1997, p. 208) cultural similarity measure —
    # also the interaction probability used in model.step().
    return np.mean(a1.feature == a2.feature)
# Proportion of features that are identical — exactly Axelrod's (1997, p. 208)
    # cultural similarity measure; also the interaction probability used in model.step()

# This map integer node ids back to 2-D grid coordinates ────────────────
def grid_layout(graph, N):
    # node % N = column; N - 1 - node // N = row, inverted so node 0
    # sits top-left, matching the orientation of Axelrod's Figure 1.
    return {node: (node % N, N - 1 - node // N) for node in graph.nodes()}

# ── Component 1: spatial map of edge-level cultural similarity ────────────────
# @solara.component makes this reactive: Mesa re-calls it after every step
# so the map updates live without a manual refresh.
@solara.component
def CulturalSimilarityMap(model):
    update_counter.get()
      # subscribes this component to the step counter so it redraws each tick
    N = int(np.sqrt(model.no_agents))
    # recover grid side-length from total agent count
    graph = model.graph
    # the NetworkX grid graph shared with the model
    pos = grid_layout(graph, N)
    # One similarity score per edge; passed directly to nx.draw_networkx
    # so each edge is coloured by the cultural similarity of its endpoints.
    edge_colors = [
        similarity(model.agents_list[e[0]], model.agents_list[e[1]])
        for e in graph.edges()
    ]

    fig, ax = plt.subplots(figsize=(5, 5))
    # square figure to preserve the N×N grid aspect ratio
    nx.draw_networkx(
        graph, pos, ax=ax,
        node_size=18,
        node_color="#888888",
        edge_color=edge_colors,
        edge_cmap=plt.cm.RdBu,
        edge_vmin=0, edge_vmax=1,
        width=3.5,
        with_labels=False,
        # node id labels would clutter the grid
    )
     # nx.draw_networkx produces no colorbar automatically, so a ScalarMappable
    # is built manually with the same cmap; set_array([]) satisfies matplotlib
    # without binding data to it.
    sm = cm.ScalarMappable(cmap=plt.cm.RdBu, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Cultural similarity (0 = none, 1 = identical)", fontsize=8)
    ax.set_title("Map of Cultural Similarities", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    solara.FigureMatplotlib(fig)
    plt.close(fig)  # release memory; figures accumulate on every redraw without this

def DifferencePlot(model):
    update_counter.get()
    data = model.datacollector.get_model_vars_dataframe()
    fig, ax = plt.subplots(figsize=(5, 4))
    # Dividing by no_agents converts raw step counts to "events per site",
    # normalising time by territory size so runs on different grids are comparable
    # (Axelrod 1997, p. 219).
    steps = data.index / model.no_agents
    ax.plot(steps, data["Difference"], color="steelblue", label="Difference")
    ax.set_xlabel("Events per Site", fontsize=9)
    ax.set_ylabel("Total Cultural Difference\n(Σ differing features per edge)", fontsize=8)
    ax.set_title("Cultural Difference Over Time", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    solara.FigureMatplotlib(fig)
    plt.close(fig)

# Keys must match axl_model.__init__ argument names exactly so SolaraViz
# can pass updated values through when the user resets the model.
model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "N": {
        "type": "SliderInt",
        "value": 10, # default 10×10 = 100 agents, matching Axelrod's Table 2 baseline
        "label": "Grid size N (N × N agents)",
        "min": 5,
        "max": 20,
        "step": 1,
    },
    "features": {
        "type": "SliderInt",
        "value": 5, # F in the paper; more features → higher chance of shared traits → faster convergence
        "label": "Cultural features (F)",
        "min": 1,
        "max": 15,
        "step": 1,
    },
    "traits": {
        "type": "SliderInt",
        "value": 10,  # q in the paper; more traits → more possible cultures → more stable regions
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
