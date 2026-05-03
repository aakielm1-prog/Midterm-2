from mesa import Agent
import numpy as np

class axl_agent(Agent):
    '''Individual agent represented by a node in a network'''
    
    def __init__(self, unique_id, feat, trt, model):
        # ── Register with the Mesa model ─────────────────────────────────────
        # Mesa 3 changed the Agent constructor signature: unique_id is no longer
        # passed to super().__init__().  Calling super().__init__(model) correctly
        # registers this agent with the model's internal scheduler and agent list.
        super().__init__(model) 
        # Store the integer node id manually so the rest of the code (e.g. graph
        # neighbour look-ups in model.step) can still use agent.unique_id to index
        # into agents_list — preserving backward compatibility with Mesa 2 style.
        self.unique_id = unique_id
         # ── Initialise a random culture vector ────────────────────────────────
        # np.random.randint(low=0, high=trt, size=feat) draws `feat` independent
        # integers uniformly from {0, 1, …, trt-1}.  This mirrors Axelrod's Table 1
        # (p. 209) where every site starts with a randomly assigned culture, ensuring
        # that most neighbouring pairs initially share very few features (low similarity
        # → low probability of interaction → slow early convergence).
        # `low=0` is inclusive; `high=trt` is exclusive, so the range is exactly
        # the q possible traits Axelrod specifies.
        self.feature = np.random.randint(low=0, high=trt, size=feat)
    
    def step(self):
          # Axelrod's model is *asynchronous*: at each event the model picks one
        # active site at random and resolves its interaction before moving on
        # (Axelrod 1997, p. 209, footnote 5).  So, all the interaction logic
        # lives in model.step(), which selects agents and updates their features
        # directly.  This agent-level step() is intentionally left empty. The agents
        # do not activate themselves because they are passive recipients of the model's
        # random-selection mechanism.
        return