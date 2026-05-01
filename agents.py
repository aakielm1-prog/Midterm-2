from mesa import Agent
import numpy as np

class axl_agent(Agent):
    '''Individual agent represented by a node in a network'''
    
    def __init__(self, unique_id, feat, trt, model):
        super().__init__(model)        # Mesa 3: remove unique_id here
        self.unique_id = unique_id     # set it manually so the rest of the code still works
        self.feature = np.random.randint(low=0, high=trt, size=feat)
    
    def step(self):
        return