from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import pandas as pd

def softmax(x: pd.Series) -> pd.Series:
    e_x = np.exp(x - x.max())
    return e_x / e_x.sum()

## POLICY ######################################################################
# Policy defines the tactic used to handle the exploration-exploitation tradeoff
class Policy(ABC):
    #
    @abstractmethod
    def decide(self, rewards: pd.Series, time: int) -> int:
        pass


# EpsGreedPolicy utilises simple (non-decaying) routine
# - selects random action with probability = threshold
# - the best action (highest predicted reward) otherwise
class EpsGreedPolicy(Policy):
    #
    def __init__(self, thresh: int):
        self.thresh = thresh
    #
    def decide(self, rewards: pd.Series, time: int) -> int:
        limit = np.random.random()
        if rewards.unique().shape[0] == 1:
            action = rewards.reset_index()["index"].sample(1).iat[0]        
        elif limit > self.thresh:
            action = rewards.argmax()
        else:
            action = rewards.reset_index()["index"].sample(1).iat[0]
        return(action)


# AdaGreedPolicy utilises time-decaying adaptive routine
# - selects random action if the maximum predicted reward is lower than threshold
# - the best action (highest predicted reward) otherwise
# - the threshold decreases over time (iterations)
# - can utilise softmax to map real-values predictions into probability distribution
class AdaGreedPolicy(Policy):
    #
    def __init__(self, thresh: int, decay: int, soft: bool = True):
        self.inthresh = thresh
        self.thresh = thresh
        self.decay = decay
        self.soft = soft
    #
    def set_params(self, **parameters) -> None:
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
    #
    def get_params(self) -> Dict[str,int]:
        return {"decay": self.decay, "thresh": self.thresh}
    #
    def decide(self, rewards: pd.Series, time: int) -> int:
        if self.soft:
            rewards = softmax(rewards)
        #
        maxR = rewards.max()
        if rewards.unique().shape[0] == 1:
            action = rewards.reset_index()["index"].sample(1).iat[0]
        elif maxR > self.thresh:
            action = rewards.argmax()
        else:
            action = rewards.reset_index()["index"].sample(1).iat[0]
        self.thresh = self.inthresh * self.decay ** time
        return(action)