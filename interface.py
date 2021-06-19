from abc import ABC, abstractmethod
from typing import List

import pandas as pd


###############################################################################
# Data provider                                                               #
###############################################################################
class DataProvider(ABC):
    
    def __init__(self):
        super().__init__()
        self.contexts = [pd.DataFrame()]
        self.actions = [int]
        self.rewards = [float]
    
    def collect(self, context: pd.DataFrame, action: int, reward: float) -> None:
        self.contexts.append(context)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def size(self) -> int:
        return(len(self.contexts))
    
    @abstractmethod
    def provide(self, size: int) -> pd.DataFrame:
        pass


class BatchProvider(DataProvider):
    
    def provide(self, size: int) -> pd.DataFrame:
        if size == 0:
            raise ValueError("Parameter `size` must be nonzero")
        elif size < 0:
            out = pd.concat(self.contexts)
            out = out.assign(action = self.actions, reward = self.rewards)
        elif size >= self.size():
            raise ValueError(f"Value of parameter `size` is too large, max allowed: {self.size()}")
        else:
            start = self.size() - size
            out = pd.concat(
                self.contexts[start: self.size()]
                )
            out = out.assign(
                action = self.actions[start : self.size()],
                reward = self.rewards[start : self.size()]
                )
        return(out)


###############################################################################
# Model                                                                       #
###############################################################################
class Model(ABC):

    def __init__(self, actions: int):
        super().__init__()
        self.actions = actions
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame):
        pass


###############################################################################
# Agent                                                                       #
###############################################################################
class Agent(ABC):
    
    def __init__(self, provider: DataProvider, model: Model):
        super().__init__()
        self.provider = provider
        self.model = model

    @abstractmethod
    def decide(self) -> None:
        pass

    @abstractmethod
    def update(self) -> None:
        pass


###############################################################################
# Environment                                                                 #
###############################################################################
class Environment(ABC):

    def __init__(self):
        super.__init__()
            
    @abstractmethod
    def evaluate(self) -> None:
        pass

###############################################################################
# Learn                                                                       #
###############################################################################
def learn(a: Agent, e: Environment) -> None:
    pass
