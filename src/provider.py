from abc import ABC, abstractmethod

from numpy.random import randint
import pandas as pd

## DATAPROVIDER ################################################################
# DataProvider serves as data repository, saving and providing the historical data
# - context, selected action and observed reward 
class DataProvider(ABC):
    #
    def __init__(self, batchsize: int):
        self.contexts = []
        self.actions = []
        self.rewards = []
        self.defsize = batchsize
    #
    def size(self) -> int:
        return(len(self.contexts))
    #
    def __checksize__(self, size: int, lt: bool = False) -> None:
        if size == 0:
            raise ValueError("Parameter `size` must be nonzero")
        elif (size > self.size()) and not lt:
            raise ValueError(f"Value of parameter `size` is too large, max allowed: {self.size()}")
    #
    def collect(self, context: pd.DataFrame, action: int, reward: float) -> None:
        self.contexts.append(context)
        self.actions.append(action)
        self.rewards.append(reward)
    #
    @abstractmethod
    def provide(self, size: int) -> pd.DataFrame:
        pass


# BatchProvider provides last X observations
# - it is mostly useful for oracles capable of online learning
class BatchProvider(DataProvider):
    #
    def provide(self, size: int = None) -> pd.DataFrame:
        if size is None:
            size = self.defsize
        #
        self.__checksize__(size)
        if size < 0:
            out = (
                pd.concat(self.contexts)
                .assign(action = self.actions, reward = self.rewards)
            )
        else:
            start = self.size() - size
            out = (
                pd.concat(self.contexts[start: self.size()])
                .assign(
                    action = self.actions[start : self.size()],
                    reward = self.rewards[start : self.size()]
                )
            )
        return(out)


# SampleProvider provides sample (with replacement) of size X 
# - it is mostly useful for oracles incapable of online learning
# - it serves as a practical approximation of Thompson sampling
class SampleProvider(DataProvider):
    #
    def provide(self, size: int = None) -> pd.DataFrame:
        if size is None:
            size = self.defsize
        #
        self.__checksize__(size, True)
        if size < 0:
            size = self.size()
        ix = randint(self.size(), size = size)
        out = pd.concat([self.contexts[i] for i in ix])
        out = out.assign(
            action = [self.actions[i] for i in ix],
            reward = [self.rewards[i] for i in ix]
            )
        return(out)

