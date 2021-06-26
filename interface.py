from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
from numpy.random import randint, choice
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

###############################################################################
# Data provider                                                               #
###############################################################################
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


###############################################################################
# Oracle                                                                      #
###############################################################################
class Oracle(ABC):
    #
    def __init__(self, actions: List[int]):
        self.actions = actions
        self.oracles = {a : LinearRegression() for a in actions}
    #
    def fit(self, X: pd.DataFrame) -> None:
        for i in self.actions:
            self.__fit_oracle__(i, X.query("action == @i"))
    #
    def predict(self, X: pd.DataFrame) -> pd.Series:
        out = {
            i: self.__predict_oracle__(i, X)
            for i in self.actions
        }
        return(pd.Series(out))
        
    #
    def __check_fitted__(self, oracle: int) -> bool:
        try:
            check_is_fitted(self.oracles[oracle])
            return(True)
        except NotFittedError:
            return(False)
    #
    @abstractmethod
    def __fit_oracle__(self, oracle: int, X: pd.DataFrame):
        pass
    #
    @abstractmethod
    def __predict_oracle__(self, oracle:int, X: pd.DataFrame):
        pass


class LinRegOracle(Oracle):
    #
    def __fit_oracle__(self, oracle: int, X: pd.DataFrame) -> None:
        if X.shape[0] == 0:
            return 
        y = X[["reward"]]
        X = X.drop(["reward","action"], axis = 1)
        self.oracles[oracle] = LinearRegression().fit(X,y)
    #
    def __predict_oracle__(self, oracle:int, X: pd.DataFrame) -> int:
        if not self.__check_fitted__(oracle):
            return 0
        return(self.oracles[oracle].predict(X)[0][0])


###############################################################################
# Policy                                                                      #
###############################################################################
class Policy(ABC):
    #
    @abstractmethod
    def set_params(self) -> None:
        pass
    #
    @abstractmethod
    def get_params(self) -> Dict[str,int]:
        pass
    #
    @abstractmethod
    def decide(self, rewards: pd.Series, time: int) -> int:
        pass


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
        if maxR > self.thresh:
            action = rewards.argmax()
        else:
            action = rewards.reset_index()["index"].sample(1).iat[0]
        self.thresh = self.inthresh * self.decay ** time
        return(action)


###############################################################################
# Agent                                                                       #
###############################################################################
class Agent(ABC):
    #    
    def __init__(self, provider: DataProvider, oracle: Oracle, policy: Policy):
        super().__init__()
        self.provider = provider
        self.oracle = oracle
        self.policy = policy
    #
    def act(self, X: pd.DataFrame, time: int) -> int:
        pred = self.oracle.predict(X)
        return(self.policy.decide(pred, time))
    #
    def save_iter(self, context: pd.DataFrame, action:int, reward: int) -> None:
        self.provider.collect(context, action, reward)
    #
    def update(self) -> None:
        history = self.provider.provide()
        self.oracle.fit(history)


###############################################################################
# Environment                                                                 #
###############################################################################
class Environment(ABC):
    #
    @abstractmethod
    def get_context(self) -> pd.DataFrame:
        pass
    #
    @abstractmethod
    def evaluate(self, action: int) -> None:
        pass

class ChurnEnvironment(Environment):
    #
    def __init__(self, data: pd.DataFrame, rewards: pd.DataFrame, balanced = True):
        #
        # validate data
        if not (data.columns == "action").any():
            raise AttributeError("Column 'action' not found in the DataFrame")
        self.data = data
        #
        # validate rewards
        cols = pd.Series(["act","pred","val"])
        if not cols.isin(rewards.columns).all():
            raise AttributeError(
                f"Table `rewards` has wrong colnames, expected: {cols.values}, got: {rewards.columns.values}"
                )
        self.rewards = rewards
        #
        # finish the setup  
        self.balanced = balanced
        self.index = 0
        self.actions = data.action.unique()
    #
    def get_context(self) -> pd.DataFrame:
        if self.balanced:
            act = choice(self.actions)
            out = self.data.loc[self.data.action == act].sample(1)
        else:
            out = self.data.sample(1)
        self.index = out.index[0]
        return(out.drop(["action"], axis = 1))
    #
    def evaluate(self, action: int) -> int: 
        obs = self.data.loc[self.index]
        out = self.rewards.loc[
            (self.rewards.act == obs.action) & (self.rewards.pred == action)
            ]
        return(out.val.iat[0])

###############################################################################
# Learn                                                                       #
###############################################################################
def learn(agent: Agent, env: Environment, iters: int, update_freq: int) -> None:
    for i in range(iters):
        if i > 0 and i % update_freq == 0:
            agent.update()
        cx = env.get_context()
        act = agent.act(cx, i)
        rew = env.evaluate(act)
        agent.save_iter(cx, act, rew)
    return(agent)

def softmax(x: pd.Series) -> pd.Series:
    e_x = np.exp(x - x.max())
    return e_x / e_x.sum()