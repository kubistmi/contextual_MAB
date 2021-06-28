from abc import ABC, abstractmethod

from numpy.random import choice
import pandas as pd

from src.provider import DataProvider
from src.oracle import Oracle
from src.policy import Policy

## AGENT #######################################################################
# Agent is wrapper around DataProvider, Oracle and Policy
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
    #
    def replay(self) -> None:
        history = self.provider.provide(self.provider.size())
        self.oracle.fit(history)


## ENVIRONMENT #################################################################
# Environment handles the provision of context and rewards based on specified data
class Environment(ABC):
    #
    @abstractmethod
    def get_context(self) -> pd.DataFrame:
        pass
    #
    @abstractmethod
    def evaluate(self, action: int) -> None:
        pass


# ChurnEnvironment is implementation of Environment for the Telco-customer-churn dataset
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


# LEARN ########################################################################
# Learn function handles the iterative learning of the Agent using the Environment
def learn(agent: Agent, env: Environment, iters: int, update_freq: int, replay_freq : int = None) -> Agent:
    replay = True
    if replay_freq is None:
        replay = False
    #
    for i in range(iters):
        if i > 0:
            if replay and (i % replay_freq == 0):
                agent.replay()
            elif i % update_freq == 0:
                agent.update()
        cx = env.get_context()
        act = agent.act(cx, i)
        rew = env.evaluate(act)
        agent.save_iter(cx, act, rew)
    return(agent)