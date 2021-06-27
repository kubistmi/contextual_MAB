import os
import pickle

import mlflow as mf
import numpy as np
import pandas as pd

from src.policy import AdaGreedPolicy, EpsGreedPolicy
from src.provider import BatchProvider, SampleProvider
from src.oracle import NeuralOracle, OnRegOracle, RegTreeOracle
from src.domain import Agent, ChurnEnvironment, learn

ITER = 8000

data = pd.read_csv("data/cleaned.csv")
data = data.assign(action = np.where(data.churn == "No", 0, 1))
data = data.drop(["customerid", "churn"], axis = 1)
data = pd.get_dummies(data, drop_first = True)

rewards = pd.DataFrame({
    "act":  [0,    0,    1,    1],
    "pred": [0,    1,    0,    1],
    "val":  [1, -1.5,   -2,  0.5]
})

# parameters
ada_thresh = np.array([0.8, 0.9, 0.99])
ada_decay = np.array([0.9, 0.99, 0.999])
eps_thresh = np.array([0.1, 0.2, 0.3])
eps_decay = np.array([0])
retrains = np.array([100, 200])

# OnReg specific ###############################################################
# prod = pd.core.reshape.util.cartesian_product([eps_thresh, eps_decay, retrains])
# param_grid = pd.DataFrame({
#     "thresh": prod[0],
#     "decay": prod[1],
#     "retrain": prod[2]
# })
# param_grid = param_grid.assign(size = param_grid.retrain)

# RegTree specific #############################################################
sample_size = np.array([300, 500])
depths = np.array([3, 5, 7, 9])
prod = pd.core.reshape.util.cartesian_product([eps_thresh, eps_decay, depths, sample_size, retrains])
param_grid = pd.DataFrame({
    "thresh": prod[0],
    "decay": prod[1],
    "depth": prod[2],
    "size": prod[3],
    "retrain": prod[4]
})

# Neural specific ##############################################################

env = ChurnEnvironment(data, rewards)

def evaluate(x):
    p = x.to_dict()
    with mf.start_run():
        mf.log_param("oracle", "RegTree")
        mf.log_param("policy", "EpsGreed")
        mf.log_param("provider", "sample")
        mf.log_params(p)
        agent = Agent(SampleProvider(int(p["size"])), RegTreeOracle([0,1], -2, p["depth"]), EpsGreedPolicy(thresh = p["thresh"]))#, decay = p["decay"]))
        agent = learn(agent, env, iters = ITER, update_freq = p["retrain"])
        rew = pd.Series(agent.provider.rewards)
        act = pd.Series(agent.provider.actions)
        #
        dirname = f"models/tree_Eps_{p['depth']}_{p['size']}_{p['retrain']}_{p['thresh']}_{p['decay']}/"
        # dirname = f"models/SGD_Eps_{p['size']}_{p['retrain']}_{p['thresh']}_{p['decay']}/"
        os.makedirs(dirname)
        rew.to_csv(dirname + "rewards.csv", index = False)
        act.to_csv(dirname + "actions.csv", index = False)
        pickle.dump(agent, open(dirname + "agent.pkl", "wb"))
        #
        mf.log_metric("mean_reward", rew.mean())
        mf.log_metric("mean_action", act.mean())
        mf.log_artifact(dirname)

param_grid.apply(evaluate, axis=1)