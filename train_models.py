from datetime import datetime
import os
import pickle

import mlflow as mf
import numpy as np
import pandas as pd

from interface import  AdaGreedPolicy, Agent, BatchProvider, SampleProvider, ChurnEnvironment, OnRegOracle, SampleProvider, learn

ITER = 8000

data = pd.read_csv("data/cleaned.csv")
data = data.assign(action = np.where(data.churn == "No", 0, 1))
data = data.drop(["customerid", "churn"], axis = 1)
data = pd.get_dummies(data, drop_first = True)

rewards = pd.DataFrame({
    "act":  [0,    0,    1,    1],
    "pred": [0,    1,    0,    1],
    "val":  [1, -1.5,   -2,  0.5]
    #"val":  [0,  -50, -350,   30]
})

ada_thresh = np.array([0.8, 0.9, 0.99])
ada_decay = np.array([0.9, 0.99, 0.999])
retrains = np.array([100, 200])

# RegTree specific
# sample_size = np.array([300, 500])
# depths = np.array([3, 5, 7, 9])
# prod = pd.core.reshape.util.cartesian_product([ada_decay, ada_thresh, depths, sample_size, retrains])
# param_grid = pd.DataFrame({
#     "thresh": prod[0],
#     "decay": prod[1],
#     "depth": prod[2],
#     "size": prod[3],
#     "retrain": prod[4]
# })

prod = pd.core.reshape.util.cartesian_product([ada_decay, ada_thresh, retrains])
param_grid = pd.DataFrame({
    "thresh": prod[0],
    "decay": prod[1],
    "retrain": prod[2]
})
param_grid = param_grid.assign(size = param_grid.retrain)

env = ChurnEnvironment(data, rewards)

def evaluate(x):
    p = x.to_dict()
    with mf.start_run():
        mf.log_param("oracle", "OnReg")
        mf.log_param("policy", "AdaGreed")
        mf.log_param("provider", "batch")
        mf.log_params(p)
        agent = Agent(BatchProvider(int(p["size"])), OnRegOracle([0,1], -2), AdaGreedPolicy(thresh = p["thresh"], decay = p["decay"]))
        agent = learn(agent, env, iters = ITER, update_freq = p["retrain"])
        rew = pd.Series(agent.provider.rewards)
        act = pd.Series(agent.provider.actions)
        #
        # dirname = f"models/SGD_{p['depth']}_{p['size']}_{p['retrain']}_{p['thresh']}_{p['decay']}/"
        dirname = f"models/SGD_{p['size']}_{p['retrain']}_{p['thresh']}_{p['decay']}/"
        os.makedirs(dirname)
        rew.to_csv(dirname + "rewards.csv", index = False)
        act.to_csv(dirname + "actions.csv", index = False)
        pickle.dump(agent, open(dirname + "agent.pkl", "wb"))
        #
        mf.log_metric("mean_reward", rew.mean())
        mf.log_metric("mean_action", act.mean())
        mf.log_artifact(dirname)

param_grid.apply(evaluate, axis=1)