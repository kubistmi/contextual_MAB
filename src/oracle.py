from abc import ABC, abstractmethod
from typing import List

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

import tensorflow as tf
from tensorflow.keras.models import clone_model
from tensorflow.keras import layers

## ORACLE ######################################################################
# Oracle is a wrapper around set of models used to predict reward based on context
class Oracle(ABC):
    #
    def __init__(self, actions: List[int], min_reward: int):
        self.minrew = min_reward
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


# LinRegOracle utilises linear regression model from sklearn
# - offline mode only, so it should be used ONLY with SampleProvider
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
            return self.minrew
        return(self.oracles[oracle].predict(X)[0][0])

# RegTreeOracle utilises regression tree model from sklearn
# - offline mode only, so it should be used ONLY with SampleProvider
class RegTreeOracle(Oracle):
    #
    def __init__(self, actions: List[int], min_reward:int, depth: int = None):
        super().__init__(actions, min_reward)
        self.depth = depth
    #
    def __fit_oracle__(self, oracle: int, X: pd.DataFrame) -> None:
        if X.shape[0] == 0:
            return 
        y = X[["reward"]]
        X = X.drop(["reward","action"], axis = 1)
        self.oracles[oracle] = DecisionTreeRegressor(max_depth = self.depth).fit(X,y)
    #
    def __predict_oracle__(self, oracle:int, X: pd.DataFrame) -> int:
        if not self.__check_fitted__(oracle):
            return self.minrew
        return(self.oracles[oracle].predict(X)[0])

# OnRegOracle utilises online (SGD based) linear regression model from sklearn
# - trained using partial_fit, therefore allows for online method of training
# - intended to be used with BatchProvider 
class OnRegOracle(Oracle):
    #
    def __init__(self, actions: List[int], min_reward:int):
        self.minrew = min_reward
        self.actions = actions
        self.scalers = {a : StandardScaler() for a in actions}
        self.oracles = {a : SGDRegressor() for a in actions}
    #
    def __fit_oracle__(self, oracle: int, X: pd.DataFrame) -> None:
        if X.shape[0] == 0:
            return 
        y = X["reward"]
        X =  X.drop(["reward","action"], axis = 1)
        self.scalers[oracle] = self.scalers[oracle].partial_fit(X)
        X = self.scalers[oracle].transform(X)
        self.oracles[oracle] = self.oracles[oracle].partial_fit(X,y)
    #
    def __predict_oracle__(self, oracle:int, X: pd.DataFrame) -> int:
        if not self.__check_fitted__(oracle):
            return self.minrew
        X = self.scalers[oracle].transform(X)
        return(self.oracles[oracle].predict(X)[0])

# NeuralOracle wraps around sequential neural net model from tensorflow
# - Keras fitting regime allows for online method of training
# - intended to be used with BatchProvider
class NeuralOracle(Oracle):
    #
    def __init__(self, actions: List[int], min_reward:int, base_model: tf.keras.Sequential, batch: int,  epochs: int, verbose : int = 2):
        self.minrew = min_reward
        self.actions = actions
        self.base_model = base_model
        self.epochs = epochs
        self.batch = batch
        self.verbose = verbose
        self.fitted = {a: False for a in actions}
        self.scalers = {a : StandardScaler() for a in actions}
        self.oracles = {a : clone_model(base_model) for a in actions}
        [self.oracles[a].compile(loss = 'mse', optimizer = 'adam') for a in actions]
    #
    def __fit_oracle__(self, oracle: int, X: pd.DataFrame) -> None:
        if X.shape[0] == 0:
            return
        y = X[["reward"]]
        X = X.drop(["reward","action"], axis = 1)
        #
        self.scalers[oracle] = self.scalers[oracle].partial_fit(X)
        X = self.scalers[oracle].transform(X)
        data = tf.data.Dataset.from_tensor_slices((X, y)).batch(self.batch)
        self.oracles[oracle].fit(data, epochs = self.epochs, verbose = self.verbose)
        self.fitted[oracle] = True
    #
    def __predict_oracle__(self, oracle:int, X: pd.DataFrame) -> int:
        if not self.fitted[oracle]:
            return self.minrew
        X = self.scalers[oracle].transform(X)
        return(self.oracles[oracle].predict(X)[0][0])

# base model serves as a template on how to specify this parameter in NeuralOracle 
base_model = tf.keras.Sequential()
base_model.add(layers.Dense(20, activation = 'relu', input_shape=(30,)))
base_model.add(layers.Dense(10, activation = 'relu'))
base_model.add(layers.Dense(1))