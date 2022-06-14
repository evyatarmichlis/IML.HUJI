from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator



def split_data(i,X,y,k):
    range_x = np.arange(X.shape[0])
    separator = np.remainder(range_x, k)
    x_1,x_2 = X[separator == i] , X[separator != i]
    y_1,y_2 = y[separator == i] , y[separator != i]
    return x_1,x_2 , y_1,y_2

def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """

    val_score = []
    train_score = []
    for i in range(cv):
        x_1,x_2 , y_1,y_2 = split_data(i,X,y,cv)
        h_i = estimator.fit(x_2,y_2)
        val_score.append(scoring(y_1, h_i.predict(x_1)  ))
        train_score.append(scoring(y_2, h_i.predict(x_2) ))

    return np.array(val_score).mean(), np.array(train_score).mean()



