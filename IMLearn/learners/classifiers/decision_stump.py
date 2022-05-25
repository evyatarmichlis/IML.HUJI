from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None


    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        best_loss = np.inf
        for j in range(X.shape[1]):
            for sign in [1, -1]:
                thr, loss = self._find_threshold(X[:, j], y, sign)
                if loss < best_loss:
                    best_loss = loss
                    self.threshold_,  self.sign_,   self.j_  = thr, sign, j




    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return  (((X[:, self.j_] >= self.threshold_).astype(int))*2-1) * self.sign_



    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        from ...metrics import misclassification_error
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature
    
        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for
    
        labels: ndarray of shape (n_samples,)
            The labels to compare against
    
        sign: int
            Predicted label assigned to values equal to or above threshold
    
        Returns
        -------
        thr: float
            Threshold by which to perform split
    
        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold
    
        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
    
        """
        best_loss = np.inf
        thr = -np.inf
        for v in values:
            y_pred = (values >= v).astype(int)
            y_pred[y_pred == 1] = sign
            y_pred[y_pred == 0] = -sign

            loss = self._loss_helper(labels, y_pred, np.abs(labels))
            if loss < best_loss:
                best_loss = loss
                thr = v
        if thr == np.min(values):
            return -np.inf, best_loss
        return thr, best_loss



    def _loss(self, X: np.ndarray, y: np.ndarray) :
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return self._loss_helper(y,  self._predict(X), np.abs(y))

    def _loss_helper(self, y_true: np.ndarray, y_pred: np.ndarray, d: np.ndarray) :


        return d @ (np.sign(y_true) != np.sign(y_pred))
