from __future__ import annotations
import numpy as np
import pandas as pd
import sklearn
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    x = np.linspace(-1.2, 2, n_samples)
    eps = np.random.normal(0, noise, n_samples)
    f = (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    y = f + eps
    x_df = pd.Series(x)
    y_df_series = pd.Series(y)
    y_df_series.name = 'y'
    train_X, train_Y, test_X, test_y = split_train_test(x_df, y_df_series, 2/3)
    train_X = train_X.to_numpy().squeeze()
    train_Y = train_Y.to_numpy()
    test_X = test_X.to_numpy().squeeze()
    test_y = test_y.to_numpy()
    # show data: real vs noisy train and test
    fig = go.Figure(
        [go.Scatter(x=x, y=f, mode="markers+lines", name="true values",
                    marker=dict(color="black", opacity=.7), ),

         go.Scatter(x=train_X, y=train_Y, fill=None, mode="markers", name="train values",marker=dict(color="red")),

         go.Scatter(x=test_X, y=test_y, fill=None, mode="markers", name="test values",marker=dict(color="blue")), ],
        layout=go.Layout(title=f"Generated Data with the f(x) and with the noise"))
    fig.show()


    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10

    train_score = []
    val_score = []
    degrees = np.arange(11)
    for deg in degrees:
        h_i = PolynomialFitting(deg)
        val, train = cross_validate(h_i, train_X, train_Y, mean_square_error)
        train_score.append(train)
        val_score.append(val)
    fig = go.Figure(data=[go.Bar(x=degrees, y=val_score, name="Validation Error",marker=dict(color="blue")),
                           go.Bar(x= degrees, y=train_score, name="Training Error",marker=dict(color= "red") )])
    fig.update_layout(title_text= "Average Error per  Polynomial Degree",
                       xaxis_title="Degree", yaxis_title="Mean Square Error loss")
    fig.show()


    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_degree = np.argmin(val_score)
    best_fit = PolynomialFitting(k=best_degree).fit(train_X, train_Y)
    test_error = mean_square_error(best_fit.predict(test_X), test_y)

    print(f' The Best Degree is {best_degree}, The test Error is {test_error} ')


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_X,test_X = X[:n_samples],X[n_samples:]
    train_y,test_y =  y[:n_samples],y[n_samples:]
    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lambdas = np.linspace(0.01, 3, n_evaluations)

    ridge_train_errors = []
    ridge_valid_errors = []
    lasso_train_errors = []
    lasso_valid_errors = []
    for l in lambdas:
        train_error,valid_error = cross_validate(RidgeRegression(l),train_X,train_y,mean_square_error)
        ridge_train_errors.append(train_error),ridge_valid_errors.append(valid_error)

        train_error,valid_error = cross_validate(Lasso(l),train_X,train_y,mean_square_error)
        lasso_train_errors.append(train_error),lasso_valid_errors.append(valid_error)

    fig = go.Figure(
        [go.Scatter(x=lambdas, y=ridge_train_errors, mode="markers", name="true values",
                    marker=dict(color="blue", opacity=.7), ),

         go.Scatter(x=lambdas, y=ridge_valid_errors, fill=None, mode="markers", name="train values",marker=dict(color="red")) ],
        layout=go.Layout(title="Ridge Regression"))
    fig.show()
    fig = go.Figure(
        [go.Scatter(x=lambdas, y=lasso_train_errors, mode="markers", name="true values",
                    marker=dict(color="blue", opacity=.7), ),

         go.Scatter(x=lambdas, y=lasso_valid_errors, fill=None, mode="markers", name="train values",marker=dict(color="red")) ],
        layout=go.Layout(title="lasso Regression"))
    fig.show()





    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_lasso_error = lambdas[np.argmin(lasso_valid_errors)]
    best_ridge_error = lambdas[np.argmin(ridge_valid_errors)]
    print(f"Best Validation Error for the Lasso  {best_lasso_error}")
    print(f"Best Validation Error for the Ridge  {best_ridge_error}")


    ridge = RidgeRegression(best_ridge_error).fit(train_X, train_y)
    lasso = Lasso(best_lasso_error).fit(train_X, train_y)
    least_square = LinearRegression().fit(train_X, train_y)
    print(f"Error For Lasso: {mean_square_error(test_y, lasso.predict(test_X))}")
    print(f"Error For Ridge: {ridge.loss(test_X, test_y)}")
    print(f"Error For Least Squares: {least_square.loss(test_X, test_y)}")


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500,noise=10)
    select_regularization_parameter()