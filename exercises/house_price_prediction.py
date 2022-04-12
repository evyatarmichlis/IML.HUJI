from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"

def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)
    df.dropna(inplace=True)
    df.drop(['long', 'date', 'lat', 'id'], axis=1, inplace=True)
    df = df.drop(df.index[df.bedrooms <= 0])
    df = df.drop(df.index[df.sqft_living <= 0])
    df = df.drop(df.index[df.floors <= 0])
    df = df.drop(df.index[df.bathrooms < 0])
    df = df.drop(df.index[df.price < 0])
    # df = pd.get_dummies(df, columns=['zipcode'])

    df['yr_built_or_renovated'] = df[['yr_built', 'yr_renovated']].max(axis=1)
    df.drop(['yr_built', 'yr_renovated'], axis=1, inplace=True)
    price = df.pop('price')

    return df, price

def pearson_correlation(v1,v2) -> float:
    "calculate the pearson correlation"
    sigma1 = np.std(v1)
    sigma2 = np.std(v2)
    cov_matrix = np.cov(v1,v2)

    return (cov_matrix[0][1]/(sigma1*sigma2))





def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for feature in X.columns:
        feature_col = X.loc[:, feature]
        pearson = pearson_correlation( feature_col , y)
        x_axis = feature_col
        y_axis = y
        go.Figure([go.Scatter(x=x_axis, y=y_axis, name='correlation' + str(pearson), showlegend=True,
                                    # why need name here if i have title down there?
                                    marker=dict(color="black", opacity=.7), mode="markers",
                                    line=dict(color="black", width=1))],
                        layout=go.Layout(title=r"$\text{(1) feature and price}  $",
                                         xaxis={"title": feature},
                                         yaxis={"title": "price"},
                                         height=400)).show()


if __name__ == '__main__':
    np.random.seed(0)


    # Question 1 - Load and preprocessing of housing prices dataset
    df, price = load_data(r'C:\Users\Evyatar\Desktop\IML\IML.HUJI\datasets\house_prices.csv')
    # Question 2 - Feature evaluation with respect to response
    # feature_evaluation(df,price,output_path= "C:/Users/Evyatar/Desktop/IML/EX2/plots")


    # Question 3 - Split samples into training- and testing sets.
    train_x,train_y,test_x,test_y = split_train_test(df,price,0.75)
    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    price_np = price.to_numpy()
    means = []
    stds = []
    mse_list = {}
    for i in range(10,101):
        loss = []
        for j in  range(10):
            x_train_plot = train_x.sample(frac=(i / 100))
            y_train_plot = train_y.loc[x_train_plot.index]
            linear_regression = LinearRegression()
            linear_regression._fit(x_train_plot.to_numpy(),
                                                       y_train_plot.to_numpy())
            l = linear_regression._loss(test_x.to_numpy(), test_y.to_numpy())
            loss.append(l)
        loss = np.array(loss)
        means.append(loss.mean())
        stds.append(loss.std())

    means = np.asarray(means)
    stds = np.asarray(stds)
    std_minus_2 = (means) - 2 * (stds)
    std_plus_2 = (means) + 2 * (stds)
    fig = go.Figure()
    ms = np.arange(10, 101)
    fig.add_trace(go.Scatter(x=ms, y=means, mode="markers+lines", name="Mean Loss", line=dict(dash="dash"), marker=dict(color="black", opacity=.7)))
    fig.add_trace(go.Scatter(x=ms, y=std_minus_2, fill=None, mode="lines", line=dict(color="lightgrey"), showlegend=False))
    fig.add_trace(go.Scatter(x=ms, y=std_plus_2, fill='tonexty', mode="lines", line=dict(color="lightgrey"), showlegend=False))
    fig.update_layout(title = "Loss and Variance As Function Of Sample Percentage",
                        xaxis_title="Percentage",
                        yaxis_title="MSE Loss")
    fig.show()

