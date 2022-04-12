import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"
import plotly.graph_objects as go

def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date'])
    df.dropna(inplace=True)
    df = df[(df.Day >= 1) & (df.Day <= 31)]
    df = df[(df.Year >= 0)]
    df = df[(df.Month >= 1) & (df.Month <= 12)]
    df = df[(df.Temp >= -20) & (df.Temp<= 50)] # range of normal temp
    df["DayOfYear"] = df["Date"].dt.dayofyear
    return df






if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data(r'C:\Users\Evyatar\Desktop\IML\IML.HUJI\datasets\City_Temperature.csv')



    # # Question 2 - Exploring data for specific country
    #
    israel = df[df.Country == "Israel"]
    fig = px.scatter(x= israel.DayOfYear, y= israel.Temp, color= israel.Year.astype(str))
    fig.update_layout(title= f"TEMPERATURE PER DAY OF YEAR",
                      xaxis_title = f"day of the year",
                      yaxis_title = f"temperature ",
                      legend_title = f"year"
                      )
    fig.show()
    month = israel.groupby(['Month']).Temp.agg(std='std')
    month_bar = px.bar(month, y="std", title="STD By Different Months")
    month_bar.show()
    # Question 3 - Exploring differences between countries
    month_and_country = df.groupby(['Month','Country']).Temp.agg(mean = 'mean',std='std').reset_index()
    fig = px.line(month_and_country,x='Month',y='mean',color=month_and_country.Country.astype(str),title="Mean "
                                                                                                         "Temarture "
                                                                                                         "Of diffrent "
                                                                                                         "Countrys",
                  error_y= 'std')
    fig.update_layout(legend_title = f"Country")
    fig.show()
    # Question 4 - Fitting model for different values of `k
    train_x, train_y, test_x, test_y = split_train_test(israel.DayOfYear,israel.Temp,0.75)
    losses = []
    for k in range(1,11):
        p_f = PolynomialFitting(k)
        p_f._fit(train_x.to_numpy(),train_y.to_numpy())
        l = p_f._loss(test_x.to_numpy(), test_y.to_numpy())
        losses.append(round(l,2))
    losses = np.asarray(losses)
    print(losses)
    ms = np.arange(1,11)
    loss_by_degree_bar = px.bar(x=ms, y=losses, title="Error Value by Polynomial Degree",text_auto=True)
    loss_by_degree_bar.show()


    # Question 5 - Evaluating fitted model on different countries
    best_model = PolynomialFitting(5)
    best_model._fit(train_x.to_numpy(), train_y.to_numpy())
    country_and_loss = dict()
    for country in df.Country.unique():
        if country != 'Israel' :
            curr_country = df[df.Country == country]
            loss = best_model._loss(curr_country.DayOfYear,curr_country.Temp)
            country_and_loss[country] = loss
    loss_bar = px.bar( x= country_and_loss.keys(), y=country_and_loss.values(),
                                    title="Israel-fitted Model on Other Countries")
    loss_bar.show()
