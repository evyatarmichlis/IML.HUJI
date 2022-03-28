from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    sample_size = 1000
    u = UnivariateGaussian()
    X = np.random.normal(10,1,sample_size)
    u = u.fit(X)
    print( "( mu is "+str(u.mu_)+ ", var is: " + str(u.var_) + ")" )
    # Question 2 - Empirically showing sample mean is consistent
    absolute_dist = []
    X_copy = np.copy(X)
    ms = []
    for x in range(10,len(X)+10,10):
        current_G = UnivariateGaussian()
        current_G.fit(X_copy[: x])
        ms.append(x)
        absolute_dist.append(np.abs(current_G.mu_ - u.mu_))
    go.Figure([go.Scatter(x=ms, y=absolute_dist, mode='markers+lines',
                          name=r'$\widehat\mu$')],
              layout=go.Layout(
                  title=r"$\text{Distance between Estimation of "
                        r"Expectation to "
                        r"real Expectation As Function Of Number Of Samples}$",
                  xaxis_title="$m\\text{ - number of samples}$",
                  yaxis_title="Dist",
                  height=300)).show()



    # Question 3 - Plotting Empirical PDF of fitted model #TODO check how it suposte to look like
    pdfs = u.pdf(X)
    go.Figure([go.Scatter(x= X, y=pdfs, mode='markers',
                          name=r'$\widehat\mu$')],
              layout=go.Layout(
                  title=r"$\text{ PDFS of previously drawn sample }$",
                  xaxis_title="$\\text{ sample number}$",
                  yaxis_title="PDF",
                  height=600)).show()

    # array = np.array([1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1, -3, 1, -4, 1, 2, 1,
    #           -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1, 0, 3, 5, 0, -2])
    # print(u.log_likelihood(1,1,array))
def test_multivariate_gaussian():
    m =MultivariateGaussian()
    # Question 4 - Draw samples and print fitted model
    mean = [0, 0, 4, 0]
    cov = np.array([[1,0.2,0,0.5],[0.2,2,0,0],[0,0,1,0],[0.5,0,0,1]])
    X = np.random.multivariate_normal(mean,cov , size=1000).T
    m.fit(X)
    print("mu is : \n" + str(m.mu_))
    print("cov matrix is: "
          " \n" + str(m.cov_))
    # Question 5 - Likelihood evaluation
    vec1 = np.linspace(-10, 10, 200)
    vec2 = np.linspace(-10, 10, 200)
    likelihood = [[m.log_likelihood(np.array([f1, 0, f3, 0]).T, cov, X) for i,f1 in enumerate(vec1)] for j,f3 in enumerate(vec2)]
    like = np.asarray(likelihood)

    fig = px.imshow(likelihood,
                    labels=dict(x="F1", y="F3", color="LIKELIHOOD"),
                    x=vec1,
                    y=vec2
)
    fig.update_xaxes(side="top")
    fig.show()
    fig = px.imshow(likelihood, text_auto=True)


    # fig.show()




    # Question 6 - Maximum likelihood
    result = np.where(like == np.amax(like))
    print("max value of f1 :" + str(vec1[result[1]]))
    print("max value of f3 :" + str(vec1[result[0]]))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
