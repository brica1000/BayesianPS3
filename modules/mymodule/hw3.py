# -*- coding: utf-8 -*-
"""
Bayesian PS 3
"""
from scipy.stats import norm # note scale is stdev.
from scipy.stats import uniform
# from scipy.stats import gamma # must set a = shape (alpha from wiki), rate (1/ scale[beta])
from scipy.stats import t
from scipy.special import gamma as gammaf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import statsmodels.api as sm


from bokeh import mpl
from bokeh.plotting import figure, output_file
from bokeh.models import Label
from bokeh.layouts import column
from bokeh.io import show



def process(beta_not = np.transpose([0,10,5000,10000,10000])):
    # Load and format the data
    df = pd.read_table('HDATA.txt', delim_whitespace=True)  # It seems numpy and pandas play nice
    df.head()
    #df.shape
    X = df.loc[:,'lot':'sty']
    y = df.sell
    X = sm.add_constant(X)
    X.head

    # Some useful functions
    def constants(X, y, N=546, k=5, beta_not = beta_not,
                             var_beta = [10000**2,5**2,25000**2,5000**2,5000**2],
                             s_not_sq = 5000**2, v_not = 5):
        XX = np.matmul(np.transpose(X),X)
        v = N-k
        sigma_beta = np.diag(var_beta)
        V_not = sigma_beta*((v_not-2)/(v_not*s_not_sq)) #Is this the problem?
        beta_hat = np.linalg.lstsq(X,y)[0] # We only want the coefficients, so take first output
        s_sq = (np.matmul(np.transpose(y-np.matmul(X,beta_hat)),(y-np.matmul(X,beta_hat))))/v
        vs_bar_sq =( v*s_sq + v_not*s_not_sq +
                np.matmul(np.transpose(beta_hat - beta_not),
                np.matmul(np.linalg.inv(V_not + np.linalg.inv(XX)),(beta_hat-beta_not))))
        V_bar_inv= (np.linalg.inv(V_not) + XX)
        v_bar = N + v_not
        beta_bar = np.matmul(np.linalg.inv(V_bar_inv),
                             (np.matmul(np.linalg.inv(V_not),beta_not) + np.matmul(XX,beta_hat)))

        return( beta_not, V_not, s_not_sq, v_not, v, beta_hat,
                s_sq, beta_bar, vs_bar_sq, V_bar_inv, v_bar)


    ( beta_not, V_not, s_not_sq, v_not, v, beta_hat,
                s_sq, beta_bar, vs_bar_sq, V_bar_inv, v_bar) = constants (X,y)

    # Posteriors
    def post_stats(X, y, N=546, k=4, beta_not = np.transpose([0,10,5000,10000,10000]),
                             var_beta = [10000**2,5**2,25000**2,5000**2,5000**2],
                             s_not_sq = 5000**2, v_not = 5):
        (beta_not, V_not, s_not_sq, v_not, v, beta_hat, s_sq, beta_bar, vs_bar_sq, V_bar_inv, v_bar) = constants (X,y)
        exp_beta = beta_bar
        v_beta = vs_bar_sq*np.linalg.inv(V_bar_inv)/(v_bar-2)
        exp_h = vs_bar_sq**1*v_bar
        v_h = 2*vs_bar_sq**-2*v_bar
        return (exp_beta, v_beta, exp_h, v_h)

    (exp_beta, v_beta, exp_h, v_h) = post_stats(X, y, N=546, k=4, beta_not = np.transpose([0,10,5000,10000,10000]),
                             var_beta = [10000**2,5**2,25000**2,5000**2,5000**2],
                             s_not_sq = 5000**2, v_not = 5)

    def plot_betas():
        output_file("lines.html")
        graph_titles = ['B0','B1','B2','B3','B4']
        graphs = []
        for b,v,graph_title in zip(beta_bar,v_beta.diagonal(),graph_titles):
            post_b = norm.rvs(size = 1000, loc=b, scale = math.sqrt(v))
            post_b = pd.Series(post_b)
            post_b.plot.kde()            # graph = figure(title=graph_title, x_axis_label='posterior beta', y_axis_label='density')
            graph = mpl.to_bokeh()
            graphs.append(graph)
        show(column(graphs))

    plot_betas()



def monte(p=.5, N=1000, data=(0,0)): # Enter data as (x,y)
    mean = [0,0]
    cov = [[1,p],[p,1]]
    x,y= np.random.multivariate_normal(mean, cov, N).T
    fig = figure(title="simple line example", x_axis_label='x', y_axis_label='y')
    fig.circle(x, y, color='grey')
    fig.xaxis.bounds = (-3, 3)
    fig.yaxis.bounds = (-3, 3)
    # Means
    mean_1 = sum(x)/N
    mean_2 = sum(y)/N
    mytext = Label(x=-3, y=-2.5, text='mean_1= '+ str(mean_1) + '\n mean_2 = '+ str(mean_2))
    fig.add_layout(mytext)
    # Variance
    var_1 = sum(np.square(x))/N - mean_1**2
    var_2 = sum(np.square(y))/N - mean_2**2
    vartext = Label(x=-3, y=-3, text='var_1= '+ str(var_1) + '\n var_2 = '+ str(var_2))
    fig.add_layout(vartext)
    # Add another layer
    if data != (0,0):
        z,q = data
        fig.circle(z,q, color='red')
    show(fig)


def gibbs(iterr1=1000, iterr2=1000, p=.5):
    x1 = np.random.normal()
    x = [x1]
    y = [0]  # to keep things symmetric
    for i in range(iterr1):
        y.append(np.random.normal(loc=p*x[-1],scale=math.sqrt(1-p**2)))
        x.append(np.random.normal(loc=p*y[-1],scale=math.sqrt(1-p**2)))
    monte(p=p, N=iterr2, data = (x,y))
    return (x,y)









if __name__ == "__main__":
    process()
