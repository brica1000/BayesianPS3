# -*- coding: utf-8 -*-
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

from pandas.tools.plotting import autocorrelation_plot

from bokeh import mpl
from bokeh.plotting import figure, output_file
from bokeh.models import Label
from bokeh.layouts import column
from bokeh.io import show


"""
The Data

X1=norm.rvs(size=100)
X2=norm.rvs(size=100)
X=np.column_stack((X1,X2))
X=pd.DataFrame(X)
eps=norm.rvs(size=100)
latent_y = 3*X1 + 10*X2 + eps
y = []
i = range(100)
for entry in latent_y:
    if entry > 0:
        y.append(1)
    else:
        y.append(0)
y=pd.Series(y)
"""
def create_data():
    X1=norm.rvs(size=100)
    X2=norm.rvs(size=100)
    X=np.column_stack((X1,X2))
    X=pd.DataFrame(X)
    eps=norm.rvs(size=100)
    latent_y = 3*X1 + 10*X2 + eps
    y = []
    i = range(100)
    for entry in latent_y:
        if entry > 0:
            y.append(1)
        else:
            y.append(0)
    y=pd.Series(y)
    return (X,y,latent_y)


"""
Make draws from NG(beta_bar,V_bar,s_min_sq,v_bar), so we need
our constants, a function to draw from NG, and to draw our y_stars,
and finally our gibbs sampler to update our draws as we converge
"""
def constants(X, y, N=100, k=2, beta_not = [.5,.5], var_beta = [1,1]):
    XX = np.matmul(np.transpose(X),X)
    V_not = np.diag(var_beta)
    V_not_inv = np.linalg.inv(V_not)
    beta_hat = np.linalg.lstsq(X,y)[0] # We only want the coefficients, so take first output
    V_bar_inv = V_not_inv + XX
    V_bar = np.linalg.inv(V_bar_inv)
    beta_bar = np.matmul(V_bar,
                         (np.matmul(V_not_inv,beta_not) + np.matmul(XX,beta_hat)))
    return(beta_bar, V_bar)

def NG_draw(beta_bar, V_bar):
    # Set h=1 for identification, so we just need beta
    beta_draw = np.random.multivariate_normal(beta_bar, V_bar)
    return (beta_draw)

def y_star_draw(beta,y,X,y_star_prev):
    y_star = []
    for dep,i,j in zip(y,range(len(X)),y_star_prev):
        z = norm.rvs(loc=np.matmul(X.loc[i],beta))
        if dep == 1:
            if z >= 0:
                y_star.append(z)
            else:
                y_star.append(j)
        else:
            if z < 0:
                y_star.append(z)
            else:
                y_star.append(j)
    return y_star

def gibbs(X,y,iterrs=500,burn=100):
    y_star = pd.Series(norm.rvs(size=len(y)))  # Initial y_star
    betas = []
    y_stars = []
    for i in range(iterrs):
        (beta_bar, V_bar_inv) = constants (X,y_star,beta_not=[3,10])
        (beta_draw)=NG_draw(beta_bar, V_bar_inv)
        new_y_star = y_star_draw(beta_draw,y,X,y_star)
        y_star = new_y_star
        betas.append(beta_draw)
        y_stars.append(y_star)
    posterior_draw = pd.DataFrame(betas)
    y_stars = pd.Series(y_stars)
    (b0,b1) = posterior_draw.loc[burn:iterrs].mean()
    return (b0,b1, y_stars, posterior_draw)



"""Use in production function"""
def full_gibbs(X, y, iterrs=500, burn=100):
    graphs = []
    (b0,b1, y_stars, posterior_draws) = gibbs(X,y,iterrs=iterrs,burn=burn)
    # Convergence information
    posterior_ts = posterior_draws.plot()
    posterior_ts = mpl.to_bokeh()
    graphs.append(posterior_ts)
    auto = autocorrelation_plot(posterior_draws)
    auto = mpl.to_bokeh()
    graphs.append(auto)
    # Posterior densities of our parameters
    for beta in posterior_draws:
        graph = str(beta)
        graph = posterior_draws[beta][burn:iterrs-1].plot.kde()
        plt.title("Posterior density of " + str(beta))
        graph = mpl.to_bokeh()
        graphs.append(graph)
    return graphs




def back_ground(X,y):
    N = len(y)
    x = X[1]
    radii = X[0] * .2
    colors = [
        "#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(150+80*x, 30+80*y)
    ]
    TOOLS="resize,crosshair,pan,wheel_zoom,box_zoom,reset,box_select,lasso_select"
    # create a new plot with the tools above, and explicit ranges
    p = figure(tools=TOOLS, x_range=(-2,2), y_range=(-20,20))
    # add a circle renderer with vectorized colors and sizes
    p.circle(x,y, radius=radii, fill_color=colors, fill_alpha=0.6, line_color=None)
    return p











if __name__ == "__main__":
    full_gibbs(X,y)
