# -*- coding: utf-8 -*-
from scipy.stats import norm # note scale is stdev.
from scipy.stats import uniform
import scipy.stats as st
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
import statsmodels  # This is ugly!

from matplotlib import cm
from numpy import linspace

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

# Real data
def load_data():
    # Load and format the data
    df = pd.read_csv('data_try_2.csv')
    # Use the below path for production
    # df = pd.read_csv('/home/brica999/BayesianPS3/data_try_2.csv')
    df['member_years'] = pd.to_numeric(df['member_since'].str[2:6]).apply(lambda x: 2017-x)
    male = []
    for entry in df.sex:
        if entry == 'Male':
            male.append(1)
        else:
            male.append(0)
    df['male'] = pd.Series(male)
    df['age_num'] = pd.to_numeric(df.age.str[2:4])
    df = df[df.profile_len != 'flag'] # Remove the entries with no profile_len
    df = df[df.sex != 'other'] # Drop the other sex, it confounds
    df['profile_len'] = pd.to_numeric(df.profile_len)
    df = df.dropna()    # One final clean
    df.index = range(len(df)) # We have screwed up our index
    X = df[['age_num', 'profile_len', 'member_years', 'male']]
    y = df.good
    X = sm.add_constant(X)
    return X,y


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

def gibbs(X,y,iterrs=500,burn=100, beta_not = [1,1], var_beta = [1,1]):
    y_star = pd.Series(norm.rvs(size=len(y)))  # Initial y_star
    betas = []
    y_stars = []
    for i in range(iterrs):
        (beta_bar, V_bar_inv) = constants (X,y_star,beta_not=beta_not, var_beta=var_beta)
        (beta_draw)=NG_draw(beta_bar, V_bar_inv)
        new_y_star = y_star_draw(beta_draw,y,X,y_star)
        y_star = new_y_star
        betas.append(beta_draw)
        y_stars.append(y_star)
    posterior_draw = pd.DataFrame(betas)
    y_stars = pd.Series(y_stars)
    betas = posterior_draw.loc[burn:iterrs].mean()
    return (betas, y_stars, posterior_draw)

# Create plots of  our evolving distributions as we change the prior belief
def prior_sens(list_of_priors, iterrs=500, burn=100):
    colors = [ cm.viridis(x) for x in linspace(0, 1, len(list_of_priors)) ]
    (X,y,latent_y) = create_data()
    for prior,color in zip(list_of_priors,colors):
        (betas, y_stars, posterior_draws) = gibbs(X,y,iterrs=iterrs,burn=burn, beta_not=prior)
        for beta, i in zip(posterior_draws,range(len(prior))):
            plt.figure(i+1)
            posterior_draws[beta][burn:iterrs-1].plot.kde(c=color)
            plt.title("Posterior density of " + str(beta))
    plots = []
    for i in range(len(list_of_priors[0])):  # We cant send to bokehsooner, we arent done with the graph
        plt.figure(i+1)
        plot = mpl.to_bokeh()
        plots.append(plot)
    return plots

# Helper functions for autocorrelations that will work with Bokeh.
def auto_plot(df):
    plots = []
    for i in range(df.shape[1]): # For each column in the dataframe
        title = "ACF of our series of Beta " + str(i) + "'s"
        plot = better_auto(df[i],title=title)
        plots.append(plot)
    return plots

def better_auto(series, title):
    acc = statsmodels.tsa.stattools.acf(series, nlags=len(series))
    time = list(range(len(series)))
    plot = figure(title=title, x_axis_label='lag number', y_axis_label='ac')
    plot.line(time, acc, legend="Autocorrelations", line_width=2)
    return plot

def convergence(posterior_df,burn,alpha=.1):
    T = len(posterior_df[0]) - 1
    ABC = T - burn
    A = burn + .33*ABC
    B = burn + .66*ABC
    rt_a = math.sqrt(.33*ABC)
    rt_c = rt_a
    upper = st.norm.ppf(1 - alpha/2)
    lower = upper * -1
    text = []
    for i in posterior_df:
        mean_a = posterior_df[i].loc[burn:A].mean()
        std_a = posterior_df[i].loc[burn:A].std()
        mean_c = posterior_df[i].loc[B:T].mean()
        std_c = posterior_df[i].loc[B:T].std()
        cd = (mean_a - mean_c)/(std_a/rt_a + std_c/rt_c)
        if cd < upper and cd > lower:
            text.append("Beta " + str(i) + " has converged!")
    return text

"""Production Gibbs"""
def full_gibbs(X, y, beta_not, var_beta, iterrs=500, burn=100):
    graphs = []
    (betas, y_stars, posterior_draws) = gibbs(X,y,iterrs=iterrs,burn=burn, beta_not=beta_not, var_beta=var_beta)
    # Convergence information
    posterior_ts = posterior_draws.plot()
    posterior_ts.title.set_text('Posterior draws for each series of betas')
    posterior_ts = mpl.to_bokeh()
    graphs.append(posterior_ts)
    ac_plots = auto_plot(posterior_draws)
    for plot in ac_plots:
        graphs.append(plot)
    # Posterior densities of our parameters
    for beta in posterior_draws:
        graph = str(beta)
        graph = posterior_draws[beta][burn:iterrs-1].plot.kde()
        plt.title("Posterior density of Beta " + str(beta))
        graph = mpl.to_bokeh()
        graphs.append(graph)
    text = convergence(posterior_draws, burn)
    return (graphs, text)









# Home page picture
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
