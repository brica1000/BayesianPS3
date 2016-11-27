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

from bokeh import mpl
from bokeh.plotting import figure, output_file
from bokeh.layouts import column
from bokeh.io import show


def process(x_star=.5):
    def create_data(beta=2, h=1, N=100):
        x = uniform.rvs(size=N)
        eps = norm.rvs(size=N, scale=1/h**2)
        y = x*beta + eps
        return x, y, eps, N


    # output to static HTML file
    #output_file("lines.html")

    # create a new plot with a title and axis labels
    #p = figure(title="simple line example", x_axis_label='x', y_axis_label='y')
    (x,y,eps,N) = create_data()
    #p.circle(x, y)
    #show(p)

    def constants(x, y, N=100, beta_not = 2, V_not = 1, s_not_minus_sq = 1, v_not = 1):
        v = N-1
        sum_xsq = sum(x**2)
        beta_hat = np.dot(x,y) /  sum_xsq
        s_sq = sum((y-beta_hat*x)**2)/N
        beta_bar = (V_not**-1*beta_not + sum_xsq*beta_hat)/(V_not**-1 + sum_xsq)
        vs_bar_sq =( v*s_sq + v_not*s_not_minus_sq**-1 +
            (beta_hat - beta_not)**2*(V_not + sum_xsq**-1)**-1)
        V_bar = 1*(V_not**-1 + sum_xsq)**-1
        v_bar = N + v_not
        return( beta_not, V_not, s_not_minus_sq, v_not, v, sum_xsq, beta_hat,
            s_sq, beta_bar, vs_bar_sq, V_bar, v_bar)

    def plot_predictive(x=x,y=y,x_star=x_star, sample=100):
        tees = []
        output_file("lines.html")
        p = figure(title="simple line example", x_axis_label='x', y_axis_label='y')
        (beta_not, V_not, s_not_minus_sq, v_not, v, sum_xsq, beta_hat,
            s_sq, beta_bar, vs_bar_sq, V_bar, v_bar) = constants(x,y)
        for i in range(sample):
            tees.append(t.rvs(v_bar, beta_bar*x_star, vs_bar_sq**-1*v_bar*(1+x_star**2*V_bar)))
        p.circle(x,y)
        p.square(x_star*np.ones(len(tees)),tees, color='red')
        ser = pd.Series(tees)
        q = figure(title="simple line example", x_axis_label='x', y_axis_label='y')
        ser.plot.kde()
        q = mpl.to_bokeh()
        show(column(p,q))


    plot_predictive()



'''
    """a) Generate data"""
    def create_data(beta=2, h=1, N=100):
        x = uniform.rvs(size=N)
        eps = norm.rvs(size=N, scale=1/h**2)
        y = x*beta + eps
        return x, y, eps, N


    fig = plt.figure() # For pythonanywhere
    # Check our distribution
    (x,y, eps, N) = create_data()
    plt.plot(x,y,'ro')
    plt.show()




#fig.savefig("graph.png") # For pythonanywhere
#http://www.pythonanywhere.com/user/brica1000/files/home/brica1000/graph.png # For pythonanywhere

print(' b) Calculate predictive y* for x=.5 using the joint post')
# Assume a normal gamma prior
# Our constants
def constants(x, y, N=100, beta_not = 2, V_not = 1, s_not_minus_sq = 1, v_not = 1):
    v = N-1
    sum_xsq = sum(x**2)
    beta_hat = np.dot(x,y) /  sum_xsq
    s_sq = sum((y-beta_hat*x)**2)/N
    beta_bar = (V_not**-1*beta_not + sum_xsq*beta_hat)/(V_not**-1 + sum_xsq)
    vs_bar_sq =( v*s_sq + v_not*s_not_minus_sq**-1 +
        (beta_hat - beta_not)**2*(V_not + sum_xsq**-1)**-1)
    V_bar = 1*(V_not**-1 + sum_xsq)**-1
    v_bar = N + v_not
    return( beta_not, V_not, s_not_minus_sq, v_not, v, sum_xsq, beta_hat,
        s_sq, beta_bar, vs_bar_sq, V_bar, v_bar)


# Distribtutions
# Joint posterior Beta, h: first sample from h, then from B, this isn't strait forward
# post_h = gamma.rvs(size = 1, loc=vs_bar_sq**-1*v_bar, scale=math.sqrt(v_bar))
# post_b = norm.rvs(size = 1, loc=beta_bar ,scale = math.sqrt(V_bar))

# From the lecture, y* is distibuted t
# Lets plot x=.5 distribution on top of our data, in blue
def plot_predictive(x=x,y=y,x_star=.5, sample=100):
    tees = []
    plt.clf()
    fig = plt.figure()
    (beta_not, V_not, s_not_minus_sq, v_not, v, sum_xsq, beta_hat,
        s_sq, beta_bar, vs_bar_sq, V_bar, v_bar) = constants(x,y)
    for i in range(sample):
        tees.append(t.rvs(v_bar, beta_bar*x_star, vs_bar_sq**-1*v_bar*(1+x_star**2*V_bar)))
    plt.subplot(211)
    plt.plot(x,y,'ro')
    plt.plot(x_star*np.ones(len(tees)),tees,'bx')
    plt.subplot(212)
    ser = pd.Series(tees)
    ser.plot.kde()
    plt.show()
    fig.savefig("graph.png")

plot_predictive()

print(' c) Calculate posterior statistics, and Bayes factor ')
# Posteriors
def post_stats(x, y, N=100, x_star=.5, V_not=1, v_not=1, beta_not=2, s_not_minus_sq = 1):
    ( beta_not, V_not, s_not_minus_sq, v_not, v, sum_xsq, beta_hat,
     s_sq, beta_bar, vs_bar_sq, V_bar, v_bar) = constants(x,y,N=N,V_not=V_not,v_not=v_not,beta_not=beta_not,s_not_minus_sq=s_not_minus_sq)
    exp_beta = beta_bar
    v_beta = vs_bar_sq*V_bar/(v_bar-2)
    exp_h = vs_bar_sq**1*v_bar
    v_h = 2*vs_bar_sq**-2*v_bar
    exp_y = x_star * beta_bar
    v_y = vs_bar_sq*(1+x_star**2*V_bar)/(v_bar-2)
    return (exp_beta, v_beta, exp_h, v_h, exp_y, v_y)

(exp_beta, v_beta, exp_h, v_h, exp_y, v_y) = post_stats(x, y)
print('exp_beta = {}, v_beta = {}, exp_h = {}, v_h = {}, exp_y = {}, v_y = {}'.format(exp_beta, v_beta, exp_h, v_h, exp_y, v_y))

# Bayes factors
def which_M(x, y, eps, N=100, restriction_b=0, V_not=1, v_not=1, beta_not=2, s_not_minus_sq = 1):
    ( beta_not, V_not, s_not_minus_sq, v_not, v, sum_xsq, beta_hat,
     s_sq, beta_bar, vs_bar_sq, V_bar, v_bar) = constants(x,y,N=N,V_not=V_not,v_not=v_not,beta_not=beta_not,s_not_minus_sq=s_not_minus_sq)
    pM_one = ( gammaf(v_bar/2)*(v_not*vs_bar_sq/v_not)**(-v_not/2)*(V_bar/V_not)**(1/2)*(vs_bar_sq)**(-v_bar)/2 )/(gammaf(v_not/2)*math.pi**(N/2))
    y = eps # make our restriction, is this right?
    ( beta_not, V_not, s_not_minus_sq, v_not, v, sum_xsq, beta_hat,
     s_sq, beta_bar, vs_bar_sq, V_bar, v_bar) = constants(x,y,N=N,V_not=V_not,v_not=v_not,beta_not=beta_not,s_not_minus_sq=s_not_minus_sq)
    pM_two = ( gammaf(v_bar/2)*(v_not*vs_bar_sq/v_not)**(-v_not/2)*(V_bar/V_not)**(1/2)*(vs_bar_sq)**(-v_bar)/2 )/(gammaf(v_not/2)*math.pi**(N/2))
    factor = pM_one/pM_two
    if factor > 1:
        text = 'Our unrestricted model is more likely, Bayes factor = '
    else:
        text = 'Model 2 is more likely, Bayes factor = '
    return (text, factor, pM_one, pM_two)

(text, factor, pM_one, pM_two) = which_M(x, y, eps)
print(text, factor, pM_one, pM_two)

print('d) How do our posterior statistics, and our Bayes factors change as'  '\n'
'we change V_not? ')
# V_not is our prior on the variance of Beta, right?
# => smaller means we are more sure of beta?
for i in [1000000,100,10,.1,.01]:
     (exp_beta, v_beta, exp_h, v_h, exp_y, v_y) = post_stats(x, y, V_not=i)
     print('V_not = {}, exp_beta = {}, v_beta = {}, exp_h = {}, v_h = {}, exp_y = {}, v_y = {}'.format(i, exp_beta, v_beta, exp_h, v_h, exp_y, v_y))
# Notice our posterior variance of beta gets smaller as V_not decreases.

# What about the Bayes factor? Maybe it shouldn't change?
for i in [1000000,100,10,.1,.01]:
    (text, factor, pM_one, pM_two)=which_M(x,y,eps,V_not=i)
    print('V_not = {}, text = {} {}, pM_one = {}, pM_two = {}'.format(i, text, factor, pM_one, pM_two))
# If V_not is small, we are sure, and our Bayes factor blows up.

print( ' e) Same question, now change v_not')
for i in [1000000,100,10,.1,.01]:
     (exp_beta, v_beta, exp_h, v_h, exp_y, v_y) = post_stats(x, y, v_not=i)
     print('v_not = {}, exp_beta = {}, v_beta = {}, exp_h = {}, v_h = {}, exp_y = {}, v_y = {}'.format(i, exp_beta, v_beta, exp_h, v_h, exp_y, v_y))
# Notice our posterior mean and variance of h gets smaller as V_not decreases.

# What about the Bayes factor? Maybe it shouldn't change?
for i in [1000000,100,10,.1,.01]:
    (text, factor, pM_one, pM_two)=which_M(x,y,eps,v_not=i)
    print('v_not = {}, text = {} {}, pM_one = {}, pM_two = {}'.format(i, text, factor, pM_one, pM_two))
# There is a problem when v_not is too large.

print(' f) Set beta_not far from 2, and repeat d')
for i in [1000000,100,10,.1,.01]:
    (exp_beta, v_beta, exp_h, v_h, exp_y, v_y) = post_stats(x, y, V_not=i, beta_not=0)
    print('V_not = {}, exp_beta = {}, v_beta = {}, exp_h = {}, v_h = {}, exp_y = {}, v_y = {}'.format(i, exp_beta, v_beta, exp_h, v_h, exp_y, v_y))
# Things change alot!

# What about the Bayes factor? Maybe it shouldn't change?
for i in [1000000,100,10,.1,.01]:
    (text, factor, pM_one, pM_two)=which_M(x,y,eps,V_not=i, beta_not=0)
    print('V_not = {}, text = {} {}, pM_one = {}, pM_two = {}'.format(i, text, factor, pM_one, pM_two))
# Our prior now makes us think that beta is really 0!

print(' g) Set h, i.e. s_not__minus_sq, far from 1, and repeat e')
for i in [1000000,100,10,.1,.01]:
     (exp_beta, v_beta, exp_h, v_h, exp_y, v_y) = post_stats(x, y, v_not=i, s_not_minus_sq=100)
     print('v_not = {}, exp_beta = {}, v_beta = {}, exp_h = {}, v_h = {}, exp_y = {}, v_y = {}'.format(i, exp_beta, v_beta, exp_h, v_h, exp_y, v_y))
# Notice our posterior mean  of h gets smaller as v_not decreases.

# What about the Bayes factor? Maybe it shouldn't change?
for i in [1000000,100,10,.1,.01]:
    (text, factor, pM_one, pM_two)=which_M(x,y,eps,v_not=i,s_not_minus_sq=100)
    print('v_not = {}, text = {} {}, pM_one = {}, pM_two = {}'.format(i, text, factor, pM_one, pM_two))
# When v_not is large, we favor our restricted model.

print(' h) My findings:  Our analysis is indeed sensitive to our priors, ' '\n'
'indicating that we should be aware or their power to influence the outcome.')

print(' i) Now change N and see how things change')
(x,y, eps, N) = create_data(N=10)
plt.plot(x,y,'ro')
plt.show()

plot_predictive()
(exp_beta, v_beta, exp_h, v_h, exp_y, v_y) = post_stats(x, y)
print('exp_beta = {}, v_beta = {}, exp_h = {}, v_h = {}, exp_y = {}, v_y = {}'.format(exp_beta, v_beta, exp_h, v_h, exp_y, v_y))

# Let's look at if our data can overcome our prior on beta with such a small sample.
for i in [1000000,100,10,.1,.01]:
     (exp_beta, v_beta, exp_h, v_h, exp_y, v_y) = post_stats(x, y, N=10, beta_not=0, V_not=i)
     print('V_not = {}, exp_beta = {}, v_beta = {}, exp_h = {}, v_h = {}, exp_y = {}, v_y = {}'.format(i, exp_beta, v_beta, exp_h, v_h, exp_y, v_y))
for i in [1000000,100,10,.1,.01]:
    (text, factor, pM_one, pM_two)=which_M(x,y,eps, N=10, beta_not=0, V_not=i)
    print('V_not = {}, text = {} {}, pM_one = {}, pM_two = {}'.format(i, text, factor, pM_one, pM_two))
# It cannot.

(x,y, eps, N) = create_data(N=1000)
plt.plot(x,y,'ro')
plt.show()

plot_predictive()
(exp_beta, v_beta, exp_h, v_h, exp_y, v_y) = post_stats(x, y)
print('exp_beta = {}, v_beta = {}, exp_h = {}, v_h = {}, exp_y = {}, v_y = {}'.format(exp_beta, v_beta, exp_h, v_h, exp_y, v_y))

# Let's look at if our data can overcome our prior on beta with a large sample.
for i in [1000000,100,10,.1,.01]:
     (exp_beta, v_beta, exp_h, v_h, exp_y, v_y) = post_stats(x, y, N=1000,beta_not=2, V_not=i)
     print('V_not = {}, exp_beta = {}, v_beta = {}, exp_h = {}, v_h = {}, exp_y = {}, v_y = {}'.format(i, exp_beta, v_beta, exp_h, v_h, exp_y, v_y))
for i in [1000000,100,10,.1,.01]:
    (text, factor, pM_one, pM_two)=which_M(x, y, eps, N=1000, beta_not=2, V_not=i)
    print('V_not = {}, text = {} {}, pM_one = {}, pM_two = {}'.format(i, text, factor, pM_one, pM_two))
# It cannot.
'''
if __name__ == "__main__":
    process()
