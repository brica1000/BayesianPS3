from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, HttpResponseRedirect
from django.core.urlresolvers import reverse
import numpy as np
import pandas as pd

from .models import Code
from .forms import StatsInputForm

from modules.mymodule import hw,hw3,bayesian_probit

from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.resources import CDN

from scipy.stats import norm



def index(request):
    (X,y,latent_y) = bayesian_probit.create_data()
    p = bayesian_probit.back_ground(X,latent_y)
    script, div = components(p, CDN)
    return render(request, 'stats/index.html', {'script':script,'div':div})


def edit_code(request):
    if request.method == "POST":
        form = StatsInputForm(request.POST)
        if form.is_valid():
            result = form.save(commit=False)
            result.save()
            return HttpResponseRedirect(reverse('results'))
    else:
        form = StatsInputForm(initial={'title': '[0,10,5000,10000,10000]'})
    return render(request, 'stats/edit_code.html', {'form': form})

def results(request):
    results = Code.objects.all()
    x = eval(results[len(results)-1].title) # Input the last title as out beta_not
    if len(x) == 5:
        plots = hw3.process(beta_not = np.transpose(x))
        priors = x
        script, div = components(plots, CDN)
        return render(request, 'stats/results.html', {'script':script,'div':div,'priors':priors})
    else:
        return HttpResponseRedirect(reverse('edit_code'))


def edit_gibbs(request):
    if request.method == "POST":
        form = StatsInputForm(request.POST)
        if form.is_valid():
            result = form.save(commit=False)
            result.save()
            return HttpResponseRedirect(reverse('gibbs_results'))
    else:
        form = StatsInputForm(initial={'title': '(.5, 1000, 1000)'})
    return render(request, 'stats/edit_gibbs.html', {'form': form})

def gibbs_results(request):
    results = Code.objects.all()
    (p, iterr1, iterr2) = eval(results[len(results)-1].title) # Inputs
    if len((p, iterr1, iterr2)) == 3:
        plot = hw3.gibbs(p=p, iterr1=iterr1, iterr2=iterr2)
        script, div = components(plot, CDN)
        return render(request, 'stats/gibbs_results.html', {'script':script,'div':div,})
    else:
        return HttpResponseRedirect(reverse('edit_gibbs'))

"""Data Augmentation"""
def probit(request):
    return render(request, 'stats/probit.html', {})

def probit_input(request):
    if request.method == "POST":
        form = StatsInputForm(request.POST)
        if form.is_valid():
            result = form.save(commit=False)
            result.save()
            inputs = Code.objects.all()  # We can be more clever
            (beta_not, var_beta, iterrs, burn) = eval(inputs[len(inputs)-1].title) # Inputs
            (X,y,latent_y) = bayesian_probit.create_data()
            (plots, text) = bayesian_probit.full_gibbs(X,y,iterrs=iterrs, beta_not=beta_not, var_beta=var_beta, burn=burn)
            script, div = components(plots, CDN)
            return render(request, 'stats/probit_input.html', {'form':form,'script':script,'div':div,'text':text,})
    else:
        form = StatsInputForm(initial={'title': "[3,10], [1,1], 200, 100"})
    return render(request, 'stats/probit_input.html', {'form':form,})

def sensitivity(request):
    if request.method == "POST":
        form = StatsInputForm(request.POST)
        if form.is_valid():
            result = form.save(commit=False)
            result.save()
            inputs = Code.objects.all()  # We can be more clever
            (list_of_priors, list_of_var_beta ,iterrs, burn) = eval(inputs[len(inputs)-1].title) # Inputs
            plots = bayesian_probit.prior_sens(list_of_priors, list_of_var_beta, iterrs=iterrs, burn=burn)
            script, div = components(plots, CDN)
            return render(request, 'stats/sensitivity.html', {'form':form,'script':script,'div':div,})
    else:
        form = StatsInputForm(initial={'title': '[ [3,10], [5,12], [30,100] ],  [ [10,10], [10,10], [10,10] ], 200, 100'})
    return render(request, 'stats/sensitivity.html', {'form':form,})

def actual(request):
    if request.method == "POST":
        form = StatsInputForm(request.POST)
        if form.is_valid():
            result = form.save(commit=False)
            result.save()
            inputs = Code.objects.all()  # We can be more clever
            (beta_not, var_beta, iterrs, burn) = eval(inputs[len(inputs)-1].title) # Inputs
            (X,y) = bayesian_probit.load_data()
            (plots, text) = bayesian_probit.full_gibbs(X,y,iterrs=iterrs,burn=burn,beta_not=beta_not,var_beta=var_beta)
            script, div = components(plots, CDN)
            return render(request, 'stats/actual.html', {'form':form,'script':script,'div':div,'text':text,})
    else:
        form = StatsInputForm(initial={'title': "[0,0,0,0,0], [200,200,200,200,200], 200, 100"})
    return render(request, 'stats/actual.html', {'form':form,})

def sensitivity_actual(request):
    if request.method == "POST":
        form = StatsInputForm(request.POST)
        if form.is_valid():
            result = form.save(commit=False)
            result.save()
            inputs = Code.objects.all()  # We can be more clever
            (list_of_priors, list_of_var_beta ,iterrs, burn) = eval(inputs[len(inputs)-1].title) # Inputs
            plots = bayesian_probit.prior_sens(list_of_priors, list_of_var_beta, iterrs=iterrs, burn=burn)
            script, div = components(plots, CDN)
            return render(request, 'stats/sensitivity_actual.html', {'form':form,'script':script,'div':div,})
    else:
        form = StatsInputForm(initial={'title': '[ [0,0,0,0,0], [5,5,5,5,5], [10,10,10,10,10] ],  [ [100,100,100,100,100], [.1,.1,.1,.1,.1], [10,10,10,10,10] ], 200, 100'})
    return render(request, 'stats/sensitivity_actual.html', {'form':form,})
