from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, HttpResponseRedirect
from django.core.urlresolvers import reverse
import numpy as np

from .models import Code
from .forms import StatsInputForm

from modules.mymodule import hw,hw3

from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.resources import CDN




def index(request):
    return render(request, 'stats/index.html', {})


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
