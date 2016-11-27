from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, HttpResponseRedirect
from django.core.urlresolvers import reverse
import numpy as np

from .models import Code
from .forms import StatsInputForm

from modules.mymodule import hw,hw3


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
        hw3.process(beta_not = np.transpose(x))
        priors = x
        output = 'Wooo!'
        return render(request, 'stats/results.html', {'priors': priors, 'output': output})
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
        hw3.gibbs(p=p, iterr1=iterr1, iterr2=iterr2)
        output = 'Wooo!'
        return render(request, 'stats/gibbs_results.html', {'output': output,})
    else:
        return HttpResponseRedirect(reverse('edit_gibbs'))
