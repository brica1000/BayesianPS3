from django.conf.urls import url
from django.conf import settings
from django.conf.urls.static import static

from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^input_code/', views.edit_code, name='edit_code'),
    url(r'^results/', views.results, name='results'),
    url(r'^edit_gibbs/', views.edit_gibbs, name='edit_gibbs'),
    url(r'^gibbs_results/', views.gibbs_results, name='gibbs_results'),
    url(r'^probit/', views.probit, name='probit'),
    url(r'^probit_input/', views.probit_input, name='probit_input'),
    url(r'^sensitivity/', views.sensitivity, name='sensitivity'),
    url(r'^actual/', views.actual, name='actual'),

    ]
