from django.urls import path

from . import views

app_name = 'ml'
urlpatterns = [
    path('', views.index, name='index'),
    path('results/', views.results, name='results'),
    path('confirm/', views.confirm, name='confirm'),
    path('train/', views.train, name='train'),

 ]