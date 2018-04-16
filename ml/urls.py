from django.urls import path
from django.views.generic import TemplateView

from . import views

app_name = 'ml'
urlpatterns = [
    path('', views.index, name='index'),
    path('results/', views.results, name='results'),
    path('confirm/', views.confirm, name='confirm'),
    path('train/', views.train, name='train'),
	path('oi/', TemplateView.as_view(template_name="oi.html"),
                   name='oi'),
 ]