from django.shortcuts import get_object_or_404, render
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.views import generic
from django.utils import timezone
import csv
import codecs
import logging, sys
from io import TextIOWrapper
import pandas
from .models import Dataset, Label


#index page
def index(request):
    return render(request, 'ml/index.html', locals())

#csv showing and editing page
def results(request):
	#reads CSV file sent from index
    if request.POST and request.FILES:
        name = request.FILES['csv_file'].name
        csvfile = TextIOWrapper(request.FILES['csv_file'].file, encoding='utf-8')
        reader = pandas.read_csv(csvfile, delimiter = ',')
        
        #get first row as labels
        labels = list(reader)
        nlabels = len(labels)
        
        #rest of rows as values
        samples = reader.values
        nsamples= len(reader)
        
        #save dataset in db
        dst = Dataset(dataset_name = name, labels_number = nlabels, samples_number = nsamples)
        dst.save()

        #save labels in db
        for label in labels: 
        	smpls = list(reader[label].values)
        	lbl = Label(dataset = dst, label_name = label, samples = smpls)
        	lbl.save()

        #list of labels with content	
        labels_content = []
        labels_content = Label.objects.filter(dataset_id=dst.id).order_by('pk')

        context = {'labels': labels, 'labels_content': labels_content, 'dataset': dst, 'samples': samples}

    return render(request, 'ml/results.html', context)