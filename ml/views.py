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
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import numpy




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
        nsamples= len(reader)*nlabels
        
        #save dataset in db
        dst = Dataset(dataset_name = name, labels_number = nlabels, samples_number = nsamples)
        dst.save()
        request.session['dataset'] = dst.id
        label_ids = []

        #save labels in db
        for label in labels: 
            smpls = list(reader[label].values)
            lbl = Label(dataset = dst, label_name = label, samples = smpls)
            lbl.save()
            label_ids.append(lbl.id)            

        #list of labels with content
        request.session['labels'] = label_ids
        labels_content = []
        labels_content = Label.objects.filter(dataset_id=dst.id).order_by('pk')

        context = {'labels': labels, 'labels_content': labels_content, 'dataset': dst, 'samples': samples}

        return render(request, 'ml/results.html', context)

#confirm page
def confirm(request):
    dataset_id = request.session['dataset']
    label_ids = request.session['labels'] 
    ignored = []
    #get data from form
    if request.POST:
        #labels to ignore
        for ids in label_ids:
            if str(ids) in request.POST.getlist('ignore'):
                lbl = Label.objects.get(id=ids)
                lbl.use_as = 'IG' 
                lbl.save()
                ignored.append(lbl.id)

        request.session['ignored'] = ignored
        #targeted label        
        ids = request.POST.get('target')
        lbl = Label.objects.get(id=ids)
        lbl.use_as = 'OU' 
        lbl.save()
        request.session['target'] = lbl.id
        #get percentages
        percs = request.POST.get('percentages')
        dst = Dataset.objects.get(id = dataset_id)

        if ('na' not in percs) and ('60' not in percs):
            percs = percs.split("-")
            dst.train_percent = int(percs[0])
            dst.test_percent = int(percs[1])
            dst.eval_percent = int(percs[2])
        #get nature of dataset    
        dst.nature = request.POST.get('purpose')     
        dst.save()

    context = {'dataset': dst}
    return render(request, 'ml/confirm.html', context)

#train page
def train(request):
    dataset_id = request.session['dataset']
    label_ids = request.session['labels']
    X = []
    if request.POST:
        X = Label.objects.get(id = request.session['target']).samples
        y = []
        for ids in label_ids:
            if ids != request.session['target']:
                lbl = Label.objects.get(id=ids)
                y.append(lbl.samples)

    #x, x_test, y, y_test = train_test_split(xtrain,labels,test_size=0.2,train_size=0.8)
    #x_train, x_cv, y_train, y_cv = train_test_split(x,y,test_size = 0.25,train_size =0.75)

    a = numpy.array(X)
    b = numpy.array(y)
    dst = Dataset.objects.get(id = dataset_id)
    #a.reshape((int(dst.samples_number/len(label_ids)),1))
    a = a.reshape((1599, 1))
    b = b. reshape((1599, 11))
    print(a)
    print(b)
    #TODO: double array for numbers and text, put this into numbers
    X_train, X_test, y_train, y_test = train_test_split(a, b, test_size = (Dataset.objects.get(id = dataset_id).test_percent)/100)
    #model = LinearSVC()
    #model.fit(X_train, y_train)
    #predict_data = model.predict(y_test)
    #print (predict_data)


    return render(request, 'ml/train.html', locals())