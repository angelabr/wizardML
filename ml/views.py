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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy
from sklearn.metrics import mean_squared_error
import lime
import lime.lime_tabular
from sklearn.svm import SVR
from sklearn.svm import SVC
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
        
        #raw data into session
        request.session['dstfile'] = reader.to_json()

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
                ignored.append(lbl.label_name)
        #ignored into session        
        request.session['ignored'] = ignored
        #targeted label        
        ids = request.POST.get('target')
        lbl = Label.objects.get(id=ids)
        lbl.use_as = 'OU' 
        lbl.save()
        request.session['target'] = lbl.label_name
        #get percentages
        percs = request.POST.get('percentages')
        dst = Dataset.objects.get(id = dataset_id)
        #check percentages not default
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

    if request.POST:
        total = pandas.read_json(request.session['dstfile'], orient='records')
        X = total[[request.session['target']]]
        sub = total.drop(request.session['target'], 1)
        if request.session['ignored']:
            for ig in request.session['ignored']:
                sub = sub.drop(ig, 1)
        y = sub        

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = ((Dataset.objects.get(id = request.session['dataset']).test_percent)/100)*2)
    model = SVC(probability=True)
    #model = LogisticRegression()
    #model = SVR()
    model.fit(y_train, X_train)
    predict_data = model.predict(y_test)
    #print(predict_data)
    #predict_data = model.predict(numpy.array([[10.8, 0.45, 0.33, 2.5, 0.099, 20.0, 38.0, 0.99818, 3.24, 0.71, 10.8]]))
    #acc = mean_squared_error(X_test,predict_data)
    #print (acc*100)
    features=[]
    label_ids = request.session['labels'] 
    for ids in label_ids:
       lbl = Label.objects.get(id=ids)  
       features.append(lbl.label_name)
    #print (features)
    #predict = lambda x: predict_data.astype(float)     
    explainer = lime.lime_tabular.LimeTabularExplainer(y_test, feature_names=features, class_names=request.session['target'],discretize_continuous=False)
    i = numpy.random.randint(0, len(y_test))
    print(y_test.iloc[i])
    print(model.predict([y_test.iloc[i]]))
    exp = explainer.explain_instance(y_test.iloc[i], model.predict_proba, num_features=11)
    print(exp.as_list())

    return render(request, 'ml/train.html', locals())