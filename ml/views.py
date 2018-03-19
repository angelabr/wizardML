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
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier




#index page
def index(request):
    return render(request, 'ml/index.html', locals())

#csv showing and editing page
def results(request):
    #reads CSV file sent from index
    if request.POST and request.FILES:
        name = request.FILES['csv_file'].name
        file = request.FILES['csv_file']
        #csvfile = TextIOWrapper(request.FILES['csv_file'].file, encoding='utf-8')

        reader = pandas.read_csv(file)
########################################################
        reviews = []
        for i in reader['quality']:
            if i >= 3 and i <= 5:
                reviews.append('1')
            elif i >= 6 and i <= 6:
                reviews.append('2')
            elif i >= 7 and i <= 8:
                reviews.append('3')
        reader['Reviews'] = reviews
########################################################
        #get first row as labels
        labels = list(reader)
        nlabels = len(labels)
        
        #rest of rows as values
        samples = reader.values
        nsamples= len(reader)*nlabels
  
        #raw data into session
        request.session['dstfile'] = reader.to_json(orient ='records')

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

        y = total[[request.session['target']]]
        sub = total.drop(request.session['target'], 1)
        if request.session['ignored']:
            for ig in request.session['ignored']:
                sub = sub.drop(ig, 1)
        x = sub

        sc = StandardScaler()
        x = sc.fit_transform(x)


        x2_train, x2_test, y2_train, y2_test = train_test_split(sub, y, test_size = ((Dataset.objects.get(id = request.session['dataset']).test_percent)/100))

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = ((Dataset.objects.get(id = request.session['dataset']).test_percent)/100))

        model = SVC(probability = True)
        model.fit(x_train, y_train) 

        model2 = SVC(probability = True)
        model2.fit(x2_train, y2_train) 
        
        
        predict_data = model.predict(x_test)

        #print confusion matrix and accuracy score
        predict = accuracy_score(y_test, predict_data)*100
        print(predict*100)
        #print(X_test)
        #print(predict_data)

        #predict_data = model.predict(numpy.array([[8.1, 0.56, 0.28, 1.7, 0.368, 16.0, 56.0, 0.9968, 3.11, 1.28, 9.3]]))
        print(x2_test)
        # 8.1 0.56    0.28    1.7 0.368   16.0    56.0    0.9968  3.11    1.28    9.3 5
        #acc = accuracy_score(y_test, predict_data)
        #print (acc)

        features = []
        label_ids = request.session['labels'] 
        for ids in label_ids:
           lbl = Label.objects.get(id=ids)  
           features.append(lbl.label_name)
        explainer = lime.lime_tabular.LimeTabularExplainer(x2_test, feature_names=features, class_names=request.session['target'],discretize_continuous=False)

        i = numpy.random.randint(0, len(x2_test))
        exp = explainer.explain_instance(x2_test.iloc[i], model2.predict_proba, num_features=Dataset.objects.get(id = request.session['dataset']).labels_number-1)
        print(exp.as_list())

        context = {'accuracy': predict, 'predict_data': predict_data, 'test_data': x2_test.values, 'explanation': exp.as_list()}

    return render(request, 'ml/train.html', context)