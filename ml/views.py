from django.shortcuts import get_object_or_404, render, redirect
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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
import six
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as opy
import pickle
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
import tkinter.filedialog
import time
from mttkinter import mtTkinter as tk
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
        # reviews = []
        # for i in reader['quality']:
        #     if i >= 3 and i <= 5:
        #         reviews.append('1')
        #     elif i >= 6 and i <= 6:
        #         reviews.append('2')
        #     elif i >= 7 and i <= 8:
        #         reviews.append('3')
        # reader['Reviews'] = reviews
########################################################
        #get first row as labels
        labels = list(reader)
        nlabels = len(labels)
        
        #rest of rows as values
        samples = []
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
            smpls.insert(0, label)
            lbl = Label(dataset = dst, label_name = label, samples = smpls)
            lbl.save()
            label_ids.append(lbl.id)   
            samples.append(smpls)         
        #list of labels with content
        labels_content = Label.objects.filter(dataset_id=dst.id).order_by('pk')
        ordered = []
        i = 0
        for l in labels_content:
            samples[i].insert(0, l.id)
            i+=1
            ordered.append(l.id)

        request.session['labels'] = ordered
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
    request.session['done'] = 0
    context = {'dataset': dst}
    return render(request, 'ml/confirm.html', context)

def oi(request):
    return render(request, 'ml/oi.html', locals())

#train page
def train(request):
    matplotlib.use('Agg')

    if request.POST or request.FILES:
        print(request)    
        #get the whole dataset raw for easier processing
        total = pandas.read_json(request.session['dstfile'], orient='records')

        #y for labels, x for features
        y = total[[request.session['target']]]
        sub = total.drop(request.session['target'], 1)
        if request.session['ignored']:
            for ig in request.session['ignored']:
                sub = sub.drop(ig, 1)
        x = sub.sort_index(axis=1)
        scaler = MinMaxScaler()
        x = scaler.fit_transform(x)
        x = pandas.DataFrame(x.tolist())

        x2_train, x2_test, y2_train, y2_test = train_test_split(x, y, test_size = ((Dataset.objects.get(id = request.session['dataset']).test_percent)/100))
        
        names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
         "Decision Tree", "Random Forest", "AdaBoost", "Logistic Regression",
         "Naive Bayes", "QDA"]

        classifiers = [
            KNeighborsClassifier(),
            SVC(kernel="linear", probability=True),
            SVC(kernel="rbf", probability=True),
            DecisionTreeClassifier(),
            RandomForestClassifier(n_jobs=1),
            AdaBoostClassifier(),
            LogisticRegression(),
            GaussianNB(),
            QuadraticDiscriminantAnalysis()
            ]
      #------------------hyperparameter tuning--------------------
       
        if request.FILES.get("model_file"):
            print ("aqui")
            model2 = pickle.load(request.FILES.get("model_file"))
            print(model2)
        else:
            #compare all the classifier, store their scores
            if request.session['done'] is not 1:
                tuned_parameters = [dict(n_neighbors=list(range(1, 31))),
                            {'C': [1, 10, 100, 1000]},
                            {'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                            {'n_estimators': [16, 32],'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]},
                            {'n_estimators': [200, 700], 'max_features': ['auto', 'sqrt', 'log2']},
                            {'n_estimators': [1,5,10,20,30,40,50,60,70,80,90,100], 'learning_rate': [0.001, 0.01, 0.1, 1] },
                            {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }]

                scs = []
                for name, clf in zip(names, classifiers):
                    clf.fit(x2_train, y2_train)
                    score = clf.score(x2_test, y2_test)
                    print("SCORE: "+str(score)+" NAME: "+name)
                    scs.append(score)
                #pick best classifier    
                max_value = max(scs)
                max_index = scs.index(max_value)
                request.session['max'] = max_index    
                    
                if max_index <= 6:
                    model2 = GridSearchCV(classifiers[max_index], tuned_parameters[max_index], cv=5)
                    model2.fit(x2_train, y2_train)
                    request.session['params'] = tuned_parameters[max_index]
                    print(model2.best_params_)

                else:
                    model2 = classifiers[max_index]
                    model2.fit(x2_train, y2_train)

                pickle.dump(model2, open('tmp_model.sav', 'wb'))    
                predict_data = model2.predict(x2_test)
                predict = model2.score(x2_test, y2_test)*100
                print(predict)
                request.session['predict'] = predict.tolist()
                request.session['predict_data'] = predict_data.tolist()
                request.session['done'] = 1    
            
            else:
                model2=pickle.load(open('tmp_model.sav', 'rb'))

        #Examples for testing
        #fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol,quality        
        #          7.0,             0.49,      0.49,           5.6,     0.06,               26.0,               121.0, 0.9974,3.34,   0.76,    10.5,     5
        #         11.2,             0.28,      0.56,           1.9,    0.075,               17.0,                60.0,  0.998,3.16,   0.58,     9.8,     6
        features = []
        label_ids = request.session['labels'] 

        #select labels that will be used
        for ids in label_ids:
           lbl = Label.objects.get(id=ids)
           if (lbl.label_name not in request.session['target']) and (lbl.label_name not in request.session['ignored']):   
                features.append(lbl.label_name)

        #some initializations        
        result, proba = 0, []        
        analyse = []
        tmp = six.StringIO()
        div = six.StringIO()
        way = 0
        div2 = 0
        #labels for results analysis
        #IF DATA HAS BEEN INTRODUCED MANUALLY
        if request.POST.get(list(sub)[0]):
            for l in list(sub):
                analyse.append(float(request.POST.get(l)))
            #sc = StandardScaler()
            scaler = MinMaxScaler()
            x = pandas.DataFrame(analyse)
            x = scaler.fit_transform(x)
            x2 = x.flatten()
            x = numpy.reshape(x,(1,len(x)))      
            x = pandas.DataFrame(x.tolist())
            result = model2.predict(x)
            proba = model2.predict_proba(x)
            targets = set(numpy.array(total[request.session['target']]))
            targets = ''.join(str(e) for e in targets)  
            explainer = lime.lime_tabular.LimeTabularExplainer(x2_train, feature_names=features, class_names=targets, discretize_continuous=False)
            df = pandas.Series(x2, list(sub), name='001')
            exp2 = explainer.explain_instance(df, model2.predict_proba, num_features=len(list(sub)), top_labels=1)
            exp2.save_to_file('ml/templates/oi.html')

            f = open("ml/templates/oi.html","r", encoding="utf8")
            lines = f.readlines()
            f.close()
            a = open("ml/templates/oi.html","w", encoding="utf8")
            for line in lines:
              if "templateSettings.interpolate" not in line:
                a.write(line)
                print(".")
            a.close()
            x = x2

        if request.FILES and request.POST.get('evaluate'):
            file = request.FILES['csv_evaluate']
            reader = pandas.read_csv(file)
    ########################################################
            # reviews = []
            # for i in reader['quality']:
            #     if i >= 3 and i <= 5:
            #         reviews.append('1')
            #     elif i >= 6 and i <= 6:
            #         reviews.append('2')
            #     elif i >= 7 and i <= 8:
            #         reviews.append('3')
            # reader['Reviews'] = reviews
    ########################################################
            sub = reader
            if request.session['target'] in list(reader):
                y = reader[[request.session['target']]]
                sub = reader.drop(request.session['target'], 1)
            if request.session['ignored']:
                for ig in request.session['ignored']:
                    sub = sub.drop(ig, 1)
            sub = sub.sort_index(axis=1)
            x = sub
            request.session['x'] = x.values.tolist()
            scaler = MinMaxScaler()
            x = scaler.fit_transform(x)
            x = pandas.DataFrame(x.tolist())
            request.session['xs'] = x.values.tolist()
            result = model2.predict(x)
            proba = model2.predict_proba(x)

            request.session['proba'] = proba.tolist()
            request.session['result'] = result.tolist()
            way = 2
            x = x.values

        if request.POST.get('results'):
            div2 = 1
            x = request.session['xs'][int(request.POST.get('results'))-1]
            x = numpy.array(x)
            x2 = x.tolist()
            x = numpy.reshape(x,(1,len(x)))      
            x = pandas.DataFrame(x)
            proba = request.session['proba'][int(request.POST.get('results'))-1]
            result = model2.predict(x)
            targets = set(numpy.array(total[request.session['target']]))
            targets = ''.join(str(e) for e in targets)  
                      #LIME explanator
            explainer = lime.lime_tabular.LimeTabularExplainer(x2_train, feature_names=features, class_names=targets, discretize_continuous=False)
            df = pandas.Series(x2, list(sub), name='001')
            exp2 = explainer.explain_instance(df, model2.predict_proba, num_features=len(list(sub)), top_labels=1)
            exp2.save_to_file('ml/templates/oi.html')

            f = open("ml/templates/oi.html","r", encoding="utf8")
            lines = f.readlines()
            f.close()
            a = open("ml/templates/oi.html","w", encoding="utf8")
            for line in lines:
              if "templateSettings.interpolate" not in line:
                a.write(line)
                print(".")
            a.close()
            x = x2

        if request.POST.get('save_model'):
            f = tk.filedialog.asksaveasfile(mode='wb', defaultextension=".sav")
            if f is not None: # asksaveasfile return `None` if dialog closed with "cancel".
                pickle.dump(model2, open(f.name, 'wb'))
                loaded_model = pickle.load(open(f.name, 'rb'))
                f.close()
      
        if request.POST.get('save_results'):
            f = tk.filedialog.asksaveasfile(mode='wb', defaultextension=".csv")
            if f is not None: # asksaveasfile return `None` if dialog closed with "cancel". 
                with open(f.name, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',',
                                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    lbls = list(sub)
                    lbls.append(request.session['target'])
                    writer.writerow(lbls)
                    i = 0
                    for sample in request.session['x']:
                        sample.append(request.session['result'][i])
                        writer.writerow(sample)
                        i += 1    
                    f.close()
                            
        context = {'accuracy': request.session['predict'], 'predict_data': request.session['predict_data'], 'test_data': x2_test.values, 'div': div2, 'labels': list(sub), 'target': request.session['target'],
        'result': result, 'proba': proba, 'way': way, 'analyse': list(analyse), 'analyse_data': x, 'name':names[request.session['max']]}
    return render(request, 'ml/train.html', context)

