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
import six
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as opy
import pickle


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
        labels_content = Label.objects.filter(dataset_id=dst.id).order_by('pk')
        ordered = []
        for l in labels_content:
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

    context = {'dataset': dst}
    return render(request, 'ml/confirm.html', context)

#train page
def train(request):

    if request.POST:
        #get the whole dataset raw for easier processing
        total = pandas.read_json(request.session['dstfile'], orient='records')

        #y for labels, x for features
        y = total[[request.session['target']]]
        sub = total.drop(request.session['target'], 1)
        if request.session['ignored']:
            for ig in request.session['ignored']:
                sub = sub.drop(ig, 1)
        x = sub.sort_index(axis=1)
        x = sub      
 
        #datas
        x2_train, x2_test, y2_train, y2_test = train_test_split(x, y, test_size = ((Dataset.objects.get(id = request.session['dataset']).test_percent)/100))
        #models
        model2 = SVC(probability = True)
        model2.fit(x2_train, y2_train) 
        
        #testing and predictions
        predict_data = model2.predict(x2_test)
        predict = accuracy_score(y2_test, predict_data)*100

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

        #labels for results analysis
        #IF DATA HAS BEEN INTRODUCED MANUALLY
        if request.POST.get(list(sub)[0]):
            for l in list(sub):
                analyse.append(float(request.POST.get(l)))

            #sc = StandardScaler()
            result = model2.predict(numpy.array([analyse]))
            proba = model2.predict_proba(numpy.array([analyse]))

            #LIME explanator
            explainer = lime.lime_tabular.LimeTabularExplainer(x2_train, feature_names=features, class_names=request.session['target'], discretize_continuous=False)
            df = pandas.Series(list(analyse), list(sub), name='001')
            exp2 = explainer.explain_instance(df, model2.predict_proba, num_features=len(list(sub)))
            fig = exp2.as_pyplot_figure()
            tmp = six.StringIO()
            fig.savefig(tmp, format='svg', bbox_inches='tight')

            #Labels probabilities
            targets = set(numpy.array(total[request.session['target']]))
            data = [go.Bar(x=list(proba.flatten()),y=list(targets),orientation = 'h')]
            layout=go.Layout(title="Probabilities", xaxis={'title':'Percentage'}, yaxis={'title': request.session['target']})
            fig2 = go.Figure(data=data, layout=layout)
            div = opy.plot(fig2, auto_open=False, output_type='div')
            way = 1

        if request.FILES:
            file = request.FILES['csv_evaluate']
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

            y = reader[[request.session['target']]]
            sub = reader.drop(request.session['target'], 1)
            if request.session['ignored']:
                for ig in request.session['ignored']:
                    sub = sub.drop(ig, 1)
            sub = sub.sort_index(axis=1)
            x = sub
            request.session['x'] = x.values.tolist()       
            result = model2.predict(x)
            proba = model2.predict_proba(x)
            request.session['proba'] = proba.tolist()
            request.session['result'] = result.tolist()
            way = 2
            x=x.values

        if request.POST.get('results'):
            
            x = request.session['x'][int(request.POST.get('results'))-1]
            proba = request.session['proba'][int(request.POST.get('results'))-1]
            result = model2.predict(numpy.array([x]))
            
            #LIME explanator
            explainer = lime.lime_tabular.LimeTabularExplainer(x2_train, feature_names=features, class_names=request.session['target'], discretize_continuous=False)
            df = pandas.Series(request.session['x'][int(request.POST.get('results'))-1], list(sub), name='001')
            exp2 = explainer.explain_instance(df, model2.predict_proba, num_features=len(list(sub)))
            fig = exp2.as_pyplot_figure()
            tmp = six.StringIO()
            fig.savefig(tmp, format='svg', bbox_inches='tight')

            #Labels probabilities
            targets = set(numpy.array(total[request.session['target']]))
            data = [go.Bar(x=request.session['proba'][int(request.POST.get('results'))-1],y=list(targets),orientation = 'h')]
            layout=go.Layout(title="Probabilities", xaxis={'title':'Percentage'}, yaxis={'title': request.session['target']})
            fig2 = go.Figure(data=data, layout=layout)
            div = opy.plot(fig2, auto_open=False, output_type='div')
        
        if request.POST.get('save_model'):
            filename = 'finalized_model.sav'
            pickle.dump(model2, open(filename, 'wb'))
            loaded_model = pickle.load(open(filename, 'rb'))

        if request.POST.get('save_results'):
            with open('results.csv', 'w', newline='') as csvfile:
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

        context = {'accuracy': predict, 'predict_data': predict_data, 'test_data': x2_test.values, 'svg': tmp.getvalue(), 'labels': list(sub), 'target': request.session['target'],
        'result': result, 'proba': proba, 'svg2': div, 'way': way, 'analyse': list(analyse), 'analyse_data': x }
    return render(request, 'ml/train.html', context)