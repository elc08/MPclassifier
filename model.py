#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 14:12:31 2020

@author: elatorre
"""
import copy
import pandas as pd
import numpy as np
import random

import plotly.express as px
import plotly.graph_objects as go

from sklearn.svm import SVC
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def train_set(mut_class):
    data=copy.copy(mut_class.data)
    # Split the dataset into train and test according to the training column
    X_train=data[data['training']==1]
    # Create the labels vector for train and test sets
    y_train=X_train['class']
    # Filter only the columns corresponding to SNP and SV signatures
    X_train=np.asarray(X_train[mut_class.model.features])

    return X_train , y_train ;

def fit (mut_class,model):

    data=mut_class.data.copy()
    random.seed(123)
    if model == 'lasso' or model == 'neural':


        def warn(*args, **kwargs):
            pass
        import warnings
        warnings.warn = warn


        # Split the dataset into train and test according to the training column
        X_train=data[data['training']==1]

        # Create the labels vector for train and test sets
        y_train=X_train['class']

        # Filter only the columns corresponding to SNP and SV signatures
        X_train=np.asarray(X_train[mut_class.model.features])
        if model == 'lasso':
            # Build the Lasso model
            lasso=Lasso(random_state=1)
            parameters={'alpha':[100,10,1e-0,1e-1,1e-2,1e-4,1e-8,1e-16]}
            mut_class.model.classifier=GridSearchCV(lasso, parameters,scoring='neg_mean_squared_error',cv=5)
            mut_class.model.classifier.fit(X_train,y_train)
            print("Best: %f using %s" % (mut_class.model.classifier.best_score_, mut_class.model.classifier.best_params_))
        elif model =='neural':
            from sklearn.neural_network import MLPRegressor
            parameters= {'alpha':[100,10,1e-0,1e-1,1e-2,1e-4,1e-8,1e-16],
                         'activation':['relu','tanh','logistic'],
                         'hidden_layer_sizes':[(X_train.shape[1]+5, X_train.shape[1]+5), (X_train.shape[1]+5,10),(X_train.shape[1]),()],
                         'solver':['adam','lbfgs','sgd']}
            neural = MLPRegressor( random_state=1)
            mut_class.model.classifier=GridSearchCV(neural, parameters,scoring='neg_mean_squared_error',cv=5)
            mut_class.model.classifier.fit(X_train,y_train)
            print("Best: %f using %s" % (mut_class.model.classifier.best_score_, mut_class.model.classifier.best_params_))

    else :
        mut_class.model.classifier = model

    X=np.asarray(data[data['class']!=0][mut_class.model.features])
    prediction=mut_class.model.classifier.predict(X).ravel()
    svm_train=np.zeros((prediction.shape[0],2))
    svm_train[:,0]=prediction

    y=np.asarray(data[data['class']!=0]['class'])

    clf = SVC(kernel='linear')
    clf.fit(svm_train, y)
    mut_class.model.svm= clf

    w = clf.coef_[0]
    x_margin = -clf.intercept_[0]/w[0]
    mut_class.model.margin = x_margin



def test (mut_class):
        test_class=copy.deepcopy(mut_class.data)
        test_class=test_class[(test_class['training']==0)&(test_class['class']!=0)]
        X=np.asarray(test_class[mut_class.model.features])
        test_class['prediction']=mut_class.model.classifier.predict(X)

        # Labeling samples according to their SVM binary classification as either proficient or defficient

        # We need to create svm_pred, a 2-d array with the nn prediction on the x axis and 0s on the y axis
        svm_pred=np.zeros((test_class.shape[0], 2))
        svm_pred[:,0]=np.asarray(test_class['prediction'])

        # Then feed svm_pred into the trained SVM model and predict the outcome of each sample
        svm_pred=pd.DataFrame(mut_class.model.svm.predict(svm_pred))

        # Finally we label them in terms of proficiency or deficiency
        svm_pred=svm_pred.replace(-1,'Proficient')
        svm_pred=svm_pred.replace(1,'Deficient')

        # Append the prediction to the dataset
        test_class=test_class.reset_index()
        test_class['svm prediction']=svm_pred

        #fig = px.scatter(test_class.data, y="prediction", x="Sample type",color='Sample type')

        fig = px.strip(test_class, y="prediction", x='class',color='Sample type', hover_data=['sample'])
        fig =fig.add_trace(go.Scatter(
            x=[-1.5, 1.5],
            y=[mut_class.model.margin,mut_class.model.margin],
            name='Margin',
            mode='lines',
            line=dict(color='orange', width=3, dash='dash')
            ))
        fig.show()


def predict (mut_class) :
    # Use the model to predict the value of HR deficiency and add
    # append it as a column to original dataset
    X=np.asarray(mut_class.data[mut_class.model.features])
    mut_class.data['prediction']=mut_class.model.classifier.predict(X)

    # Labeling samples according to their SVM binary classification as either proficient or defficient

    # We need to create svm_pred, a 2-d array with the nn prediction on the x axis and 0s on the y axis
    svm_pred=np.zeros((mut_class.data.shape[0], 2))
    svm_pred[:,0]=np.asarray(mut_class.data['prediction'])

    # Then feed svm_pred into the trained SVM model and predict the outcome of each sample
    svm_pred=pd.DataFrame(mut_class.model.svm.predict(svm_pred))

    # Finally we label them in terms of proficiency or deficiency
    svm_pred=svm_pred.replace(-1,'Proficient')
    svm_pred=svm_pred.replace(1,'Deficient')
    #svm_pred.index += 1

    # Append the prediction to the dataset
    mut_class.data['SVM prediction']=svm_pred

def plot_regression (mut_class):
    if mut_class.accuracy==None:
        plot_title=f'MSclassifier model prediction.'
    else :
        plot_title=f'MSclassifier model prediction. Model accuracy: {mut_class.accuracy}'
    accuracy=mut_class.accuracy
    # Plotting the regression prediction together with SVM margin maximizer
    plot=mut_class.data.sort_values(by ='prediction' )
    plot['counter'] = range(len(plot))

    fig = px.strip(plot, y="prediction", x='counter', color='Sample type', hover_data=['sample'])
    fig.update_traces(marker=dict(size=5))
    fig = fig.update_layout(
            title=plot_title,
            xaxis_title="Samples",
            yaxis_title="Prediction"
            )

    fig =fig.add_trace(go.Scatter(
            x=[0, max(plot['counter'])],
            y=[mut_class.model.margin,mut_class.model.margin],
            name='Margin',
            mode='lines',
            line=dict(color='orange', width=3, dash='dash')
            ))
    return fig;


def plot_confusion (mut_class):
    # Create a confusion matrix containing only the ground truth sample type and the SVM regression labeling
    confusion=mut_class.data[['Sample type','SVM prediction']]
    # Filter all samples of previously unknown type
    confusion=confusion[~confusion['Sample type'].isin(['Unknown'])]

    # Compute the confusion matrix of our classifier
    #confusion_matrix(confusion['Sample type'],confusion['SVM prediction'],labels=['Proficient','Deficient'])

    accuracy=classification_report(confusion['Sample type'],confusion['SVM prediction'], target_names=['Proficient','Deficient'],output_dict=True)['accuracy']
    report=classification_report(confusion['Sample type'],confusion['SVM prediction'], target_names=['Proficient','Deficient'])
    return report, accuracy;

def ROC(mut_class):
    from sklearn.metrics import roc_curve, auc
    y_true=mut_class.data[mut_class.data['Sample type']!= 'Unknown'].replace({'Deficient': 1, 'Proficient': 0})['Sample type']
    y_score=mut_class.data[mut_class.data['Sample type']!= 'Unknown']['prediction']

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    roc= pd.DataFrame({'fpr':fpr, 'tpr':tpr})

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0,1],
        y=[0,1],
        fill=None,
        line=dict(color='orange', width=3, dash='dash')
        ))

    fig.add_trace(go.Scatter(x=roc['fpr'], y=roc['tpr'],
                            fill='tonexty',
                            mode='lines',
                            name='ROC curve'))


    fig = fig.update_layout(
            title=f'Rreciever operating characteristic  plot. AUC= {roc_auc}',
            xaxis_title="False positive rate",
            yaxis_title="True positive rate"
            )

    return fig , roc_auc


def importances (mut_class):
    from sklearn.inspection import permutation_importance
    X_train, y_train = train_set(mut_class)
    importances = permutation_importance(mut_class.model.classifier,X_train,y_train,n_repeats=50,scoring='r2')
    importances=pd.DataFrame(importances.importances_mean,columns=['importance'])
    importances.index=mut_class.model.features
    return importances.sort_values(by='importance',ascending=False)
