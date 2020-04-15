#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:42:22 2020

@author: Eric Latorre Crespo
"""

from MSclassifier import load, signature, model
import pickle


class signature_model :
    def __init__ (self):
        pass


class signature_classifier :
    def __init__ (self, vcf, positive=None, negative=None,
                        project_name='MSclassifier', reference_genome='GRCh38', exome=False, feature_list=['SBS96','ID83','DBS78'],
                            model = signature_model()):
        self.vcf = vcf
        self.positive = positive
        self.negative= negative
        self.project_name = project_name
        self.reference_genome = reference_genome
        self.exome = exome
        self.feature_list = feature_list
        self.model = model                  # model will be of signature_model class

    def load_vcf (self):
        load.load(self)
        self.sample_names = load.load_names(self)
        self.data = load.dataset (self)

    def signature_train (self,ratio=0.7,end=3) :
        self.model.signatures , self.model.features  = signature.train(self,ratio,end)

    def signature_fit (self):
        signature.fit(self)

    def model_fit (self,model_type='neural'):
        #self.model.model_type = model_type
        model.fit(self,model_type)
        self.model.importances = model.importances(self)

    def test_check (self):
        model.test(self)


    def model_predict(self):
        model.predict(self)
        values= self.data['training']
        if not all(v == 0 for v in values):
            self.confusion_matrix , self.accuracy= model.plot_confusion(self)
            self.ROC_curve , self.AUC = model.ROC(self)
        else:
            self.accuracy=None
        self.plot = model.plot_regression (self)


    def export(self):
        pickle.dump(self , open( self.vcf+'/output/' + self.project_name+ ".p", "wb" ) )
