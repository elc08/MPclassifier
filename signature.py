#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 13:34:08 2020

@author: elatorre
"""
import pandas as pd
import random

from sigproextractor import sigpro as sig
from scipy.optimize import nnls
from sklearn import preprocessing


def train(mut_class, ratio, end):
    random.seed(123)

    end_name = '.exome' if mut_class.exome is True else '.all'

    # restart training counter to 0 (needed in case you retrain a dataset)
    mut_class.data['training'] = 0

    # Locate deficient and proficient samples and select a random fraction using ratio
    training_def = pd.DataFrame(mut_class.data[(mut_class.data['Sample type']=='Deficient')].sample(frac=ratio,random_state=1)['sample'])
    training_pro = pd.DataFrame(mut_class.data[(mut_class.data['Sample type']=='Proficient')].sample(frac=ratio,random_state=1)['sample'])

    # Modify data.training column to 1 for all samples used for training
    mut_class.data.loc[mut_class.data['sample'].isin(training_def['sample']),'training']=1
    mut_class.data.loc[mut_class.data['sample'].isin(training_pro['sample']),'training']=1

    # Create 2 lists containing the names of all samples in the deficient and proficient training sets
    training_prof = list(mut_class.data[(mut_class.data.training==1) & (mut_class.data['Sample type']=='Proficient')]['sample'])
    training_def = list(mut_class.data[(mut_class.data.training==1) & (mut_class.data['Sample type']=='Deficient')]['sample'])

    # Given the count matrix of all samples and a list of samples to subset
    # This functions creates the matrix of all samples in the subset.
    def split_train(path, samples):
        matrix = pd.read_csv(path, sep='\t')
        train = pd.DataFrame(matrix['MutationType'])

        for sample in samples:
            train[sample] = matrix.filter(like=sample)
        return train

    # Extract the matrix of mutations for both proficient and deficient samples and merge
    signatures_all = []
    features = []
    for feature in mut_class.feature_list:
        # Create the path to the mutations matrix
        feature_mut = ''.join([i for i in feature if not i.isdigit()])
        feature_path = mut_class.vcf+'output/'+feature_mut+ '/' + mut_class.project_name+'.'+ feature+ end_name

        # Retrieve the samples for training for a mutation matrix
        matrix_train_prof=split_train(feature_path, training_prof)
        matrix_train_def=split_train(feature_path, training_def)

        # Export these matrices
        path_to_matrix_train_prof=feature_path + feature+ '_matrix_train_prof'
        path_to_matrix_train_def=feature_path + feature+ '_matrix_train_def'
        matrix_train_prof.to_csv(path_to_matrix_train_prof,sep='\t',index=False)
        matrix_train_def.to_csv(path_to_matrix_train_def,sep='\t',index=False)

        sig.sigProfilerExtractor("table", mut_class.vcf+'/output/'+feature_mut+ "/signatures_prof", path_to_matrix_train_prof, endProcess=end)
        sig.sigProfilerExtractor("table", mut_class.vcf+'/output/'+feature_mut+ "/signatures_def", path_to_matrix_train_def, endProcess=end)

        feature2 = feature
        if feature_mut == 'ID':
            feature2 = 'SBSINDEL'
        if feature_mut == 'DBS':
            feature2 = 'SBSDINUC'

        path_to_prof_signatures = mut_class.vcf+'/output/'+feature_mut+ "/signatures_prof/"+feature+'/Suggested_Solution/De_Novo_Solution/De_Novo_Solution_Signatures_' + feature2+'.txt'
        path_to_def_signatures = mut_class.vcf+'/output/'+feature_mut+ "/signatures_def/"+feature+'/Suggested_Solution/De_Novo_Solution/De_Novo_Solution_Signatures_' + feature2+'.txt'

        # Load the siganture
        signatures_prof=pd.read_csv(path_to_prof_signatures,sep='\t')
        signatures_def=pd.read_csv(path_to_def_signatures,sep='\t')

        # Define a useful function to rename signature names according to their mutation type and HR status
        def column_rename (data,mutation_type,HR_status):
            new_columns=['MutationsType']
            for i in range(1,data.shape[1]):
                new_columns.append(mutation_type+'_'+HR_status +'_' +str(i))
            data.columns=new_columns

        column_rename(signatures_prof,feature,'pro')
        column_rename(signatures_def,feature,'def')

        signatures=pd.merge(signatures_prof, signatures_def, on='MutationsType')
        signatures_all.append (signatures)
        features= features+ [k for k in signatures.columns]
        features.remove('MutationsType')

    return signatures_all , features

def fit(mut_class):

    end_name = '.exome' if mut_class.exome is True else '.all'

    # drop colums from the dataset if they are already present to prevent duplication of features
    mut_class.data = mut_class.data.drop([value for value in mut_class.data.columns if value in mut_class.model.features], axis=1)

    # fit the extracted signatures for each sample in the data
    for index, signatures in enumerate(mut_class.model.signatures):
        # load a signatures matrix
        signatures = signatures.set_index('MutationsType')

        # Extract the feature name and its associated signatures' names
        feature= signatures.columns[1].split('_')[0]
        feature_mut = ''.join([i for i in feature if not i.isdigit()])  # corresponds to the non-numerical part of the signature

        # read the mutation matrix of all samples
        matrix=pd.read_csv(mut_class.vcf + '/output/'+feature_mut+'/'+ mut_class.project_name + '.'+ feature +end_name, sep='\t')
        matrix= matrix.set_index('MutationType')

        projection= pd.DataFrame(0, index=matrix.columns, columns=signatures.columns)
        for col in matrix.columns:
            coef = nnls(signatures,matrix[col])[0]
            projection.loc[col]=preprocessing.scale(coef)

        mut_class.data=pd.merge(mut_class.data,projection,left_on='sample',right_index=True)

    # Normalize the signatures proportion accross each sample.
    #mut_class.data[mut_class.model.features]=mut_class.data[mut_class.model.features].div(mut_class.data[mut_class.model.features].sum(axis=1), axis=0)
