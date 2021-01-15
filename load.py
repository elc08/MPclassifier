#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:42:54 2020

@author: elatorre
"""
from SigProfilerMatrixGenerator.scripts import SigProfilerMatrixGeneratorFunc as matGen
import pandas as pd
import numpy as np
import glob

def load(mut_class):
    # Extract mutational matrices via SigProfiler
    matrices = matGen.SigProfilerMatrixGeneratorFunc(project=mut_class.project_name, genome=mut_class.reference_genome, vcfFiles=mut_class.vcf, exome=mut_class.exome)
    return


def load_names(mut_class):
    # Crete a list of all sample names.
    sample_names = glob.glob(mut_class.vcf + "/*.vcf")
    sample_names = [sample.split('/')[-1] for sample in sample_names]
    sample_names = [sample.split('.')[0] for sample in sample_names]

    return sample_names


def dataset(mut_class):
    # Create a dataframe with all participants, together with a class column
    # denoting wether each sample is proficient or deficient.
    # Also, append a column initialized with zeros to determine if samples
    # will be used for training.

    # Initialize dataframe with the list of samples
    data = pd.DataFrame(mut_class.sample_names, columns=['sample'])
    # Initialize class column.
    data['class'] = np.zeros(data.shape[0])

    if mut_class.positive is not None:  # check if there exist positive samples
        deficient = pd.read_csv(mut_class.positive, sep=' ', names=['sample'])
        # locate rows in data corresponding to deficient samples
        # and assign class = 1 in those rows
        data.loc[data['sample'].isin(deficient['sample']), 'class'] = 1

    if mut_class.negative is not None:  # check if there exist negative samples
        proficient = pd.read_csv(mut_class.negative, sep=' ', names=['sample'])

        # locate rows in data corresponding to proficient samples
        # and assign class = -1 in those rows
        data.loc[data['sample'].isin(proficient['sample']), 'class'] = -1

    # Add a column with the class names
    clas = data['class']
    clas = clas.replace(0, 'Unknown')
    clas = clas.replace(1, 'Deficient')
    clas = clas.replace(-1, 'Proficient')
    data['Sample type'] = clas

    # We add a binary column to denote samples that will be used for training.
    data['training'] = np.zeros(data.shape[0])

    return data
