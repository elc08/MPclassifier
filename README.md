# MSclassifier

MSclassifier is a python package that automatizes the development of a classifier based on mutational signatures. Although this package was originaly developped to predict the Homologous Recombination (HR) status of high grade serous ovarian cancer (HGSOC), its applications extend far beyond. MSclassifier simplifies the process of extracting mutational signatures and training a neural network to produce a neural network together with a classification margin that can be easily exported and shared to fit and predict new data.

## Getting Started

### Installing MSclassifier
MSclassifier is currently available as a test python package in the pypi repository. To install use the following command:

```
python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps MSclassifier
```

### Prerequisites

MSclassifier relies on SigProfiler to create mutational matrices and extract mutational signatures. Both packages together with the apropriate reference genome need to be installed. The details on the installation and the full functionality of SigProfiler package suite can be found in :

https://osf.io/t6j7u/wiki/home/

https://osf.io/s93d5/wiki/home/

```
pip install SigProfilerMatrixGenerator
pip install sigproextractor

from SigProfilerMatrixGenerator import install as genInstall
genInstall.install('GRCh38', bash=True)
```
MSclassifier uses plotly to produce graphics and scikit to train a neural network.

```
pip install plotly==4.6.0
pip install -U scikit-learn

```

Further support on the installation of plotly can be found in:
https://pypi.org/project/plotly/

Further standard package prerequisits include:
  - pandas
  - numpy
  - scipy



## Overview of MSclassifier

MSclassifier relies on the detection of mutational signatures, that carry the footprint of mutational events found in the genomes, to train a grid of shallow regressor neural networks and produce a threshold for classification. 

A notebook including everything needed to get a project up and runing has been included.

[notebook-example](https://github.com/elc08/MSclassifier/blob/master/Introduction%20to%20MSclassifier.ipynb)

### MSclassifier object

```
class MSclassifier.signature_classifier : (vcf, positive=None, negative=None, project_name='MSclassifier', reference_genome='GRCh38', exome=False, feature_list=['SBS96','ID83','DBS78'],model = signature_model()):
```

Parameters:

- vcf : str
    path to a folder containing the all .vcf files.

- positive : str , Default = None
    path to a .txt file containing the list of all positive samples.

- negative : str , Default = None
    path to a .txt file containing the list of all negative samples.

- project_name : str , Default = 'MSclassifier'
    Project name that will be used for referencing throughout the project.

- model :  signature_model class, Default=None
    Model used to train or predict the output of the classifier.
    
- reference_genome : str in {‘GRCh38’, ‘GRCh37’,'GRCm38','GRCm37'} , Default = 'GRCh38'
    Genome reference used during the process of variant calling. reference_genome is only
    used as an argument for SigProfilerMatrixFunc, therefore admits all supported genomes in the package

- exome : boolean, Default = False
    option to filter vcf files to only retain variant calls present in the exome

feature_list : list, Default = ['SBS96','ID83','DBS78']
    List of any mutational profile in the output of SigProfilerMatrixFunc. These are the features that will be used to train the classifier.


## Authors

**Eric Latorre Crespo** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)
