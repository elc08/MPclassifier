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

### MSclassifier object

```
class MSclassifier.linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
```



### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.
