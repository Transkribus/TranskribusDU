# Resources associated to our DAS 2018 Paper (under review)

__"Comparing Machine Learning Approaches for Table Recognition in Historical Register Books"__

  Stéphane Clinchant
  Hervé Déjean
  Jean-Luc Meunier
  Naver Labs Europe
  Meylan, France
  firstname.lastname@naverlabs.com 
 
  Eva Lang
  Bischöfliches Ordinariat Passau
  Archiv des Bistums Passau
  Passau, Germany
  eva.lang@bistum-passau.de

  Florian Kleber
  TU Wien
  Computer Vision Lab	 
  Vienna, Austria
  kleber@cvl.tuwien.ac.at


Here, we explain how to reproduce our experiments or do your own experiments on the same data.

# Data

in resources/DAS_2018, you will find files listed below. 

The show_shape.py utility will help you understanding the content of each file. 

The data is provided both as Transkribus XML files for consumption by TranskribusDU tools and as Numpy data.

##Training data
As XML files:
- __abp.tar__ : train collection, in TranskribusDU XML format
- __das_abp_models_fold_def.tar__ : folds definition, ready for use by TranskribusDU tools.

As numpy (list of) arrays (gzipped cPickle files):
- __abp_DAS_CRF_X.pkl__   
- __abp_DAS_CRF_Y.pkl__
- __abp_DAS_CRF_Xr.pkl__ : edges have been reversed (features adapted accordingly)
- fold I for I in [1,4]: __abp_CV_fold___<I>___tlXrlY_trn.pkl__mkdir DAS

NOTE: an X is a tuple (NF, E, EF), where NF is the node feature matrix, E is the edge definition, EF is the edge feature matrix.

##Test data
Where textlines were manually defined:
- __abp_DAS_col9142.tar__ : test collection, in TranskribusDU XML format

As numpy arrays, X, Y and X-reversed:
- __abp_DAS_col9142_CRF_X.pkl__
- __DAS_col9142_l_Y_GT.pkl__  
- __abp_DAS_col9142_CRF_Xr.pkl__   (reversed edges)

##Test data produced by automatic textline recognition
Here, textlines were automatically idenfitified and we do not have any manual annotation for them. So the only possible evaluation is on the final task: table line accuracy.

As XML files:
- abp_DAS_col9142_workflow.tar

As Numpy arrays
- __abp_DAS_col9142_workflow_X.pkl__   
- __abp_DAS_col9142_workflow_Xr.pkl__    (reversed edges)

# Reproducing our Experiments

We ran our experiments with those version of software:
> python -c 'import numpy; print numpy.__version__'
1.13.3
> python -c 'import scipy; print scipy.__version__'
1.0.0
> python --version
Python 2.7.14
> python -c 'import cvxopt; print cvxopt.__version__'
1.1.9

## Training & Testing

### CRF and Logit-standard

#### Training a full model:
> python $SRC/tasks/DU_ABPTable_Quantile.py   das_abp_models abp_full   --trn abp --crf-max_iter=1500

#### BIESO evaluation with the 4-folds cross-validation
> python $SRC/tasks/DU_ABPTable_Quantile.py --fold-run 1  das_abp_models abp_CV --crf-max_iter=1500 
> python $SRC/tasks/DU_ABPTable_Quantile.py --fold-run 2  das_abp_models abp_CV --crf-max_iter=1500 
> python $SRC/tasks/DU_ABPTable_Quantile.py --fold-run 3  das_abp_models abp_CV --crf-max_iter=1500 
> python $SRC/tasks/DU_ABPTable_Quantile.py --fold-run 4  das_abp_models abp_CV --crf-max_iter=1500 
> python $SRC/tasks/DU_ABPTable_Quantile.py --fold-finish das_abp_models abp_CV  

#### BIESO evaluation on test collection
> python $SRC/tasks/DU_ABPTable_Quantile.py   das_abp_models abp_full --tst abp_DAS_col9142_CRF

#### Applying full model after automatic textline recognition
> python $SRC/tasks/DU_ABPTable_Quantile.py das_abp_models abp_full --run abp_DAS_col9142_workflow


### GCN, ECN and Logit-1conv

TODO @Stéphane

#### Training
 
#### BIESO evaluation with the 4-folds cross-validation

### Table Row Evaluation

TODO @Hervé





