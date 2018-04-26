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
## WARNING: in the DAS article the first dataset (144 pages) has duplicated pages... The new correct correspoding dataset has 111 pages. It does not change the comparison between tools (new results will be published soon)

in resources/DAS_2018, you will find files listed below. 

Images and pagexml can be also found  at https://zenodo.org/record/1226879#.WuFitE6g9PY


The show_shape.py utility will help you understanding the content of each pickle file. 

The data is provided both as Transkribus PageXml files for consumption by TranskribusDU tools and as Numpy data.

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

mkdir abp_CV
python $SRC/tasks/DU_ABPTable.py  --fold abp --fold-init 4  abp_CV  das_abp_models
#### BIESO evaluation with the 4-folds cross-validation
> python $SRC/tasks/DU_ABPTable.py --fold-run 1  das_abp_models abp_CV --crf-max_iter=1500
> python $SRC/tasks/DU_ABPTable.py --fold-run 2  das_abp_models abp_CV --crf-max_iter=1500
> python $SRC/tasks/DU_ABPTable.py --fold-run 3  das_abp_models abp_CV --crf-max_iter=1500
> python $SRC/tasks/DU_ABPTable.py --fold-run 4  das_abp_models abp_CV --crf-max_iter=1500
> python $SRC/tasks/DU_ABPTable.py --fold-finish das_abp_models abp_CV

#### BIESO evaluation on test collection
> python $SRC/tasks/DU_ABPTable.py   das_abp_models abp_full --tst abp_DAS_col9142_CRF

#### Applying full model after automatic textline recognition
> python $SRC/tasks/DU_ABPTable.py das_abp_models abp_full --run abp_DAS_col9142_workflow


### GCN, ECN and Logit-1conv

First of all, the code is located in src/gcn folder but it has developed under Python 3.
Then, our main dependencies are the following:
    scikit-learn==0.19.1
    scipy==1.0.0
    tensorflow==1.4.0


#### Training

    The main command is DAS_exp.py where we pass the data directory, a fold and a configid corresponding to a particular model
    architecture.

    #!/usr/bin/env bash

    dpath='../../usecases/ABP/resources/DAS_2018/'
    for fold in 1 2 3 4
     do
        #for configid in 0 1 5 33 44
         do
           python DAS_exp.py --dpath=$dpath --fold=$fold --configid=$configid --out_dir=out_das
        done
    done

#### BIESO evaluation with the 4-folds cross-validation
    In src/gcn
    python print_res.py out_das/
    This will print the results for the configurations/models found in the directory.
    The first four columns correspond to the Fold and the last one is the average of the four folds.

    C0 0.4576 0.4495 0.4319 0.4371 0.4440\\

    C1 0.6409 0.6249 0.6342 0.3043 0.5511\\

    ...

    C5 0.9644 0.9463 0.9261 0.9117 0.9371\\

    C33 0.9669 0.9184 0.9135 0.9138 0.9282\\

    C0=Logit Model; C1=Logit-1Conv ; C5=3Layers-10Conv ; C33=8Layers-1Conv

### Table Row Evaluation
TODO @Hervé





