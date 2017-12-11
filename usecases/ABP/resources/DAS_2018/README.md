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

in resources/DAS_2018, you will find:

## Xml files
- __abp_DAS_col9142.tar__ : test collection, in TranskribusDU XML format
- __abp.tar__ : train collection, in TranskribusDU XML format
- __das_abp_models_fold_def.tar__ : folds definition, ready for use by TranskribusDU tools.

##Training data as gzipped cPickle files

Training data Xs
- __abp_DAS_CRF_X.pkl__   
- __abp_DAS_CRF_Y.pkl__
- __abp_DAS_CRF_Xr.pkl__ : edges have been reversed (features adapted accordingly)
- fold I for I in [1,4]: __abp_CV_fold___<I>___tlXrlY_trn.pkl__mkdir DAS

##Test data as gzipped cPickle files

Test data X and X-reversed:
- __abp_DAS_col9142_CRF_X.pkl__
- __abp_DAS_col9142_CRF_Xr.pkl__   (reversed edges)

Test data groundtruth labels:
- __DAS_col9142_l_Y_GT.pkl__  

Test data, where the text lines were found by a program (rather than by human annotation):
- __abp_DAS_col9142_workflow_X.pkl__   
- __abp_DAS_col9142_workflow_Xr.pkl__    (reversed edges)

