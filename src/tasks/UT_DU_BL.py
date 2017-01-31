__author__ = 'sclincha'

import sys
sys.path.append('..')

import unittest



try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version





import crf.FeatureDefinition
#from crf.FeatureDefinition_PageXml_std import FeatureDefinition_PageXml_StandardOnes
from crf.FeatureDefinition_PageXml_FeatSelect import FeatureDefinition_PageXml_FeatSelect

import DU_BL_Task

import numpy as np
from IPython import embed
import sklearn
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV, RidgeClassifier
import pdb
from sklearn.metrics import accuracy_score,roc_auc_score
import scipy.sparse as sp







class UT_DU_BL(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(UT_DU_BL, self).__init__(*args, **kwargs)
        #Z=cPickle.load(open('UT_line_embedding.pickle','rb'))
        self.modelName='UT_MODEL_FeatSelection'
        self.modeldir='./UT_model'
        self.collection_name ='../test_data/test_3820_feature_selection/col'
        self.doer = DU_BL_Task.DU_BL_V1(self.modelName, self.modeldir)
        self.doer.addFixedLR()
        #Remove Previous file in anya
        self.doer.rm()

    def test_training(self):
        self.doer.train_save_test([self.collection_name], [], False)


    def test_getSelectedFeatures(self):
        #Redo a Traning if not done
        self.doer.train_save_test([self.collection_name], [], True)
        node_transformer,edge_transformer =self.doer._mdl.getTransformers()


        self.assertTrue(node_transformer is not None)
        print(node_transformer)

        #The Feature definition object is instantiated but not saved after
        #Only the node_transformer and the edge transformer
        ngrams_selected=FeatureDefinition_PageXml_FeatSelect.getNodeTextSelectedFeatures(node_transformer)
        print('########  Tokens Selected')
        for wstring in ngrams_selected:
            print(wstring)

        self.assertTrue('HHH' in  ngrams_selected)
        self.assertTrue('PPP' in  ngrams_selected)








if __name__ == '__main__':
    unittest.main()






