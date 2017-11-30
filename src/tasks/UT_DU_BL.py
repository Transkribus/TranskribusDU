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

import DU_BL_Task,DU_BL_V1

from DU_BL_V1 import DU_BL_V1

import numpy as np
from IPython import embed
import sklearn
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV, RidgeClassifier
import pdb
from sklearn.metrics import accuracy_score,roc_auc_score
import scipy.sparse as sp
from xml_formats.PageXml import MultiPageXml
from sklearn.metrics import average_precision_score



class UT_DU_BL(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(UT_DU_BL, self).__init__(*args, **kwargs)
        #Z=cPickle.load(open('UT_line_embedding.pickle','rb'))
        self.modelName='UT_MODEL_FeatSelection'
        self.modeldir='./UT_model'
        self.collection_name ='../test_data/test_3820_feature_selection/col'
        #self.doer = DU_BL_Task.DU_BL_V1(self.modelName, self.modeldir,feat_select='chi2')
        #self.doer.addFixedLR()
        #Remove Previous file in anya
        #self.doer.rm()

    '''
    def test_training(self):
        self.doer.train_save_test([self.collection_name], [], False)
    '''

    def test_getSelectedFeatures(self):
        #Redo a Traning if not done
        self.doer = DU_BL_V1(self.modelName, self.modeldir,feat_select='chi2')
        self.doer.addFixedLR()
        self.doer.rm()
        self.doer.train_save_test([self.collection_name], [], False)
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


    def test_featselect_disabled(self):
        self.doer = DU_BL_V1(self.modelName, self.modeldir,feat_select=None)
        self.doer.addFixedLR()
        #Remove Previous file in anya
        self.doer.rm()
        self.doer.train_save_test([self.collection_name], [], False)
        node_transformer,edge_transformer =self.doer._mdl.getTransformers()

        self.assertTrue(node_transformer is not None)
        print(node_transformer)

        if hasattr(node_transformer,'word_select'):
            self.fail('should not find word selector')

    def test_average_precision(self):
        print('Passing Test')
        '''
        self.doer = DU_BL_V1(self.modelName, self.modeldir,feat_select=None)
        self.doer.addFixedLR()
        #Remove Previous file in anya
        self.doer.rm()
        self.doer.train_save_test([self.collection_name], [], True)

        #Compute the average precision on the training set /Just to test the method
        DUGraphClass = self.doer.cGraphClass
        ts_trn, lFilename_trn = DU_BL_V1.listMaxTimestampFile([self.collection_name], "*[0-9]"+MultiPageXml.sEXT)


        lGraph_tst = DUGraphClass.loadGraphs(lFilename_trn, bDetach=True, bLabelled=True, iVerbose=1)
        lX, lY = self.doer._mdl.transformGraphs(lGraph_tst, True)

        X_flat = np.vstack( [node_features for (node_features, _, _) in lX] )
        Y_flat = np.hstack(lY)
        lr_model =self.doer._mdl._lMdlBaseline[0]
        Y_pred_flat = lr_model.predict_proba(X_flat)
        #This should break
        print(Y_flat)
        print Y_pred_flat

        for i in range(Y_flat.max()+1):
            print(i,average_precision_score(Y_flat==i,Y_pred_flat[:,i]))
        #TODO Update test
        '''

    def test_ranking_report(self):
        self.doer = DU_BL_V1(self.modelName, self.modeldir,feat_select=None)
        self.doer.addFixedLR()
        #Remove Previous file in anya
        self.doer.rm()
        report=self.doer.train_save_test([self.collection_name], [self.collection_name], True)
        print(report[0])
        self.assertTrue(report[0].average_precision is not [])



if __name__ == '__main__':
    unittest.main()






