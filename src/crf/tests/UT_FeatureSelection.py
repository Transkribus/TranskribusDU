__author__ = 'sclincha'

import sys,os
sys.path.append('../..')

import numpy as np
import unittest



try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version


from crf.FeatureSelection import pointwise_mutual_information_score,mutual_information

from crf.FeatureSelection import *

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from IPython import embed

import scipy.sparse as sp






def compute_expected_score(P_w_q,P_w_nq,P_nw_q,P_nw_nq,P_w,P_q,P_nw,P_nq):
    eps=1e-8
    return P_w_q *( np.log(P_w_q+eps) - np.log(P_w+eps) -np.log(P_q+eps) ) \
                           + P_w_nq *( np.log(P_w_nq+eps) - np.log(P_w+eps) -np.log(P_nq+eps) )\
                           + P_nw_q *( np.log(P_nw_q+eps) - np.log(P_nw+eps) -np.log(P_q+eps)  )\
                           + P_nw_nq *(np.log(P_nw_nq+eps) - np.log(P_nw+eps) -np.log(P_nq+eps) )



class UT_FeatureSelection(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(UT_FeatureSelection, self).__init__(*args, **kwargs)
        self.X_list=['aktum es jacta es','this is a page number1',
                     'page_number2','page number 3','aktum dejean meunier',
                     'akktum meunier',
                     'bbla contracting billon','contract ixizi tu million','project con tract','bli bli dskdf']
        self.Y_list =['title','pn','pn','pn','title','title','others','others','others','others']
        self.cvect  = CountVectorizer(lowercase=True, max_features=1000, analyzer = 'char', ngram_range=(2,4))
        self.cvect_w  = CountVectorizer(lowercase=True, max_features=1000, analyzer = 'word')


    def test_chi2(self):
        feat_selector=SelectKBest(chi2, k=20)
        text_pipeline = Pipeline([('tf', self.cvect),('word_selector',feat_selector)])
        text_pipeline.fit(self.X_list,self.Y_list)

        #Test Something Here
        #embed()
        cvect=text_pipeline.named_steps['tf']
        #Index to Word String array
        I2S_array =np.array(cvect.get_feature_names())
        fs=text_pipeline.named_steps['word_selector']
        selected_indices=fs.get_support(indices=True)
        print(I2S_array[selected_indices])


    def test_pointwise_mutualinfo(self):
        feat_selector=SelectKBest(pointwise_mutual_information_score, k=20)
        text_pipeline = Pipeline([('tf', self.cvect),('word_selector',feat_selector)])
        text_pipeline.fit(self.X_list,self.Y_list)

        #Test Something Here
        #embed()
        cvect=text_pipeline.named_steps['tf']
        #Index to Word String array
        I2S_array =np.array(cvect.get_feature_names())
        fs=text_pipeline.named_steps['word_selector']
        selected_indices=fs.get_support(indices=True)
        print(I2S_array[selected_indices])

        #Check Mutual Info Value

    def test_mutualinfo(self):
        #Define matrix here
        X_list =[[1.5,0,0],
                 [0,0,1],
                 [1,0,1],
                 [1,1,1],
                 [0,1,0]]

        Y_list = [1,0,0,0,2]
        X_csr =sp.csr_matrix(np.array(X_list))


        nd =5.0
        q=np.zeros(int(nd))
        q[0]=1

        Xbin = sp.csr_matrix(X_csr)
        Xbin.data=np.ones(Xbin.data.shape)


        #Matrix should be word times doc
        gi=mutual_information(Xbin,q,npmi_norm=False)


        #Test for first word and category
        #  w  = 1 2
        #  Nw = 0 2

        P_w_q   = 1.0/nd
        P_w_nq  = 2.0/nd
        P_nw_q = 0.0/nd
        P_nw_nq = 2.0/nd

        P_w = 3.0/nd
        P_q = 1.0/nd
        P_nw =2.0/nd
        P_nq =4.0/nd


        eps=1e-8
        expected_score_0 = P_w_q *( np.log(P_w_q+eps) - np.log(P_w+eps) -np.log(P_q+eps) ) \
                           + P_w_nq *( np.log(P_w_nq+eps) - np.log(P_w+eps) -np.log(P_nq+eps) )\
                           + P_nw_q *( np.log(P_nw_q+eps) - np.log(P_nw+eps) -np.log(P_q+eps)  )\
                           + P_nw_nq *(np.log(P_nw_nq+eps) - np.log(P_nw+eps) -np.log(P_nq+eps) )

        print(expected_score_0,gi[0])
        self.assertAlmostEquals(expected_score_0,gi[0])

        #DO other test


        q2=np.zeros(int(nd))
        q2[1:4]=1

        #Test for last word and category 0
        gi2=mutual_information(Xbin,q2,npmi_norm=False)
        #X_list =[[1.5,0,0],
        #         [0,0,1],
        #         [1,0,1],
        #         [1,1,1],
        #         [0,1,0]]

        P_w_q   = 3.0/nd
        P_w_nq  = 0.0/nd
        P_nw_q = 0.0/nd
        P_nw_nq = 2.0/nd

        P_w = 3.0/nd
        P_q = 3.0/nd
        P_nw =2.0/nd
        P_nq =2.0/nd

        expected_score_1=compute_expected_score(P_w_q,P_w_nq,P_nw_q,P_nw_nq,P_w,P_q,P_nw,P_nq)
        print(expected_score_1,gi2[2])
        self.assertAlmostEquals(expected_score_1,gi2[2])


    def test_select_rr_mi(self):
        #Select the best two per class
        feat_selector=SelectRobinBest(mutual_information, k=6)
        text_pipeline = Pipeline([('tf', self.cvect),('word_selector',feat_selector)])
        text_pipeline.fit(self.X_list,self.Y_list)

        #Test Something Here
        #embed()
        cvect=text_pipeline.named_steps['tf']
        #Index to Word String array
        I2S_array =np.array(cvect.get_feature_names())
        fs=text_pipeline.named_steps['word_selector']
        selected_indices=fs.get_support(indices=True)
        print(I2S_array[selected_indices])

    def test_select_rr_chi(self):
        #Select the best two per class
        #Chi2 return score and p_value ; we just keep the score
        chi_score = lambda x,y : chi2(x,y)[0]
        feat_selector=SelectRobinBest(chi_score, k=4)
        text_pipeline = Pipeline([('tf', self.cvect),('word_selector',feat_selector)])
        text_pipeline.fit(self.X_list,self.Y_list)

        #Test Something Here
        #embed()
        cvect=text_pipeline.named_steps['tf']
        #Index to Word String array
        I2S_array =np.array(cvect.get_feature_names())
        fs=text_pipeline.named_steps['word_selector']
        selected_indices=fs.get_support(indices=True)
        self.assertTrue(len(selected_indices)==4)
        print(I2S_array[selected_indices])

    def test_rr_large_nb_feat(self):
        #Select the best two per class
        #Chi2 return score and p_value ; we just keep the score
        chi_score = lambda x,y : chi2(x,y)[0]
        feat_selector=SelectRobinBest(chi_score, k=10000)
        text_pipeline = Pipeline([('tf', self.cvect),('word_selector',feat_selector)])
        text_pipeline.fit(self.X_list,self.Y_list)

        #Test Something Here
        #embed()
        cvect=text_pipeline.named_steps['tf']
        #Index to Word String array
        I2S_array =np.array(cvect.get_feature_names())
        fs=text_pipeline.named_steps['word_selector']
        selected_indices=fs.get_support(indices=True)
        self.assertTrue(len(selected_indices)==len(I2S_array))
        print(I2S_array[selected_indices])

    #def TEST save transformers ....





if __name__ == '__main__':
    unittest.main()
