
#Idea 0
#Implement Mutual Information
#Per Class Features Selection
#Most Frequent ...


#Do a mutual information criterion with mean
#Try the Round_Robin with chi2 MI  and diffrent criterion ... store the results
# Redo test to have it all



#Idea 1 Character Embedding
# MSDA

from __future__ import absolute_import
from __future__ import  print_function
from __future__ import unicode_literals

from sklearn.base import BaseEstimator

from sklearn.utils import (as_float_array, check_array, check_X_y)
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted
from scipy.sparse import issparse
from sklearn.preprocessing import LabelBinarizer

import scipy.sparse as sp
import numpy as np



from sklearn.feature_selection.base import SelectorMixin

def _clean_nans(scores):
    """
    Fixes Issue #1240: NaNs can't be properly compared, so change them to the
    smallest value of scores's dtype. -inf seems to be unreliable.
    """
    # XXX where should this function be called? fit? scoring functions
    # themselves?
    scores = as_float_array(scores, copy=True)
    scores[np.isnan(scores)] = np.finfo(scores.dtype).min
    return scores


def pointwise_mutual_information_score(X,y):

    #TODO Should binarize things
    #Filter negative mutual information

    X = check_array(X, accept_sparse='csr')
    if np.any((X.data if issparse(X) else X) < 0):
        raise ValueError("Input X must be non-negative.")

    if issparse(X):
        X_bin = sp.csr_matrix(X)
        X_bin.data=np.ones(X_bin.data.shape)

    else:
        raise ValueError('Matrix should be sparse')


    Y = LabelBinarizer().fit_transform(y)
    if Y.shape[1] == 1:
        Y = np.append(1 - Y, Y, axis=1)

    observed = safe_sparse_dot(Y.T, X_bin,dense_output=True)          # n_classes * n_features

    feature_count = X_bin.sum(axis=0).reshape(1, -1)

    n_feat=X_bin.shape[1]
    n_classes = Y.shape[1] #Should be 2 for binary classes

    class_count = Y.sum(axis=0).reshape(1, -1)

    F = np.tile(feature_count,(n_classes,1))
    C = np.tile(class_count,(n_feat,1))

    #embed()

    print(feature_count.shape)
    print(class_count.shape)

    eps = 1e-8

    #Do real Mutual Information
    PMI = np.log(eps+observed) -np.log(C.T) -np.log(F)
    PMI = np.maximum(PMI,0)


    return np.asarray(PMI.sum(axis=0)).squeeze()
    #return np.asarray(np.max(PMI,axis=0)).squeeze()


def mutual_information(Xbin,q,npmi_norm=False):
    """
    Xfiltered is Xword_doc binarized
    """

    Xcsr = Xbin.transpose().tocsr()
    DF = np.array(Xcsr.sum(axis=1)).squeeze()

    (nw, nd) = Xcsr.shape

    nd = float(nd)
    eps = 1e-8
    #Convert that as Boolean array
    #Matrix Multiplication by Chunks of 100, or parrallize
    P_DF = DF / (float(nd))
    P_NDF = 1.0 - P_DF

    log_P_DF = np.log(eps + P_DF)
    log_P_NDF = np.log(eps+ P_NDF)


    Nij = safe_sparse_dot(Xcsr, q, dense_output=False)  #Could speed up by using false
    nnzindx=np.nonzero(Nij)[0]

    #u == class q
    P_y = q.sum()/nd
    P_Ny= 1-P_y


    P_w_y = Nij[nnzindx] / nd  #Fraction of docs containing u and v
    P_Nw_y = P_y - P_w_y  #(DF[wid]- Nij)/nd # Fraction of doc containing of class y but do not contain w
    P_w_Ny = P_DF[nnzindx] - P_w_y  #(DF - Nij)/nd

    #
    #Check Values here
    '''
    debug = (eps+P_Nw_y)<0.0
    found_index = np.nonzero(debug)[0]
    print('Debugging')
    #print(np.log(eps+P_Nw_y))
    print(list(P_Nw_y[found_index]))
    print(found_index)
    print P_DF[found_index]
    '''

    #P_Nv_NDu = (nd - DF - (DF[wid]-Nij)) /nd
    P_Nw_Ny = 1.0 - P_w_y -P_Nw_y - P_w_Ny


    DS1 =np.log(P_y+eps)*np.ones(nw)[nnzindx]
    DS2 =np.log(P_Ny+eps) *np.ones(nw)[nnzindx]

    DS3 =log_P_DF[nnzindx]
    DS4 =log_P_NDF[nnzindx]

    #gi =np.zeros(nw)
    gi =-np.inf*np.ones(nw)

    #TODO Check that all the stats are indeed positive

    gi[nnzindx] = P_w_y * (np.log(eps+P_w_y) - DS1 - DS3) +\
                  P_Nw_y * ( np.log(eps + P_Nw_y) - DS1 - DS4) + \
                  P_w_Ny * (np.log(eps + P_w_Ny) - DS2 - DS3 ) +  \
                  P_Nw_Ny * ( np.log(eps + P_Nw_Ny) - DS2- DS4 )

    if npmi_norm:
        nmi_renorm =  - (P_w_y*np.log(P_w_y) + P_Nw_y*np.log(eps + P_Nw_y) + P_w_Ny *np.log(eps + P_w_Ny) + P_Nw_Ny*np.log(eps + P_Nw_Ny) )
        gi[nnzindx]/= nmi_renorm

    return gi

#Possibly add cost here on the feature selection things
#This is the RandRobin ... using cost on instances ... to select things differently on class
#

class SelectRobinBest(BaseEstimator,SelectorMixin):

    def __init__(self,score_func,k=100):
        super(SelectRobinBest, self).__init__()
        self.score_func = score_func # Score function by class
        self.k=k
        self.themask=[]


    def _check_params(self, X, y):
        #TODO Check Sparse Matrix ?
        pass


    def fit(self, X, y):
        """Run score function on (X, y) and get the appropriate features.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """

        print('Fitting RoundRobin')
        X, y = check_X_y(X, y, ['csr', 'csc'])

        if not callable(self.score_func):
            raise TypeError("The score function should be a callable, %s (%s) "
                            "was passed."
                            % (self.score_func, type(self.score_func)))

        self._check_params(X, y)
        #Compute the score per class
        Y = LabelBinarizer().fit_transform(y)
        #Y is a matrix doc times class
        nclasses= Y.shape[1]


        #Should Binarize X_bin here as well
        #It has not bin binarized here ...
        if issparse(X):
            Xbin = sp.csr_matrix(X)
            Xbin.data=np.ones(Xbin.data.shape)
        else:
            raise ValueError('Should Give Sparse Matrix Here')


        #pdb.set_trace()


        #TODO Improve this
        self.Y=Y
        self.scores_=[]
        for i in range(nclasses):
            print('Computing Scores .....')
            self.scores_.append(self.score_func(Xbin, np.squeeze(Y[:,i].T )))


        print('Selecting Features')
        self.select_feat()
        self.toto='123'

        print('Nb features selected',self.themask.sum())

        return self

    def select_feat(self):

        check_is_fitted(self, 'scores_')
        if self.k == 'all':
            self.themask=np.ones(self.scores_.shape, dtype=bool)
            return self.themask
        else:
            #Select k by class
            nb_classes=self.Y.shape[1]
            n_feat =self.scores_[0].shape
            mask = np.zeros(n_feat, dtype=bool)
            ###################################################
            S=[]
            for i in range(self.Y.shape[1]):
                scores = _clean_nans(self.scores_[i])
                # Request a stable sort. Mergesort takes more memory (~40MB per
                # megafeature on x86-64).
                #mask[np.argsort(scores, kind="mergesort")[-self.k:]] = 1
                sindx =np.argsort(scores, kind="mergesort")
                S.append(list(sindx))


            nb_sel=0
            current_class=0
            #Should save this and do it once ....
            #Why does it call it multiple times
            print("Number of Feat",n_feat)
            lk =min(self.k,n_feat[0])
            while nb_sel< lk :
                #Find a feature for the current class
                not_found=True
                while not_found:
                    #print(nb_sel,current_class,lk)
                    feat_index = S[current_class].pop()
                    if mask[feat_index]==0:
                        mask[feat_index]=1
                        nb_sel+=1
                        break
                current_class = (current_class+1)% nb_classes



            self.themask=mask
            self.scores_.append(mask)
            return self.themask


    def _get_support_mask(self):
        """
        Get the boolean mask indicating which features are selected

        Returns
        -------
        support : boolean array of shape [# input features]
            An element is True iff its corresponding feature is selected for
            retention.
        """

        check_is_fitted(self, 'scores_')
        if self.k == 'all':
            return np.ones(self.scores_.shape, dtype=bool)
        else:
            return self.themask
            #return self.scores_[-1]
            #return self.select_feat()



