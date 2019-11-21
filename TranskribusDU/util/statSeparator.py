# -*- coding: utf-8 -*-
"""
Currently provides only the computation of a linear separator

H. Déjean, JL Meunier, Copyright Naver Labs Europe 2019
"""

from sklearn import svm


def getLinearSeparator(X, Y):
    """
    Linear separator
    
    return a,b so that the linear separator has the form Y = a X + b
    """

    #C = 1.0  # SVM regularization parameter
    # clf = svm.SVC(kernel = 'linear',  gamma=0.7, C=C )
    clf = svm.SVC(kernel = 'linear')
    clf.fit(X, Y)
    w = clf.coef_[0]
    a = -w[0] / w[1]
    b = - (clf.intercept_[0]) / w[1]
    return a, b


def test_getLinearSeparator():
    import numpy as np
    
    lP = [(i, 10) for i in range(10)]
    lV = [(i, -2) for i in range(10)]
    X = np.array(lP+lV)
    Y = np.array([1]*10 + [0]*10)
    
    a,b = getLinearSeparator(X, Y)
    assert abs(a)   < 0.001
    assert abs(b-4) < 0.001
    #print(a,b)
    
    lP = [(i, 10+i) for i in range(10)]
    lV = [(i, -2+i) for i in range(10)]
    X = np.array(lP+lV)
    Y = np.array([1]*10 + [0]*10)
    
    a,b = getLinearSeparator(X, Y)
    assert abs(a-1)   < 0.001
    assert abs(b-4) < 0.001
    # print(a,b)

