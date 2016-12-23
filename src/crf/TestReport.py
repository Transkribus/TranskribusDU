# -*- coding: utf-8 -*-

"""
    TestReport of a multi-class classifier

    Copyright Xerox(C) 2016 JL. Meunier

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""
import os
import cPickle, gzip, json
import types

import numpy as np

from pystruct.utils import SaveLogger

from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report

from pystruct.learners import OneSlackSSVM
from pystruct.models import EdgeFeatureGraphCRF

from pystruct.models import ChainCRF
from pystruct.learners import FrankWolfeSSVM

from common.trace import  traceln
from common.chrono import chronoOn, chronoOff

class TestReport:
    """
    A class that encapsulates the result of a classification test
    """
    
    def __init__(self, name, l_Y_pred, l_Y, lsClassName):
        """
        compute the confusion matrix and classification report.
        Print them on stderr and return the accuracy global score and the report
        """
        
        if type(l_Y_pred) == types.ListType:
            Y_pred = np.hstack(l_Y_pred)
            Y      = np.hstack(l_Y)
        else:
            Y_Pred = l_Y_pred
            Y      = l_Y
        
        #we need to include all clasname that appear in the dataset or in the predicted labels (well... I guess so!)
        if lsClassName:
            setSeenCls = set()
            for _Y in [Y, Y_pred]:
                setSeenCls = setSeenCls.union( np.unique(_Y).tolist() )
            lsSeenClassName = [ cls for (i, cls) in enumerate(lsClassName) if i in setSeenCls]
            
        traceln("Line=True class, column=Prediction")
        a = confusion_matrix(Y, Y_pred)
#not nice because of formatting of numbers possibly different per line
#         if lsClassName:
#             #Let's show the class names
#             s1 = ""
#             for sLabel, aLine in zip(lsSeenClassName, a):
#                 s1 += "%20s %s\n"%(sLabel, aLine)
#         else:
#             s1 = str(a)
        s1 = str(a)
        if lsClassName:
            lsLine = s1.split("\n")    
            assert len(lsLine)==len(lsSeenClassName)
            s1 = "\n".join( ["%20s  %s"%(sLabel, sLine) for (sLabel, sLine) in zip(lsSeenClassName, lsLine)])    
        traceln(s1)

        if lsClassName:
            s2 = classification_report(Y, Y_pred, target_names=lsSeenClassName)
        else:
            s2 = classification_report(Y, Y_pred)
        traceln(s2)
        
        self.fScore = accuracy_score(Y, Y_pred)
        s3 = "(unweighted) Accuracy score = %.2f"% self.fScore
        traceln(s3)
    
    def __str__(self):
        return str(self.fScore)
