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
    
    def __init__(self, name, Y_pred, Y):
        pass