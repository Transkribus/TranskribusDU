# -*- coding: utf-8 -*-

"""
    Factorial CRF DU task core. Supports classical CRF and Typed CRF
    
    Copyright Xerox(C) 2016, 2017 JL. Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""




import sys, os

import numpy as np

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from common.trace import traceln

import graph.GraphModel
from crf.Model_SSVM_AD3 import Model_SSVM_AD3
from crf.Model_SSVM_AD3_Multitype import Model_SSVM_AD3_Multitype

import crf.FeatureDefinition


from .DU_CRF_Task import DU_CRF_Task

class DU_FactorialCRF_Task(DU_CRF_Task):


    def __init__(self, sModelName, sModelDir,  dLearnerConfig={}, sComment=None
                 , cFeatureDefinition=None, dFeatureConfig={}
                 ): 
        """
        Same as DU_CRF_Task except for the cFeatureConfig
        """
        self.configureGraphClass()
        self.sModelName     = sModelName
        self.sModelDir      = sModelDir
#         self.cGraphClass    = cGraphClass
        #Because of the way of dealing with the command line, we may get singleton instead of scalar. We fix this here
        self.config_learner_kwargs      = {k:v[0] if type(v) is list and len(v)==1 else v for k,v in dLearnerConfig.items()}
        if sComment: self.sMetadata_Comments    = sComment
        
        self._mdl = None
        self._lBaselineModel = []
        self.bVerbose = True
        
        self.iNbNodeType = None #is set below
        
        #--- Number of class per type
        #We have either one number of class (single type) or a list of number of class per type
        #in single-type CRF, if we know the number of class, we check that the training set covers all
        self.nbClass  = None    #either the number or the sum of the numbers
        self.lNbClass = None    #a list of length #type of number of class

        #--- feature definition and configuration per type
        #Feature definition and their config
        if cFeatureDefinition: self.cFeatureDefinition  = cFeatureDefinition
        assert issubclass(self.cFeatureDefinition, crf.FeatureDefinition.FeatureDefinition), "Your feature definition class must inherit from crf.FeatureDefinition.FeatureDefinition"
        
        #for single- or multi-type CRF, the same applies!
        self.lNbClass = [len(nt.getLabelNameList()) for nt in self.cGraphClass.getNodeTypeList()]
        self.nbClass = sum(self.lNbClass)
        self.iNbNodeType = len(self.cGraphClass.getNodeTypeList())

        self.config_extractor_kwargs = dFeatureConfig

        self.cModelClass = Model_SSVM_AD3 if self.iNbNodeType == 1 else Model_SSVM_AD3_Multitype
        assert issubclass(self.cModelClass, graph.GraphModel.GraphModel), "Your model class must inherit from graph.GraphModel.GraphModel"
        
        
