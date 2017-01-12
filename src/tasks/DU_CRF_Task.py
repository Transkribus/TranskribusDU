# -*- coding: utf-8 -*-

"""
    CRF DU task core
    
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
import os, glob
from optparse import OptionParser

from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV

from common.trace import traceln

import crf.Model
from crf.Model_SSVM_AD3 import Model_SSVM_AD3
from xml_formats.PageXml import MultiPageXml
import crf.FeatureDefinition
from crf.FeatureDefinition_PageXml_std import FeatureDefinition_PageXml_StandardOnes

class DU_CRF_Task:
    """
Document Understanding class that relies on CRF (learning with SSVM and inference with AD3, thru the pystruct library

USAGE:
- define your graph class
- choose a model name and a folder where it will be stored
- define the features and their configuration
- define the learner configuration
- instantiate this class

METHODs:
- training: train_save_test
- loading a trained model: load
- testing a trained model: test
- removing a model from the disk: rm
- if you want to use specific features: setFeatureDefinition
- if you want to also train/test against some baseline model: setBaselineList, addBaseline_LogisticRegression 

See DU_StAZH_b.py

    """
    
    cModelClass          = Model_SSVM_AD3
    cFeatureDefinition   = FeatureDefinition_PageXml_StandardOnes
    
    sMetadata_Creator = "XRCE Document Understanding CRF-based - v0.2"
    sMetadata_Comments = ""
    
    dGridSearch_LR_conf = {'C':[0.1, 0.5, 1.0, 2.0] }  #Grid search parameters for LR baseline method training

    def __init__(self, sModelName, sModelDir, cGraphClass, dFeatureConfig={}, dLearnerConfig={}, sComment=None, cFeatureDefinition=None): 
        """
        """
        self.sModelName     = sModelName
        self.sModelDir      = sModelDir
        self.cGraphClass    = cGraphClass
        self.config_extractor_kwargs    = dFeatureConfig
        self.config_learner_kwargs      = dLearnerConfig
        if sComment: self.sMetadata_Comments    = sComment
        
        self._mdl = None
        self._lBaselineModel = []
        self.bVerbose = True
        
        if cFeatureDefinition: self.cFeatureDefinition = cFeatureDefinition
        assert issubclass(self.cModelClass, crf.Model.Model), "Your model class must inherit from crf.Model.Model"
        assert issubclass(self.cFeatureDefinition, crf.FeatureDefinition.FeatureDefinition), "Your feature definition class must inherit from crf.FeatureDefinition.FeatureDefinition"
    
    #---  CONFIGURATION setters --------------------------------------------------------------------
    def setModelClass(self, cModelClass): 
        self.cModelClass = cModelClass
        assert issubclass(self.cModelClass, crf.Model.Model), "Your model class must inherit from crf.Model.Model"
        
    def setFeatureDefinition(self, cFeatureDefinition): 
        self.cFeatureDefinition = cFeatureDefinition
        assert issubclass(self.cFeatureDefinition, crf.FeatureDefinition.FeatureDefinition), "Your feature definition class must inherit from crf.FeatureDefinition.FeatureDefinition"

    #---  COMMAND LINE PARSZER --------------------------------------------------------------------
    def getBasicTrnTstRunOptionParser(cls, sys_argv0=None, version=""):
        usage = "%s <model-name> <model-directory> [--rm] [--trn <col-dir> [--warm]]+ [--tst <col-dir>]+ [--run <col-dir>]+"%sys_argv0
        description = """ 
        Train or test or remove the given model or predict using the given model.
        The data is given as a list of DS directories.
        The model is loaded from or saved to the model directory.
        """
    
        #prepare for the parsing of the command line
        parser = OptionParser(usage=usage, version=version)
        
        parser.add_option("--trn", dest='lTrn',  action="store", type="string"
                          , help="Train or continue previous training session using the given annotated collection.")    
        parser.add_option("--tst", dest='lTst',  action="store", type="string"
                          , help="Test a model using the given annotated collection.")    
        parser.add_option("--run", dest='lRun',  action="store", type="string"
                          , help="Run a model on the given non-annotated collection.")    
        parser.add_option("-w", "--warm", dest='warm',  action="store_true"
                          , help="Attempt to warm-start the training")   
        parser.add_option("--rm", dest='rm',  action="store_true"
                          , help="Remove all model files")   
        return usage, description, parser
    getBasicTrnTstRunOptionParser = classmethod(getBasicTrnTstRunOptionParser)
           
    #----------------------------------------------------------------------------------------------------------    
    def setBaselineList(self, lMdl):
        """
        Add one or several baseline methods.
        set one or a list of sklearn model(s):
        - they MUST be initialized, so that the fit method can be called at train time
        - they MUST accept the sklearn usual predict method
        - they SHOULD support a concise __str__ method
        They will be trained with the node features, from all nodes of all training graphs
        """
        self._lBaselineModel = lMdl
        
    def getBaselineList(self):
        return self._lBaselineModel
    
    def addBaseline_LogisticRegression(self):
        """
        add as Baseline a Logistic Regression model, trained via a grid search
        """
        lr = LogisticRegression(class_weight='balanced')
        mdl = GridSearchCV(lr , self.dGridSearch_LR_conf)        
        self._lBaselineModel.append(mdl)

    #----------------------------------------------------------------------------------------------------------   
    # in case you want no output at all on stderr
    def setVerbose(self, bVerbose)  : self.bVerbose = bVerbose 
    def getVerbose(self)            : return self.bVerbose 
    
    def traceln(self, *kwargs)     : 
        if self.bVerbose: traceln(*kwargs)
    #----------------------------------------------------------------------------------------------------------    
    def load(self, bForce=False):
        """
        Load the model from the disk
        if bForce == True, force the load, even if the model is already loaded in memory
        """
        if bForce or not self._mdl:
            self.traceln("- loading a %s model"%self.cModelClass)
            self._mdl = self.cModelClass(self.sModelName, self.sModelDir)
            self._mdl.load()
            self.traceln(" done")
        else:
            self.traceln("- %s model already loaded"%self.cModelClass)
            
        return
    
    def rm(self):
        """
        Remove from the disk any file for this model!!!
        """
        mdl = self.cModelClass(self.sModelName, self.sModelDir)
        
        for s in [  mdl.getModelFilename()
                  , mdl.getTransformerFilename()
                  , mdl.getConfigurationFilename()
                  , mdl.getBaselineFilename()       ]:
            if os.path.exists(s):
                self.traceln("\t - rm %s"%s) 
                os.unlink(s)
        if os.path.exists(self.sModelDir) and not os.listdir(self.sModelDir):
            self.traceln("\t - rmdir %s"%self.sModelDir) 
            os.rmdir(self.sModelDir)
        return 
    
    def train_save_test(self, lsTrnColDir, lsTstColDir, bWarm=False):
        """
        - Train a model on the tTRN collections, if not empty.
        - Test the trained model using the lTST collections, if not empty.
        - Also train/test any baseline model associated to the main model.
        - Trained models are saved on disk, for testing, redicting or further training (by warm-start)
        - if bWarm==True: warm-start the training from any data stored on disk. Otherwise, a non-empty model folder raises a ModelException
        return a test report object
        """
        self.traceln("-"*50)
        self.traceln("Model file '%s' in folder '%s'"%(self.sModelName, self.sModelDir))
        sConfigFile = os.path.join(self.sModelDir, self.sModelName+".py")
        self.traceln("  Configuration file: %s"%sConfigFile)
        self.traceln("Training with collection(s):", lsTrnColDir)
        self.traceln("Testing with  collection(s):", lsTstColDir)
        self.traceln("-"*50)
        
        #list the train and test files
        #NOTE: we check the presence of a digit before the '.' to eclude the *_du.xml files
        ts_trn, lFilename_trn = self.listMaxTimestampFile(lsTrnColDir, "*[0-9]"+MultiPageXml.sEXT)
        _     , lFilename_tst = self.listMaxTimestampFile(lsTstColDir, "*[0-9]"+MultiPageXml.sEXT)
        
        DU_GraphClass = self.cGraphClass
        
        self.traceln("- creating a %s model"%self.cModelClass)
        mdl = self.cModelClass(self.sModelName, self.sModelDir)
        
        if not bWarm:
            if os.path.exists(mdl.getModelFilename()): raise crf.Model.ModelException("Model exists on disk already, either remove it first or warm-start the training.")
            
        mdl.configureLearner(**self.config_learner_kwargs)
        mdl.setBaselineModelList(self._lBaselineModel)
        mdl.saveConfiguration( (self.config_extractor_kwargs, self.config_learner_kwargs) )
        self.traceln("\t - configuration: ", self.config_learner_kwargs )

        self.traceln("- loading training graphs")
        lGraph_trn = DU_GraphClass.loadGraphs(lFilename_trn, bDetach=True, bLabelled=True, iVerbose=1)
        self.traceln(" %d graphs loaded"%len(lGraph_trn))

        self.traceln("- retrieving or creating feature extractors...")
        try:
            mdl.loadTransformers(ts_trn)
        except crf.Model.ModelException:
            fe = self.cFeatureDefinition(**self.config_extractor_kwargs)         
            fe.fitTranformers(lGraph_trn)
            fe.cleanTransformers()
            mdl.setTranformers(fe.getTransformers())
            mdl.saveTransformers()
        self.traceln(" done")
        
        self.traceln("- training model...")
        mdl.train(lGraph_trn, True, ts_trn)
        mdl.save()
        self.traceln(" done")
        # OK!!
        self._mdl = mdl
        
        if lFilename_tst:
            self.traceln("- loading test graphs")
            lGraph_tst = DU_GraphClass.loadGraphs(lFilename_tst, bDetach=True, bLabelled=True, iVerbose=1)
            self.traceln(" %d graphs loaded"%len(lGraph_tst))
    
            oReport = mdl.test(lGraph_tst)
        else:
            oReport = None, None
            
        return oReport

    def test(self, lsTstColDir):
        """
        test the model
        return a TestReport object
        """
        self.traceln("-"*50)
        self.traceln("Trained model '%s' in folder '%s'"%(self.sModelName, self.sModelDir))
        self.traceln("Testing  collection(s):", lsTstColDir)
        self.traceln("-"*50)
        
        if not self._mdl: raise Exception("The model must be loaded beforehand!")
        
        #list the train and test files
        _     , lFilename_tst = self.listMaxTimestampFile(lsTstColDir, "*[0-9]"+MultiPageXml.sEXT)
        
        DU_GraphClass = self.cGraphClass
        
        lPageConstraint = DU_GraphClass.getPageConstraint()
        if lPageConstraint: 
            for dat in lPageConstraint: self.traceln("\t\t%s"%str(dat))
            
        self.traceln("- loading test graphs")
        lGraph_tst = DU_GraphClass.loadGraphs(lFilename_tst, bDetach=True, bLabelled=True, iVerbose=1)
        self.traceln(" %d graphs loaded"%len(lGraph_tst))

        oReport = self._mdl.test(lGraph_tst)
        return oReport

    def predict(self, lsColDir):
        """
        Return the list of produced files
        """
        self.traceln("-"*50)
        self.traceln("Trained model '%s' in folder '%s'"%(self.sModelName, self.sModelDir))
        self.traceln("Predicting for collection(s):", lsColDir)
        self.traceln("-"*50)

        if not self._mdl: raise Exception("The model must be loaded beforehand!")
        
        #list the train and test files
        _     , lFilename = self.listMaxTimestampFile(lsColDir, "*[0-9]"+MultiPageXml.sEXT)
        
        DU_GraphClass = self.cGraphClass

        lPageConstraint = DU_GraphClass.getPageConstraint()
        if lPageConstraint: 
            for dat in lPageConstraint: self.traceln("\t\t%s"%str(dat))
        
        self.traceln("- loading collection as graphs, and processing each in turn. (%d files)"%len(lFilename))
        du_postfix = "_du"+MultiPageXml.sEXT
        lsOutputFilename = []
        for sFilename in lFilename:
            if sFilename.endswith(du_postfix): continue #:)
            [g] = DU_GraphClass.loadGraphs([sFilename], bDetach=False, bLabelled=False, iVerbose=1)
            
            if lPageConstraint:
                self.traceln("\t- prediction with logical constraints: %s"%sFilename)
            else:
                self.traceln("\t- prediction : %s"%sFilename)
            Y = self._mdl.predict(g)
                
            doc = g.setDomLabels(Y)
            MultiPageXml.setMetadata(doc, None, self.sMetadata_Creator, self.sMetadata_Comments)
            sDUFilename = sFilename[:-len(MultiPageXml.sEXT)]+du_postfix
            doc.saveFormatFileEnc(sDUFilename, "utf-8", True)  #True to indent the XML
            doc.freeDoc()
            del Y, g
            self.traceln("\t done")
            lsOutputFilename.append(sDUFilename)
        self.traceln(" done")

        return lsOutputFilename

    #----------------------------------------------------------------------------------------------------------    
    def listMaxTimestampFile(cls, lsDir, sPattern):
        """
        List the file following the given pattern in the given folders
        return the timestamp of the most recent one
        Return a timestamp and a list of filename
        """
        lFn, ts = [], None 
        for sDir in lsDir:
            lsFilename = sorted(glob.iglob(os.path.join(sDir, sPattern)))  
            lFn.extend(lsFilename)
            if lsFilename: ts = max(ts, max([os.path.getmtime(sFilename) for sFilename in lsFilename]))
        return ts, lFn
    listMaxTimestampFile = classmethod(listMaxTimestampFile)
    
