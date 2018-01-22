# -*- coding: utf-8 -*-

"""
    CRF DU task core. Supports classical CRF and Typed CRF
    
    Copyright Xerox(C) 2016, 2017 JL. Meunier

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
import sys, os, glob, datetime
import types
from optparse import OptionParser

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV  #0.18.1 REQUIRES NUMPY 1.12.1 or more recent
    
try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from common.trace import traceln
from common.chrono import chronoOn, chronoOff

import crf.Model
from crf.Model_SSVM_AD3 import Model_SSVM_AD3
from crf.Model_SSVM_AD3_Multitype import Model_SSVM_AD3_Multitype

from xml_formats.PageXml import MultiPageXml
import crf.FeatureDefinition
from crf.FeatureDefinition_PageXml_std import FeatureDefinition_PageXml_StandardOnes

from crf.TestReport import TestReportConfusion

class DU_CRF_Task:
    """
Document Understanding class that relies on CRF (learning with SSVM and inference with AD3, thru the pystruct library

USAGE:
- define your graph class
- choose a model name and a folder where it will be stored
- define the features and their configuration, or a list of (feature definition and its configuration)
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
    
    cModelClass      = None     #depends on the number of node types!
    cGraphClass      = None     #class of graph in use 
    
    cFeatureDefinition   = FeatureDefinition_PageXml_StandardOnes   #I keep this for backward compa
    
    sMetadata_Creator = "XRCE Document Understanding Typed CRF-based - v0.3"
    sMetadata_Comments = ""
    
    #dGridSearch_LR_conf = {'C':[0.1, 0.5, 1.0, 2.0] }  #Grid search parameters for LR baseline method training
    dGridSearch_LR_conf   = {'C':[0.01, 0.1, 1.0, 10.0] }  #Grid search parameters for LR baseline method training
    dGridSearch_LR_n_jobs = 4                              #Grid search: number of jobs
    
    sXmlFilenamePattern = "*[0-9]"+MultiPageXml.sEXT    #how to find the Xml files

    @classmethod
    def configureGraphClass(cls, configuredClass=None):
        """
        class method to set the graph class ONCE (subsequent calls are ignored)
        """
        if cls.cGraphClass is None: #OK, let's set the class attribute!
            
            #if nothing in parameter, or we call the class method
            if configuredClass is None:
                configuredClass = cls.getConfiguredGraphClass()
                assert configuredClass is not None, "getConfiguredGraphClass returned None"
                
            cls.cGraphClass = configuredClass

        assert cls.cGraphClass is not None
        return cls.cGraphClass

    @classmethod
    def getConfiguredGraphClass(cls):
        """
        In this class method, we must return a configured graph class
        """
        raise Exception("class method getConfiguredGraphClass must be specialized")
    
    
    def __init__(self, sModelName, sModelDir, dLearnerConfig={}, sComment=None
                 , cFeatureDefinition=None, dFeatureConfig={}
                 ): 
        """
        
        """
        self.configureGraphClass()
        self.sModelName    = sModelName
        self.sModelDir     = sModelDir
        
        #Because of the way of dealing with the command line, we may get singleton instead of scalar. We fix this here
        self.config_learner_kwargs      = {k:v[0] if type(v)==types.ListType and len(v)==1 else v for k,v in dLearnerConfig.items()}
        if sComment: self.sMetadata_Comments    = sComment
        
        self._mdl = None
        self._lBaselineModel = []
        self.bVerbose = True
        
        self.iNbCRFType = None #is set below
        
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
        self.iNbCRFType = len(self.cGraphClass.getNodeTypeList())

        if self.iNbCRFType > 1:
            #check the configuration of a MULTITYPE graph
            setKeyGiven = set(dFeatureConfig.keys())
            lNT = self.cGraphClass.getNodeTypeList()
            setKeyExpected = {nt.name for nt in lNT}.union( {"%s_%s"%(nt1.name,nt2.name) for nt1 in lNT for nt2 in lNT} )
            
            setMissing = setKeyExpected.difference(setKeyGiven)
            setExtra   = setKeyGiven.difference(setKeyExpected)
            if setMissing: traceln("ERROR: missing feature extractor config for : ", ", ".join(setMissing))
            if setExtra:   traceln("ERROR: feature extractor config for unknown : ", ", ".join(setExtra))
            if setMissing or setExtra: raise ValueError("Bad feature extractor configuration for a multi-type CRF graph")
           
        self.config_extractor_kwargs = dFeatureConfig

        self.cModelClass = Model_SSVM_AD3 if self.iNbCRFType == 1 else Model_SSVM_AD3_Multitype
        assert issubclass(self.cModelClass, crf.Model.Model), "Your model class must inherit from crf.Model.Model"
        
    #---  CONFIGURATION setters --------------------------------------------------------------------
    def isTypedCRF(self): 
        """
        if this a classical CRF or a Typed CRF?
        """
        return self.iNbCRFType > 1
    
    def getGraphClass(self):    
        return self.cGraphClass
    
    def setModelClass(self, cModelClass): 
        self.cModelClass = cModelClass
        assert issubclass(self.cModelClass, crf.Model.Model), "Your model class must inherit from crf.Model.Model"
    
    def getModelClass(self):    
        return self.cModelClass
    
    def getModel(self):         
        return self._mdl
    
    def setLearnerConfiguration(self, dParams):
        self.config_learner_kwargs = dParams
        
    """
    When some class is not represented on some graph, you must specify the number of class (per type if multiple types)
    Otherwise pystruct will complain about the number of states differeing from the number of weights
    """
    def setNbClass(self, useless_stuff):    #DEPRECATED - DO NOT CALL!! Number of class computed automatically
        print   " *** setNbClass is deprecated - update your code (but it should work fine!)"
        traceln(" *** setNbClass is deprecated - update your code (but it should work fine!)")
        
    def getNbClass(self, lNbClass): #OK
        """
        return the total number of classes
        """
        return self.nbClass
    
    #---  COMMAND LINE PARSZER --------------------------------------------------------------------
    def getBasicTrnTstRunOptionParser(cls, sys_argv0=None, version=""):
        usage = """"%s <model-folder> <model-name> [--rm] [--trn <col-dir> [--warm]]+ [--tst <col-dir>]+ [--run <col-dir>]+
or for a cross-validation [--fold-init <N>] [--fold-run <n> [-w]] [--fold-finish] [--fold <col-dir>]+
[--pkl]
CRF options: [--crf-max_iter <int>]  [--crf-C <float>] [--crf-tol <float>] [--crf-njobs <int>] [crf-inference_cache <int>] [best-params=<model-name>]

        For the named MODEL using the given FOLDER for storage:
        --rm  : remove all model data from the folder
        --trn : train a model using the given data (multiple --trn are possible)
                  --warm/-w: warm-start the training if applicable
        --tst : test the model using the given test collection (multiple --tst are possible)
        --run : predict using the model for the given collection (multiple --run are possible)
        
        --fold        : enlist one collection as data source for cross-validation
        --fold-init   : generate the content of the N folds 
        --fold-run    : run the given fold, if --warm/-w, then warm-start if applicable 
        --fold-finish : collect and aggregate the results of all folds that were run.

        --pkl         : store the data as a pickle file containing PyStruct data structure (lX, lY) and exit
        
        --crf-njobs    : number of parallel training jobs
        
        --crf-XXX        : set the XXX trainer parameter. XXX can be max_iter, C, tol, inference-cache
                            If several values are given, a grid search is done by cross-validation. 
                            The best set of parameters is then stored and can be used thanks to the --best-params option.
        --best-params    : uses the parameters obtained by the previously done grid-search. 
                            If it was done on a model fold, the name takes the form: <model-name>_fold_<fold-number>, e.g. foo_fold_2
        
        """%sys_argv0

        #prepare for the parsing of the command line
        parser = OptionParser(usage=usage, version=version)
        
        parser.add_option("--trn", dest='lTrn',  action="append", type="string"
                          , help="Train or continue previous training session using the given annotated collection.")    
        parser.add_option("--tst", dest='lTst',  action="append", type="string"
                          , help="Test a model using the given annotated collection.")    
        parser.add_option("--run", dest='lRun',  action="append", type="string"
                          , help="Run a model on the given non-annotated collection.")    
        parser.add_option("--fold", dest='lFold',  action="append", type="string"
                          , help="Evaluate by cross-validation a model on the given annotated collection.")    
        parser.add_option("--fold-init", dest='iFoldInitNum',  action="store", type="int"
                          , help="Initialize the file lists for parallel cross-validating a model on the given annotated collection. Indicate the number of folds.")    
        parser.add_option("--fold-run", dest='iFoldRunNum',  action="store", type="int"
                          , help="Run one fold, prepared by --fold-init options. Indicate the fold by its number.")    
        parser.add_option("--fold-finish", dest='bFoldFinish',  action="store_true"
                          , help="Evaluate by cross-validation a model on the given annotated collection.")    
        parser.add_option("-w", "--warm", dest='warm',  action="store_true"
                          , help="To make warm-startable model and warm-start if a model exist already.")   
        parser.add_option("--pkl", dest='pkl', action="store_true"
                          , help="GZip and pickle PyStruct data as (lX, lY), and exit.")    
        parser.add_option("--rm", dest='rm',  action="store_true"
                          , help="Remove all model files")   
        parser.add_option("--crf-njobs", dest='crf_njobs',  action="store", type="int"
                          , help="CRF training parameter njobs")
        parser.add_option("--crf-max_iter"          , dest='crf_max_iter'       ,  action="append", type="int"        #"append" to have a list and possibly do a gridsearch
                          , help="CRF training parameter max_iter")    
        parser.add_option("--crf-C"                 , dest='crf_C'              ,  action="append", type="float"
                          , help="CRF training parameter C")    
        parser.add_option("--crf-tol"               , dest='crf_tol'            ,  action="append", type="float"
                          , help="CRF training parameter tol")    
        parser.add_option("--crf-inference_cache"   , dest='crf_inference_cache',  action="append", type="int"
                          , help="CRF training parameter inference_cache")    
        parser.add_option("--best-params", dest='best_params',  action="store", type="string"
                          , help="Use the best  parameters from the grid search previously done on the given model or model fold") 

        parser.add_option("--storeX" , dest='storeX' ,  action="store", type="string", help="Dev: to be use with --run: load the data and store [X] under given filename, and exit")
        parser.add_option("--applyY" , dest='applyY' ,  action="store", type="string", help="Dev: to be use with --run: load the data, label it using [Y] from given file name, and store the annotated data, and exit")
                
        return usage, None, parser
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
        #we always have one LR model per type  (yes, even for single type ! ;-) )
        lMdl = [ GridSearchCV(LogisticRegression(class_weight='balanced') 
                              , self.dGridSearch_LR_conf, n_jobs=self.dGridSearch_LR_n_jobs) for _ in range(self.iNbCRFType) ]  
            
        if self.isTypedCRF():
            traceln(" - typed CRF")
            tMdl = tuple(lMdl)
            self._lBaselineModel.append( tMdl )
            return tMdl
        else:
            assert len(lMdl) == 1, "internal error"
            [mdl] = lMdl
            self._lBaselineModel.append( mdl )
            return mdl

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
                  , mdl.getBaselineFilename()
                  , mdl._getParamsFilename(self.sModelName, self.sModelDir)       ]:
            if os.path.exists(s):
                self.traceln("\t - rm %s"%s) 
                os.unlink(s)
        if os.path.exists(self.sModelDir) and not os.listdir(self.sModelDir):
            self.traceln("\t - rmdir %s"%self.sModelDir) 
            os.rmdir(self.sModelDir)
        return 
    
    def train_save_test(self, lsTrnColDir, lsTstColDir, bWarm=False, bPickleOnly=False):
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
        ts_trn, lFilename_trn = self.listMaxTimestampFile(lsTrnColDir, self.sXmlFilenamePattern)
        _     , lFilename_tst = self.listMaxTimestampFile(lsTstColDir, self.sXmlFilenamePattern)
        
        self.traceln("- creating a %s model"%self.cModelClass)
        oReport = self._train_save_test(self.sModelName, bWarm, lFilename_trn, ts_trn, lFilename_tst, bPickleOnly)

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
        _     , lFilename_tst = self.listMaxTimestampFile(lsTstColDir, self.sXmlFilenamePattern)
        
        DU_GraphClass = self.cGraphClass
        
        lPageConstraint = DU_GraphClass.getPageConstraint()
        if lPageConstraint: 
            for dat in lPageConstraint: self.traceln("\t\t%s"%str(dat))
        
        try:
            #should work fine
            oReport = self._mdl.testFiles(lFilename_tst, lambda fn: DU_GraphClass.loadGraphs([fn], bDetach=True, bLabelled=True, iVerbose=1))
        except:
            self.traceln("- loading test graphs")
            lGraph_tst = DU_GraphClass.loadGraphs(lFilename_tst, bDetach=True, bLabelled=True, iVerbose=1)
            self.traceln(" %d graphs loaded"%len(lGraph_tst))
            oReport = self._mdl.test(lGraph_tst)

        return oReport


    def predict(self, lsColDir,docid=None):
        """
        Return the list of produced files
        """
        self.traceln("-"*50)
        self.traceln("Predicting for collection(s):", lsColDir)
        self.traceln("-"*50)

        if not self._mdl: raise Exception("The model must be loaded beforehand!")
        
        #list files
        if docid is None:
            _     , lFilename = self.listMaxTimestampFile(lsColDir, self.sXmlFilenamePattern)
        # predict for this file only
        else:
            try: 
                lFilename = [os.path.abspath(os.path.join(lsColDir[0], docid+MultiPageXml.sEXT  ))]
            except IndexError:raise Exception("a collection directory must be provided!")
            
        
        DU_GraphClass = self.getGraphClass()

        lPageConstraint = DU_GraphClass.getPageConstraint()
        if lPageConstraint: 
            for dat in lPageConstraint: self.traceln("\t\t%s"%str(dat))
        
        chronoOn("predict")
        self.traceln("- loading collection as graphs, and processing each in turn. (%d files)"%len(lFilename))
        du_postfix = "_du"+MultiPageXml.sEXT
        lsOutputFilename = []
        for sFilename in lFilename:
            if sFilename.endswith(du_postfix): continue #:)
            chronoOn("predict_1")
            lg = DU_GraphClass.loadGraphs([sFilename], bDetach=False, bLabelled=False, iVerbose=1)
            #normally, we get one graph per file, but in case we load one graph per page, for instance, we have a list
            if lg:
                for g in lg:
                    doc = g.doc
                    if lPageConstraint:
                        self.traceln("\t- prediction with logical constraints: %s"%sFilename)
                    else:
                        self.traceln("\t- prediction : %s"%sFilename)
                    Y = self._mdl.predict(g)
                    
                    g.setDomLabels(Y)
                    del Y
                del lg
                
                MultiPageXml.setMetadata(doc, None, self.sMetadata_Creator, self.sMetadata_Comments)
                sDUFilename = sFilename[:-len(MultiPageXml.sEXT)]+du_postfix
                doc.write(sDUFilename,
                          xml_declaration=True,
                          encoding="utf-8",
                          pretty_print=True
                          #compression=0,  #0 to 9 
                          )

                lsOutputFilename.append(sDUFilename)
            else:
                self.traceln("\t- no prediction to do for: %s"%sFilename)
                
            self.traceln("\t done [%.2fs]"%chronoOff("predict_1"))
        self.traceln(" done [%.2fs]"%chronoOff("predict"))

        return lsOutputFilename

    def runForExternalMLMethod(self, lsColDir, storeX, applyY, bRevertEdges=False):
        """
        HACK: to test new ML methods, not yet integrated in our SW: storeX=None, storeXY=None, applyY=None
        Return the list of produced files
        """

        self.traceln("-"*50)
        if storeX: traceln("Loading data and storing [X] (1 X per graph)")
        if applyY: traceln("Loading data, loading Y, labelling data, storing annotated data")
        self.traceln("-"*50)

        if storeX and applyY:
            raise ValueError("Either store X or applyY, not both")

        if not self._mdl: raise Exception("The model must be loaded beforehand!")
        
        #list files
        _     , lFilename = self.listMaxTimestampFile(lsColDir, self.sXmlFilenamePattern)
        
        DU_GraphClass = self.getGraphClass()

        lPageConstraint = DU_GraphClass.getPageConstraint()
        if lPageConstraint: 
            for dat in lPageConstraint: self.traceln("\t\t%s"%str(dat))
        
        if applyY: 
            self.traceln("LOADING [Y] from %s"%applyY)
            lY = self._mdl.gzip_cPickle_load(applyY)
        if storeX: lX = []
        
        chronoOn("predict")
        self.traceln("- loading collection as graphs, and processing each in turn. (%d files)"%len(lFilename))
        du_postfix = "_du"+MultiPageXml.sEXT
        lsOutputFilename = []
        for sFilename in lFilename:
            if sFilename.endswith(du_postfix): continue #:)
            chronoOn("predict_1")
            lg = DU_GraphClass.loadGraphs([sFilename], bDetach=False, bLabelled=False, iVerbose=1)
            #normally, we get one graph per file, but in case we load one graph per page, for instance, we have a list
            if lg:
                for g in lg:
                    doc = g.doc
                    if bRevertEdges: g.revertEdges()    #revert the directions of the edges
                    if lPageConstraint:
                        self.traceln("\t- prediction with logical constraints: %s"%sFilename)
                    else:
                        self.traceln("\t- prediction : %s"%sFilename)
                    if storeX:
                        [X] = self._mdl.get_lX([g])
                        lX.append(X)
                    else:
                        Y = lY.pop(0)
                        g.setDomLabels(Y)
                del lg
                
                if applyY:
                    MultiPageXml.setMetadata(doc, None, self.sMetadata_Creator, self.sMetadata_Comments)
                    sDUFilename = sFilename[:-len(MultiPageXml.sEXT)]+du_postfix
                    doc.saveFormatFileEnc(sDUFilename, "utf-8", True)  #True to indent the XML
                    doc.freeDoc()
                    lsOutputFilename.append(sDUFilename)
            else:
                self.traceln("\t- no prediction to do for: %s"%sFilename)
                
            self.traceln("\t done [%.2fs]"%chronoOff("predict_1"))
        self.traceln(" done [%.2fs]"%chronoOff("predict"))

        if storeX:
            self.traceln("STORING [X] in %s"%storeX)
            self._mdl.gzip_cPickle_dump(storeX, lX)
            
        return lsOutputFilename

    def checkLabelCoverage(self, lY):
        #check that all classes are represented in the dataset
        #it is done again in train but we do that here also because the extractor can take along time, 
        #   and we may discover afterwards it was a useless dataset.
        aLabelCount, _ = np.histogram( np.hstack(lY) , range(self.nbClass+1))
        traceln("   Labels count: ", aLabelCount, " (%d graphs)"%len(lY))
        traceln("   Labels      : ", self.getGraphClass().getLabelNameList())
        if np.min(aLabelCount) == 0:
            sMsg = "*** ERROR *** Label(s) not observed in data."
            #traceln( sMsg+" Label(s): %s"% np.where(aLabelCount[:] == 0)[0] )
            lMissingLabels = [self.getGraphClass().getLabelNameList()[i] for i in np.where(aLabelCount[:] == 0)[0]]
            traceln( sMsg+" Label(s): %s"% lMissingLabels )
            raise ValueError(sMsg)
        return True

    #-----  NFOLD STUFF
    def _nfold_Init(self, lsTrnColDir, n_splits=3, test_size=0.25, random_state=None, bStoreOnDisk=False):
        """
        initialize a cross-validation
        if bStoreOnDisk is true, store the details of each fold on disk, for paralell execution of each
        return a splitter object, training file timestamp and list 
        """
        self.traceln("-"*50)
        traceln("---------- INITIALIZING CROSS-VALIDATION ----------")
        self.traceln("Model files '%s' in folder '%s'"%(self.sModelName, self.sModelDir))
        #sConfigFile = os.path.join(self.sModelDir, self.sModelName+".py")
        #self.traceln("  Configuration file: %s"%sConfigFile)
        self.traceln("Evaluating with collection(s):", lsTrnColDir)
        self.traceln("-"*50)

        fnCrossValidDetails = os.path.join(self.sModelDir, self.sModelName+"_fold_def.pkl")
        if os.path.exists(fnCrossValidDetails):
            self.traceln("ERROR: I refuse to overwrite an existing CV setup. Remove manually the CV data! (files %s%s%s_fold* )"%(self.sModelDir, os.sep, self.sModelName))
            exit(1)
        
        #list the train files
        ts_trn, lFilename_trn = self.listMaxTimestampFile(lsTrnColDir, self.sXmlFilenamePattern)
        self.traceln("       %d train documents" % len(lFilename_trn))
        
        splitter = ShuffleSplit(n_splits, test_size, random_state)
        
        if bStoreOnDisk:
            
            crf.Model.Model.gzip_cPickle_dump(fnCrossValidDetails
                                              , (lsTrnColDir, n_splits, test_size, random_state))
            
            for i, (train_index, test_index) in enumerate(splitter.split(lFilename_trn)):
                iFold = i + 1
                traceln("---------- FOLD %d ----------"%iFold)
                lFoldFilename_trn = [lFilename_trn[i] for i in train_index]
                lFoldFilename_tst = [lFilename_trn[i] for i in test_index]
                traceln("--- Train with: %s"%lFoldFilename_trn)
                traceln("--- Test  with: %s"%lFoldFilename_tst)
                
                fnFoldDetails = os.path.join(self.sModelDir, self.sModelName+"_fold_%d_def.pkl"%iFold)
                oFoldDetails  = (iFold, ts_trn, lFilename_trn, train_index, test_index)
                crf.Model.Model.gzip_cPickle_dump(fnFoldDetails, oFoldDetails)
                #store the list for TRN and TST in a human readable form
                for name, lFN in [('trn', lFoldFilename_trn), ('tst', lFoldFilename_tst)]:
                    with open(os.path.join(self.sModelDir, self.sModelName+"_fold_%d_def_%s.txt"%(iFold, name)), "w") as fd:
                        fd.write("\n".join(lFN))
                        fd.write("\n")
                traceln("--- Fold info stored in : %s"%fnFoldDetails)
                
        return splitter, ts_trn, lFilename_trn

    def _nfold_RunFoldFromDisk(self, iFold, bWarm=False, bPickleOnly=False):
        """
        Run the fold iFold
        Store results on disk
        """
        fnFoldDetails = os.path.join(self.sModelDir, self.sModelName+"_fold_%d_def.pkl"%abs(iFold))
        traceln("--- Loading fold info from : %s"% fnFoldDetails)
        oFoldDetails = crf.Model.Model.gzip_cPickle_load(fnFoldDetails)
        (iFold_stored, ts_trn, lFilename_trn, train_index, test_index) = oFoldDetails
        assert iFold_stored == abs(iFold), "Internal error. Inconsistent fold details on disk."
        
        if iFold > 0: #normal case
            oReport = self._nfold_RunFold(iFold, ts_trn, lFilename_trn, train_index, test_index, bWarm=bWarm, bPickleOnly=bPickleOnly)
        else:
            traceln("Switching train and test data for fold %d"%abs(iFold))
            oReport = self._nfold_RunFold(iFold, ts_trn, lFilename_trn, test_index, train_index, bWarm=bWarm, bPickleOnly=bPickleOnly)
        
        fnFoldResults = os.path.join(self.sModelDir, self.sModelName+"_fold_%d_TestReport.pkl"%iFold)
        crf.Model.Model.gzip_cPickle_dump(fnFoldResults, oReport)
        traceln(" - Done (fold %d)"%iFold)
        
        return oReport

    def _nfold_Finish(self):
        traceln("---------- SHOWING RESULTS OF CROSS-VALIDATION ----------")
        
        fnCrossValidDetails = os.path.join(self.sModelDir, self.sModelName+"_fold_def.pkl")
        (lsTrnColDir, n_splits, test_size, random_state) = crf.Model.Model.gzip_cPickle_load(fnCrossValidDetails)
        
        loReport = []
        for i in range(n_splits):
            iFold = i + 1
            fnFoldResults = os.path.join(self.sModelDir, self.sModelName+"_fold_%d_TestReport.pkl"%iFold)
            traceln("\t-loading ", fnFoldResults)
            try:
                oReport = crf.Model.Model.gzip_cPickle_load(fnFoldResults)
                
                loReport.append(oReport)
            except:
                traceln("\tWARNING: fold %d has NOT FINISHED or FAILED"%iFold)

        oNFoldReport = TestReportConfusion.newFromReportList(self.sModelName+" (ALL %d FOLDS)"%n_splits, loReport) #a test report based on the confusion matrix

        fnCrossValidDetails = os.path.join(self.sModelDir, self.sModelName+"_folds_STATS.txt")
        with open(fnCrossValidDetails, "a") as fd:
            #BIG banner
            fd.write("\n\n")
            fd.write("#"*80+"\n")
            fd.write("# AGGREGATING FOLDS RESULTS  " + "%s\n"%datetime.datetime.now().isoformat())
            fd.write("#"*80+"\n\n")
            
            for oReport in loReport: 
                fd.write(str(oReport))
                fd.write("%s\n"%(" _"*30))

            fd.write(str(oNFoldReport))
            
        return oNFoldReport

    def _nfold_RunFold(self, iFold, ts_trn, lFilename_trn, train_index, test_index, bWarm=False, bPickleOnly=False):
        """
        Run this fold
        Return a TestReport object
        """
        traceln("---------- RUNNING FOLD %d ----------"%iFold)
        lFoldFilename_trn = [lFilename_trn[i] for i in train_index]
        lFoldFilename_tst = [lFilename_trn[i] for i in test_index]
        traceln("--- Train with: %s"%lFoldFilename_trn)
        traceln("--- Test  with: %s"%lFoldFilename_tst)
        
        self.traceln("- creating a %s model"%self.cModelClass)
        sFoldModelName = self.sModelName+"_fold_%d"%iFold
        
        oReport = self._train_save_test(sFoldModelName, bWarm, lFoldFilename_trn, ts_trn, lFoldFilename_tst, bPickleOnly)

        fnFoldReport = os.path.join(self.sModelDir, self.sModelName+"_fold_%d_STATS.txt"%iFold)
        with open(fnFoldReport, "w") as fd:
            fd.write(str(oReport))
        
        return oReport
    
    def nfold_Eval(self, lsTrnColDir, n_splits=3, test_size=0.25, random_state=None, bPickleOnly=False):
        """
        n-fold evaluation on the training data
        
        - list all files
        - generate a user defined number of independent train / test dataset splits. Samples are first shuffled and then split into a pair of train and test sets
        - for each split: 
            - train a CRF and all baseline model
            - test and make a TestReport
            - save the model
        - return a list of TestReports
        """
        
        splitter, ts_trn, lFilename_trn = self._nfold_Init(lsTrnColDir, n_splits, test_size, random_state)
        
        loTstRpt = []
        
        for i, (train_index, test_index) in enumerate(splitter.split(lFilename_trn)):
            oReport = self._nfold_RunFold(i+1, ts_trn, lFilename_trn, train_index, test_index, bPickleOnly=False)
            traceln(oReport)
            loTstRpt.append(oReport)
        
        return loTstRpt

    #----------------------------------------------------------------------------------------------------------    
    def _pickleData(self, mdl, lGraph, name):
        self.traceln("- Computing data structure of all graphs and features...")
        #for GCN
        bGCN_revert = False
        if bGCN_revert:
            for g in lGraph: g.revertEdges()
        lX, lY = mdl.get_lX_lY(lGraph)
        sFilename = mdl.getTrainDataFilename(name)
        if bGCN_revert:
            sFilename = sFilename.replace("_tlXlY_", "_tlXrlY_")
        self.traceln("- storing (lX, lY) into %s"%sFilename)
        mdl.gzip_cPickle_dump(sFilename, (lX, lY))
        return
                    
    def _train_save_test(self, sModelName, bWarm, lFilename_trn, ts_trn, lFilename_tst, bPickleOnly):
        """
        used both by train_save_test and _nfold_runFold
        """
        mdl = self.cModelClass(sModelName, self.sModelDir)
        
        if os.path.exists(mdl.getModelFilename()) and not bWarm and not bPickleOnly: 
            raise crf.Model.ModelException("Model exists on disk already (%s), either remove it first or warm-start the training."%mdl.getModelFilename())
            
        mdl.configureLearner(**self.config_learner_kwargs)
        mdl.setBaselineModelList(self._lBaselineModel)
        mdl.saveConfiguration( (self.config_extractor_kwargs, self.config_learner_kwargs) )
        self.traceln("\t - configuration: ", self.config_learner_kwargs )

        self.traceln("- loading training graphs")
        lGraph_trn = self.cGraphClass.loadGraphs(lFilename_trn, bDetach=True, bLabelled=True, iVerbose=1)
        self.traceln(" %d graphs loaded"%len(lGraph_trn))

        assert self.nbClass and self.lNbClass, "internal error: I expected the number of class to be automatically computed at that stage"
        if self.iNbCRFType == 1:
            mdl.setNbClass(self.nbClass)
        else:
            mdl.setNbClass(self.lNbClass)

        #for this check, we load the Y once...
        self.checkLabelCoverage(mdl.get_lY(lGraph_trn)) #NOTE that Y are in bad order if multiptypes. Not a pb here
            
        self.traceln("- retrieving or creating feature extractors...")
        chronoOn("FeatExtract")
        try:
            mdl.loadTransformers(ts_trn)
        except crf.Model.ModelException:
            fe = self.cFeatureDefinition(**self.config_extractor_kwargs)         
            fe.fitTranformers(lGraph_trn)
            fe.cleanTransformers()
            mdl.setTranformers(fe.getTransformers())
            mdl.saveTransformers()
        self.traceln(" done [%.1fs]"%chronoOff("FeatExtract"))
        
        if bPickleOnly:
            self._pickleData(mdl, lGraph_trn, "trn")
        else:
            self.traceln("- training model...")
            chronoOn("MdlTrn")
            mdl.train(lGraph_trn, True, ts_trn, verbose=1 if self.bVerbose else 0)
            mdl.save()
            self.traceln(" done [%.1fs]"%chronoOff("MdlTrn"))
        
        # OK!!
        self._mdl = mdl
        
        if lFilename_tst:
            self.traceln("- loading test graphs")
            lGraph_tst = self.cGraphClass.loadGraphs(lFilename_tst, bDetach=True, bLabelled=True, iVerbose=1)
            self.traceln(" %d graphs loaded"%len(lGraph_tst))
            if bPickleOnly:
                self._pickleData(mdl, lGraph_tst, "tst")
            else:
                oReport = mdl.test(lGraph_tst)
        else:
            oReport = None

        if bPickleOnly:
            self.traceln("- pickle done, exiting")
            exit(0)
        
        return oReport
    
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
    
# ------------------------------------------------------------------------------------------------------------------------------
class DU_FactorialCRF_Task(DU_CRF_Task):

    def __init__(self, sModelName, sModelDir, cGraphClass, dLearnerConfig={}, sComment=None
                 , cFeatureDefinition=None, dFeatureConfig={}
                 ): 
        """
        Same as DU_CRF_Task except for the cFeatureConfig
        """
        self.sModelName     = sModelName
        self.sModelDir      = sModelDir
        self.cGraphClass    = cGraphClass
        #Because of the way of dealing with the command line, we may get singleton instead of scalar. We fix this here
        self.config_learner_kwargs      = {k:v[0] if type(v)==types.ListType and len(v)==1 else v for k,v in dLearnerConfig.items()}
        if sComment: self.sMetadata_Comments    = sComment
        
        self._mdl = None
        self._lBaselineModel = []
        self.bVerbose = True
        
        self.iNbCRFType = None #is set below
        
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
        self.iNbCRFType = len(self.cGraphClass.getNodeTypeList())

        self.config_extractor_kwargs = dFeatureConfig

        self.cModelClass = Model_SSVM_AD3 if self.iNbCRFType == 1 else Model_SSVM_AD3_Multitype
        assert issubclass(self.cModelClass, crf.Model.Model), "Your model class must inherit from crf.Model.Model"
        
        
# ------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    version = "v.01"
    usage, description, parser = DU_CRF_Task.getBasicTrnTstRunOptionParser(sys.argv[0], version)

    parser.print_help()
    
    traceln("\nThis module should not be run as command line. It does nothing. (And did nothing!)")
