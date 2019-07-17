# -*- coding: utf-8 -*-

"""
    DU task core.
    
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
import json
from importlib import import_module
import random

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV  #0.18.1 REQUIRES NUMPY 1.12.1 or more recent
    
try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from common.trace import trace, traceln
from common.chrono import chronoOn, chronoOff
from common.TestReport import TestReportConfusion
from xml_formats.PageXml import MultiPageXml

from graph.GraphModel import GraphModel, GraphModelException
import graph.FeatureDefinition
from tasks import _checkFindColDir, _exit


class DU_Task:
    """
Document Understanding class

USAGE:
- get a CRF- or ECN-based instance by the getDoer class method
- choose a model name and a folder where it will be stored
- define the features and their configuration, or a list of (feature definition and its configuration)
- choose to learn using CRF or ECN (via command line options or method)
- define the learner configuration
- run the do method

    doer = DU_Task.getDoer(sModelDir, sModelName, getConfiguredGraphClass, options=.., bCRF=.., bECN=.., )
    # here, you can configure it further if needed ...
    # use the standard command line options to do something
    doer.standardDo(options)
    del doer
    
METHODs:
- training: train_save_test
- loading a trained model: load
- testing a trained model: test
- removing a model from the disk: rm
- if you want to use specific features: setFeatureDefinition
- if you want to also train/test against some baseline model: setBaselineList, addBaseline_LogisticRegression 

See DU_StAZH_b.py

    """
    VERSION          = "DU_Task_v19" 
    
    version          = None     # dynamically computed
    
    cModelClass      = None     #depends on the number of node types!
    cGraphClass      = None     #class of graph in use 
    
    cFeatureDefinition   = None # FeatureDefinition_PageXml_StandardOnes   #I keep this for backward compa
    
    sMetadata_Creator = "NLE Document Understanding"
    sMetadata_Comments = ""

    #dGridSearch_LR_conf = {'C':[0.1, 0.5, 1.0, 2.0] }  #Grid search parameters for LR baseline method training
    dGridSearch_LR_conf   = {'C':[0.01, 0.1, 1.0, 10.0] }  #Grid search parameters for LR baseline method training
    dGridSearch_LR_n_jobs = 4                              #Grid search: number of jobs
    
    # why this old-stuff??  sXmlFilenamePattern = "*[0-9]"+MultiPageXml.sEXT    #how to find the Xml files
    sXmlFilenamePattern = "*"+MultiPageXml.sEXT    #how to find the Xml files

    iNbNodeType = 1     # as of today, only CRF can do multitype
    
    def configureGraphClass(self, configuredClass=None):
        """
        class method to set the graph class ONCE (subsequent calls are ignored)
        """
        if self.cGraphClass is None: #OK, let's set the class attribute!
            
            #if nothing in parameter, or we call the class method
            if configuredClass is None:
                configuredClass = self.getConfiguredGraphClass()
                    
                assert configuredClass is not None, "getConfiguredGraphClass returned None"
                
            self.cGraphClass = configuredClass

        assert self.cGraphClass is not None
        traceln("SETUP: Graph class is %s  (graph mode %d)" % (self.cGraphClass, self.cGraphClass.getGraphMode()))
        
        return self.cGraphClass

    @staticmethod
    def DYNAMIC_IMPORT(name, package=None):
        chronoOn("import")
        trace("SETUP: Dynamic import of '%s' from '%s'" % (name, package))
        m = import_module(name, package)
        traceln("  done [%.1fs]" % chronoOff("import"))
        return m

    
    def __init__(self, sModelName, sModelDir
                 , sComment             = None
                 , cFeatureDefinition   = None
                 , dFeatureConfig       = {}
                 ): 
        self.sModelName    = sModelName
        self.sModelDir     = sModelDir
        if sComment: 
            self.sMetadata_Comments    = sComment

        self.configureGraphClass()

        
        self._mdl = None
        
        # for the conjugate mode
        self.bConjugate = False
        self.nbEdgeClass = None
                
        self._lBaselineModel = []
        self.bVerbose = True
        
        #for single- or multi-type CRF, the same applies!
        self.lNbClass = [len(nt.getLabelNameList()) for nt in self.cGraphClass.getNodeTypeList()]
        self.nbClass = sum(self.lNbClass)
        self.iNbNodeType = len(self.cGraphClass.getNodeTypeList())

        #--- feature definition and configuration per type
        #Feature definition and their config
        if cFeatureDefinition: self.cFeatureDefinition  = cFeatureDefinition
        assert issubclass(self.cFeatureDefinition, graph.FeatureDefinition.FeatureDefinition), "Your feature definition class must inherit from graph.FeatureDefinition.FeatureDefinition"
        
        self.config_extractor_kwargs = dFeatureConfig

        self.config_learner_kwargs = None   # to be set
        self.cModelClass = None             # to be specialized

    #---  DOER --------------------------------------------------------------------
    @classmethod
    def getVersion(cls):
        return str(cls.VERSION) 

    def standardDo(self, options):
        """
        do whatever is reuested by an option from the parsed command line
        
        return None
        """
        if options.rm:
            self.rm()
            return
    
        lTrn, lTst, lRun, lFold = [_checkFindColDir(lsDir) for lsDir in [options.lTrn, options.lTst, options.lRun, options.lFold]]
        
        # Validation set if any
        try:
            ratio_train_val = float(options.lVld[0])
            lVld            = []
            if ratio_train_val <= 0 or 1.0 <= ratio_train_val: raise Exception("Bad ratio, not in ]0, 1[")
        except:
            ratio_train_val = None
            lVld            = _checkFindColDir(options.lVld)
                
        #traceln("- classes: ", doer.getGraphClass().getLabelNameList())
    
        ## use. a_mpxml files
        #doer.sXmlFilenamePattern = doer.sLabeledXmlFilenamePattern
    
        if options.iFoldInitNum or options.iFoldRunNum or options.bFoldFinish:
            if options.iFoldInitNum:
                """
                initialization of a cross-validation
                """
                splitter, ts_trn, lFilename_trn = self._nfold_Init(lFold, options.iFoldInitNum, test_size=0.25, random_state=None, bStoreOnDisk=True)
            elif options.iFoldRunNum:
                """
                Run one fold
                """
                oReport = self._nfold_RunFoldFromDisk(options.iFoldRunNum, options.warm, options.bPkl)
                traceln(oReport)
            elif options.bFoldFinish:
                tstReport = self._nfold_Finish()
                traceln(tstReport)
            else:
                assert False, "Internal error"

            return
    
    
        if lFold:
            loTstRpt = self.nfold_Eval(lFold, 3, .25, None, options.bPkl)
            sReportPickleFilename = os.path.join(self.sModelDir, self.sModelName + "__report.txt")
            traceln("Results are in %s"%sReportPickleFilename)
            GraphModel.gzip_cPickle_dump(sReportPickleFilename, loTstRpt)
        elif lTrn or lTst or lRun:
            if lTrn:
                tstReport = self.train_save_test(lTrn, lTst, lVld, options.warm, options.bPkl
                                                 , ratio_train_val=ratio_train_val)
                try:    traceln("Baseline best estimator: %s"%self.bsln_mdl.best_params_)   #for GridSearch
                except: pass
                traceln(self.getModel().getModelInfo())
                if lTst:
                    traceln(tstReport)
                    if options.bDetailedReport:
                        traceln(tstReport.getDetailledReport())
            elif lTst:
                self.load()
                tstReport = self.test(lTst)
                traceln(tstReport)
                if options.bDetailedReport:
                    traceln(tstReport.getDetailledReport())
                    for test in lTst:
                        sReportPickleFilename = os.path.join('..',test, self.sModelName + "__report.pkl")
                        traceln('Report dumped into %s'%sReportPickleFilename)
                        GraphModel.gzip_cPickle_dump(sReportPickleFilename, tstReport)
        
            if lRun:
#                 if options.storeX or options.applyY:
#                     try: self.load()
#                     except: pass    #we only need the transformer
#                     lsOutputFilename = self.runForExternalMLMethod(lRun, options.storeX, options.applyY, options.bRevertEdges)
#                 else:
                self.load()
                lsOutputFilename = self.predict(lRun, bGraph=options.bGraph)
        
                traceln("Done, see in:\n  %s"%lsOutputFilename)
        else:
            traceln("No action specified in command line. Doing nothing... :)")
            
        return

    def __del__(self):
        """
        trying to clean big objects
        """
        del self._mdl
        del self._lBaselineModel
        del self.cFeatureDefinition
        del self.cModelClass         
    
        
    #---  CONFIGURATION setters --------------------------------------------------------------------
    def getGraphClass(self):    
        return self.cGraphClass
    
    def setModelClass(self, cModelClass): 
        self.cModelClass = cModelClass
        traceln("SETUP: Model class changed to ", self.cModelClass.__name__)
        assert issubclass(self.cModelClass, GraphModel), "Your model class must inherit from graph.GraphModel.GraphModel"
    
    def getModelClass(self):    
        return self.cModelClass
    
    def getModel(self):         
        return self._mdl
    
    def setLearnerConfiguration(self, dLearnerConfig, bCommandLineInput=False):
        if bCommandLineInput:
            # Because of the way of dealing with the command line, we may get singleton instead of scalar. We fix this here
            self.config_learner_kwargs      = {k:v[0] if type(v) is list and len(v)==1 else v for k,v in dLearnerConfig.items()}
        else:
            self.config_learner_kwargs = dLearnerConfig
        
        traceln("SETUP:", json.dumps(self.config_learner_kwargs
                                     , sort_keys=True, indent=4))
                
                

    def getStandardLearnerConfig(self, options):
        """
        Once the command line has been parsed, you can get the standard learner
        configuration dictionary from here.
        """
        traceln("WARNING: no learner configuration by default!!!")
        return {}
    
    """
    When some class is not represented on some graph, you must specify the number of class (per type if multiple types)
    Otherwise pystruct will complain about the number of states differeing from the number of weights
    """
    def setNbClass(self, useless_stuff):    #DEPRECATED - DO NOT CALL!! Number of class computed automatically
        traceln("SETUP: *** setNbClass is deprecated - update your code (but it should work fine!)")
        
    def getNbClass(self): #OK
        """
        return the total number of classes
        """
        return self.nbClass
    
    def setConjugateMode(self
                         , lEdgeLabel       = None    # list of labels (list of strings, or of int)
                         , funEdgeLabel_get = None    # to compute the edge labels
                         , funEdgeLabel_set = None    # to use the predicted edge labels
                         ):
        """
        to learn and predict on the conjugate graph instead of the usual graph.
        1 - The usual graph is created as always
        2 - the function <funEdgeLabel_computer> is called on each edge to compute the edge label
        3 - the conjugate is created and used for learning or predicting
        4 - the function <funEdgeLabel_applier> is called on each edge to exploit the edge predicted label
        
        
        The prototype of the functions are:
            funEdgeLabel_get(primal_graph, primal_X, primal_Y) 
                -> dual_Y
            funEdgeLabel_set(primal_graph, nd_node, edge_matrix, dual_Y) 
                -> None
                
            the dual_Y has as many rows as the primal Edge array, and in the same order
            this order also corresponds to the lEdge attribute of the graph object
            
        In case the graph has some pre-established settings, you can omit the parameters.
        """
        self.bConjugate = True
        self.cModelClass.setConjugateMode()
        self.cGraphClass.setConjugateMode(lEdgeLabel
                                         , funEdgeLabel_get
                                         , funEdgeLabel_set)
        self.nbEdgeLabel = len(self.cGraphClass.getEdgeLabelNameList())
        
        return self.bConjugate
        
                
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
        mdl = GridSearchCV(LogisticRegression(class_weight='balanced') 
                        , self.dGridSearch_LR_conf
                        , n_jobs=self.dGridSearch_LR_n_jobs)  
            
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
    
    def train_save_test(self, lsTrnColDir, lsTstColDir, lsVldColDir, bWarm=False, bPickleXY=False
                        , ratio_train_val=None):
        """
        - Train a model on the tTRN collections, if not empty.
        - Test the trained model using the lTST collections, if not empty.
        - Also train/test any baseline model associated to the main model.
        - Trained models are saved on disk, for testing, redicting or further training (by warm-start)
        - if bWarm==True: warm-start the training from any data stored on disk. Otherwise, a non-empty model folder raises a GraphModelException
        return a test report object
        """
        self.traceln("-"*50)
        self.traceln("Model files of '%s' in folder '%s'"%(self.sModelName, self.sModelDir))
        self.traceln("Training with collection(s):", lsTrnColDir)
        self.traceln("Testing with  collection(s):", lsTstColDir)
        if lsVldColDir: self.traceln("Validating with  collection(s):", lsVldColDir)
        self.traceln("  (File pattern is %s)" % self.sXmlFilenamePattern)
        self.traceln("-"*50)
        
        #list the train and test files
        #NOTE: we check the presence of a digit before the '.' to eclude the *_du.xml files
        ts_trn, lFilename_trn = self.listMaxTimestampFile(lsTrnColDir, self.sXmlFilenamePattern)
        _     , lFilename_tst = self.listMaxTimestampFile(lsTstColDir, self.sXmlFilenamePattern)
        _     , lFilename_vld = self.listMaxTimestampFile(lsVldColDir, self.sXmlFilenamePattern)
        
        self.traceln("- creating a %s model"%self.cModelClass)
        oReport = self._train_save_test(self.sModelName, bWarm, lFilename_trn, ts_trn, lFilename_tst, lFilename_vld, bPickleXY
                                        , ratio_train_val=ratio_train_val)


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
        
        if True:
            oReport = self._mdl.testFiles(lFilename_tst, lambda fn: DU_GraphClass.loadGraphs(self.cGraphClass, [fn], bDetach=True, bLabelled=True, iVerbose=1)
                                          , self.getBaselineList() != [])
        else:
            self.traceln("- loading test graphs")
            lGraph_tst = DU_GraphClass.loadGraphs(self.cGraphClass, lFilename_tst, bDetach=True, bLabelled=True, iVerbose=1)
            if self.bConjugate:
                for _g in lGraph_tst: _g.computeEdgeLabels()
            
            self.traceln(" %d graphs loaded"%len(lGraph_tst))
            oReport = self._mdl.test(lGraph_tst)

        return oReport


    def predict(self, lsColDir, docid=None, bGraph=False):
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
            except IndexError:
                raise Exception("a collection directory must be provided!")


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
            lg = DU_GraphClass.loadGraphs(self.cGraphClass, [sFilename], bDetach=False, bLabelled=False, iVerbose=1)
            #normally, we get one graph per file, but in case we load one graph per page, for instance, we have a list
            if lg:
                for i, g in enumerate(lg):
                    doc = g.doc
                    if lPageConstraint:
                        self.traceln("\t- prediction with logical constraints: %s"%sFilename)
                    else:
                        self.traceln("\t- prediction : %s"%sFilename)

                    if self.bConjugate:
                        Y = self._mdl.predict(g, bProba=True)
                        g.exploitEdgeLabels(Y)
                    else:
                        Y = self._mdl.predict(g)
                        g.setDomLabels(Y)
                    if bGraph: g.addEdgeToDOM(Y)
                    del Y

                MultiPageXml.setMetadata(doc, None, self.sMetadata_Creator, self.sMetadata_Comments)
                sDUFilename = sFilename[:-len(MultiPageXml.sEXT)]              +du_postfix
                doc.write(sDUFilename,
                          xml_declaration=True,
                          encoding="utf-8",
                          pretty_print=True
                          #compression=0,  #0 to 9
                          )
                del doc
                del lg

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
            lg = DU_GraphClass.loadGraphs(self.cGraphClass, [sFilename], bDetach=False, bLabelled=False, iVerbose=1)
            #normally, we get one graph per file, but in case we load one graph per page, for instance, we have a list
            if lg:
                for g in lg:
                    if self.bConjugate: g.computeEdgeLabels()
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
    def _nfold_Init(self, lsTrnColDir, n_splits=3, test_size=None, random_state=None, bStoreOnDisk=False):
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

        fnCrossValidDetails = os.path.join(self.sModelDir, "fold_def.pkl")
        if os.path.exists(fnCrossValidDetails):
            self.traceln("ERROR: I refuse to overwrite an existing CV setup. Remove manually the CV data! (files %s%s%s_fold* )"%(self.sModelDir, os.sep, self.sModelName))
            exit(1)
        
        #list the train files
        traceln(" - looking for %s files in %s"%(self.sXmlFilenamePattern, lsTrnColDir))
        ts_trn, lFilename_trn = self.listMaxTimestampFile(lsTrnColDir, self.sXmlFilenamePattern)
        self.traceln("       %d train documents" % len(lFilename_trn))
        
        if test_size is None:
            test_size = 1.0 / n_splits
        splitter = ShuffleSplit(n_splits, test_size, random_state)
        
        if bStoreOnDisk:
            
            GraphModel.gzip_cPickle_dump(fnCrossValidDetails
                                              , (lsTrnColDir, n_splits, test_size, random_state))
            
            for i, (train_index, test_index) in enumerate(splitter.split(lFilename_trn)):
                iFold = i + 1
                traceln("---------- FOLD %d ----------"%iFold)
                lFoldFilename_trn = [lFilename_trn[i] for i in train_index]
                lFoldFilename_tst = [lFilename_trn[i] for i in test_index]
                traceln("--- Train with: %s files"%len(lFoldFilename_trn))
                traceln("--- Test  with: %s files"%len(lFoldFilename_tst))
                assert not( set(lFoldFilename_trn).intersection(lFoldFilename_tst)), set(lFoldFilename_trn).intersection(lFoldFilename_tst)
                
                #fnFoldDetails = os.path.join(self.sModelDir, self.sModelName+"_fold_%d_def.pkl"%iFold)
                fnFoldDetails = os.path.join(self.sModelDir, "fold_%d_def.pkl" % iFold)
                oFoldDetails  = (iFold, ts_trn, lFilename_trn, train_index, test_index)
                GraphModel.gzip_cPickle_dump(fnFoldDetails, oFoldDetails)
                #store the list for TRN and TST in a human readable form
                for name, lFN in [('trn', lFoldFilename_trn), ('tst', lFoldFilename_tst)]:
                    #with open(os.path.join(self.sModelDir, self.sModelName+"_fold_%d_def_%s.txt"%(iFold, name)), "w") as fd:
                    with open(os.path.join(self.sModelDir, "fold_%d_def_%s.txt" % (iFold, name)),
                              "w") as fd:
                        fd.write("\n".join(lFN))
                        fd.write("\n")
                traceln("--- Fold info stored in : %s"%fnFoldDetails)
                
        return splitter, ts_trn, lFilename_trn

    def _nfold_RunFoldFromDisk(self, iFold, bWarm=False, bPickleXY=False):
        """
        Run the fold iFold
        Store results on disk
        """
        fnFoldDetails = os.path.join(self.sModelDir, "fold_%d_def.pkl"%abs(iFold))

        if os.path.exists(fnFoldDetails) is False:
            try:
                import fnmatch
                #Try to take an existing fold definition
                modelsFiles = os.listdir(self.sModelDir)
                found_files = fnmatch.filter(modelsFiles, '*'+"_fold_%d_def.pkl"%abs(iFold))
                if len(found_files)==1:
                    traceln('Found an existing Fold definition:',found_files[0])
                    fnFoldDetails=os.path.join(self.sModelDir,found_files[0])
                else:
                    raise Exception('Could not find a fold definition')
            except ImportError:
                traceln('Could not load Python 3 fnmatch module ')

        traceln("--- Loading fold info from : %s"% fnFoldDetails)
        oFoldDetails = GraphModel.gzip_cPickle_load(fnFoldDetails)
        (iFold_stored, ts_trn, lFilename_trn, train_index, test_index) = oFoldDetails
        assert iFold_stored == abs(iFold), "Internal error. Inconsistent fold details on disk."
        
        if iFold > 0: #normal case
            oReport = self._nfold_RunFold(iFold, ts_trn, lFilename_trn, train_index, test_index, bWarm=bWarm, bPickleXY=bPickleXY)
        else:
            traceln("Switching train and test data for fold %d"%abs(iFold))
            oReport = self._nfold_RunFold(iFold, ts_trn, lFilename_trn, test_index, train_index, bWarm=bWarm, bPickleXY=bPickleXY)
        
        fnFoldResults = os.path.join(self.sModelDir, self.sModelName+"_fold_%d_TestReport.pkl"%iFold)
        GraphModel.gzip_cPickle_dump(fnFoldResults, oReport)
        traceln(" - Done (fold %d)"%iFold)
        
        return oReport

    def _nfold_Finish(self):
        traceln("---------- SHOWING RESULTS OF CROSS-VALIDATION ----------")
        
        fnCrossValidDetails = os.path.join(self.sModelDir, "fold_def.pkl")
        (lsTrnColDir, n_splits, test_size, random_state) = GraphModel.gzip_cPickle_load(fnCrossValidDetails)
        
        loReport = []
        for i in range(n_splits):
            iFold = i + 1
            fnFoldResults = os.path.join(self.sModelDir, self.sModelName+"_fold_%d_TestReport.pkl"%iFold)
            traceln("\t-loading ", fnFoldResults)
            try:
                oReport = GraphModel.gzip_cPickle_load(fnFoldResults)
                
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

    def _nfold_RunFold(self, iFold, ts_trn, lFilename_trn, train_index, test_index, bWarm=False, bPickleXY=False):
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
        
        oReport = self._train_save_test(sFoldModelName, bWarm, lFoldFilename_trn, ts_trn, lFoldFilename_tst, [], bPickleXY)

        fnFoldReport = os.path.join(self.sModelDir, self.sModelName+"_fold_%d_STATS.txt"%iFold)
        with open(fnFoldReport, "w") as fd:
            fd.write(str(oReport))
        
        return oReport
    
    def nfold_Eval(self, lsTrnColDir, n_splits=3, test_size=None, random_state=None, bPickleXY=False):
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
            oReport = self._nfold_RunFold(i+1, ts_trn, lFilename_trn, train_index, test_index, bPickleXY=False)
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
                    
    def _train_save_test(self, sModelName, bWarm, lFilename_trn, ts_trn, lFilename_tst, lFilename_vld, bPickleXY
                         , ratio_train_val=None):
        """
        used both by train_save_test and _nfold_runFold
        if provided, try using lFilename_vld as validation set to make best model.
        """
        mdl = self.cModelClass(sModelName, self.sModelDir)
        
        if os.path.exists(mdl.getModelFilename()) and not bWarm: 
            raise GraphModelException("Model exists on disk already (%s), either remove it first or warm-start the training."%mdl.getModelFilename())
            
        mdl.configureLearner(**self.config_learner_kwargs)
        mdl.setBaselineModelList(self._lBaselineModel)
        mdl.saveConfiguration( (self.config_extractor_kwargs, self.config_learner_kwargs) )
        self.traceln("\t - configuration: ", self.config_learner_kwargs )

        self.traceln("- loading training graphs")
        lGraph_trn = self.cGraphClass.loadGraphs(self.cGraphClass, lFilename_trn, bDetach=True, bLabelled=True, iVerbose=1)
        self.traceln(" %d training graphs loaded"%len(lGraph_trn))

        if lFilename_vld:
            self.traceln("- loading validation graphs")
            lGraph_vld = self.cGraphClass.loadGraphs(self.cGraphClass, lFilename_vld, bDetach=True, bLabelled=True, iVerbose=1)
            self.traceln(" %d validation graphs loaded"%len(lGraph_vld))
        else:
            lGraph_vld = []
            if ratio_train_val is None:
                self.traceln("- no validation graphs")
            else:
                lG = [g for g in lGraph_trn]
                split_idx = int(ratio_train_val * len(lG))
                lGraph_vld = lG[:split_idx ]
                lGraph_trn = lG[ split_idx:]
                del lG
                self.traceln("- extracted %d validation graphs, got %d training graphs (ratio=%.3f)" 
                             % (len(lGraph_vld), len(lGraph_trn), ratio_train_val))

        #for this check, we load the Y once...
        if self.bConjugate:
            mdl.setNbClass(self.nbEdgeLabel)
            for _g in lGraph_trn: _g.computeEdgeLabels()
            for _g in lGraph_vld: _g.computeEdgeLabels()
        else:
            assert self.nbClass and self.lNbClass, "internal error: I expected the number of class to be automatically computed at that stage"
            if self.iNbNodeType == 1:
                mdl.setNbClass(self.nbClass)
            else:
                mdl.setNbClass(self.lNbClass)
            self.checkLabelCoverage(mdl.get_lY(lGraph_trn)) #NOTE that Y are in bad order if multiptypes. Not a pb here
            
        self.traceln("- retrieving or creating feature extractors...")
        chronoOn("FeatExtract")
        try:
            mdl.loadTransformers(ts_trn)
        except GraphModelException:
            fe = self.cFeatureDefinition(**self.config_extractor_kwargs)         
            fe.fitTranformers(lGraph_trn)
            fe.cleanTransformers()
            mdl.setTranformers(fe.getTransformers())
            mdl.saveTransformers()
        
        # pretty print of features extractors
        self.traceln("""--- Features ---
--- NODES : %s

--- EDGES : %s
--- -------- ---
""" % mdl.getTransformers())
        
        self.traceln(" done [%.1fs]"%chronoOff("FeatExtract"))
        
        if bPickleXY:
            self._pickleData(mdl, lGraph_trn, "trn")

        self.traceln("- training model...")
        chronoOn("MdlTrn")
        mdl.train(lGraph_trn, lGraph_vld, True, ts_trn, verbose=1 if self.bVerbose else 0)
        mdl.save()
        self.traceln(" done [%.1fs]"%chronoOff("MdlTrn"))
        
        # OK!!
        self._mdl = mdl
        
        if lFilename_tst:
            self.traceln("- loading test graphs")
            lGraph_tst = self.cGraphClass.loadGraphs(self.cGraphClass, lFilename_tst, bDetach=True, bLabelled=True, iVerbose=1)
            if self.bConjugate:
                for _g in lGraph_tst: _g.computeEdgeLabels()
            self.traceln(" %d graphs loaded"%len(lGraph_tst))
            if bPickleXY:
                self._pickleData(mdl, lGraph_tst, "tst")
            else:
                oReport = mdl.test(lGraph_tst)
        else:
            oReport = None

        if bPickleXY:
            self.traceln("- pickle done, exiting")
            exit(0)
        
        return oReport
    
    #----------------------------------------------------------------------------------------------------------    
    def listMaxTimestampFile(cls, lsDir, sPattern, bIgnoreDUFiles=True):
        """
        List the file following the given pattern in the given folders
        return the timestamp of the most recent one
        Return a timestamp and a list of filename
        """
        lFn, ts = [], None 
        for sDir in lsDir:
            lsFilename = sorted(glob.iglob(os.path.join(sDir, sPattern)))  
            if bIgnoreDUFiles:
                lsFilename = [s for s in lsFilename if not(os.path.splitext(s)[0].endswith("_du"))]
            lFn.extend([s.replace("\\", "/") for s in lsFilename]) #Unix-style is universal
            if lsFilename:
                ts_max =  max([os.path.getmtime(sFilename) for sFilename in lsFilename])
                ts = ts_max if ts is None else max(ts, ts_max)
                ts = max(ts, ts_max)
        return ts, lFn
    listMaxTimestampFile = classmethod(listMaxTimestampFile)


# ------------------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    usage, parser = DU_Task.getStandardOptionsParser(sys.argv[0])

    parser.print_help()
    
    traceln("\nThis module should not be run as command line. It does nothing. (And did nothing!)")
