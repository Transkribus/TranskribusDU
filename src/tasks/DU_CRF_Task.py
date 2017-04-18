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
import os, glob, datetime
from optparse import OptionParser

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit

#sklearn has changed and sklearn.grid_search.GridSearchCV will disappear in next release or so
#so it is recommended to use instead sklearn.model_selection
#BUT on Linux, unplickling of the model fails
#=> change only on Windows
#JLM 2017-03-10
#Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#   File "/opt/STools/python/2.7-x86_64/lib/python2.7/pickle.py", line 1378, in load
#     return Unpickler(file).load()
#   File "/opt/STools/python/2.7-x86_64/lib/python2.7/pickle.py", line 858, in load
#     dispatch[key](self)
#   File "/opt/STools/python/2.7-x86_64/lib/python2.7/pickle.py", line 1217, in load_build
#     setstate(state)
#   File "/opt/STools/python/2.7-x86_64/lib/python2.7/site-packages/numpy-1.12.0.dev0+25d60a9-py2.7-linux-x86_64.egg/numpy/ma/core.py", line 5875, in __setstate__
#     super(MaskedArray, self).__setstate__((shp, typ, isf, raw))
# TypeError: object pickle not returning list

import sys
try:
    #pickling fails on 0.18.1 on Linux
    from sklearn.model_selection import GridSearchCV  #0.18.1
except ImportError:
    #sklearn 0.18
    from sklearn.grid_search import GridSearchCV
    
try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from common.trace import traceln

import crf.Model
from crf.Model_SSVM_AD3 import Model_SSVM_AD3
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
    
    #dGridSearch_LR_conf = {'C':[0.1, 0.5, 1.0, 2.0] }  #Grid search parameters for LR baseline method training
    dGridSearch_LR_conf   = {'C':[0.01, 0.1, 1.0, 10.0] }  #Grid search parameters for LR baseline method training
    dGridSearch_LR_n_jobs = 4                              #Grid search: number of jobs
    
    sXmlFilenamePattern = "*[0-9]"+MultiPageXml.sEXT    #how to find the Xml files

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
        self.nbClass        = None  #if we know the number of class, we check that the training set covers all
        self._lBaselineModel = []
        self.bVerbose = True
        
        if cFeatureDefinition: self.cFeatureDefinition = cFeatureDefinition
        assert issubclass(self.cModelClass, crf.Model.Model), "Your model class must inherit from crf.Model.Model"
        assert issubclass(self.cFeatureDefinition, crf.FeatureDefinition.FeatureDefinition), "Your feature definition class must inherit from crf.FeatureDefinition.FeatureDefinition"
    
    #---  CONFIGURATION setters --------------------------------------------------------------------
    def setModelClass(self, cModelClass): 
        self.cModelClass = cModelClass
        assert issubclass(self.cModelClass, crf.Model.Model), "Your model class must inherit from crf.Model.Model"
    def getModelClass(self):
        return self.cModelClass
    
    def getModel(self):
        return self._mdl
    
    def setLearnerConfiguration(self, dParams):
        self.config_learner_kwargs = dParams
        
    def setFeatureDefinition(self, cFeatureDefinition): 
        self.cFeatureDefinition = cFeatureDefinition
        assert issubclass(self.cFeatureDefinition, crf.FeatureDefinition.FeatureDefinition), "Your feature definition class must inherit from crf.FeatureDefinition.FeatureDefinition"

    def setNbClass(self, nbClass):
        self.nbClass = nbClass
        
    #---  COMMAND LINE PARSZER --------------------------------------------------------------------
    def getBasicTrnTstRunOptionParser(cls, sys_argv0=None, version=""):
        usage = """"%s <model-folder> <model-name> [--rm] [--trn <col-dir> [--warm]]+ [--tst <col-dir>]+ [--run <col-dir>]+
or for a cross-validation [--fold-init <N>] [--fold-run <n> [-w]] [--fold-finish] [--fold <col-dir>]+
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
        lr = LogisticRegression(class_weight='balanced')
        mdl = GridSearchCV(lr , self.dGridSearch_LR_conf, n_jobs=self.dGridSearch_LR_n_jobs)        
        self._lBaselineModel.append(mdl)
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
                  , mdl._getParamsFilename()       ]:
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
        ts_trn, lFilename_trn = self.listMaxTimestampFile(lsTrnColDir, self.sXmlFilenamePattern)
        _     , lFilename_tst = self.listMaxTimestampFile(lsTstColDir, self.sXmlFilenamePattern)
        
        DU_GraphClass = self.cGraphClass
        
        self.traceln("- creating a %s model"%self.cModelClass)
        mdl = self.cModelClass(self.sModelName, self.sModelDir)
        
        if not bWarm:
            if os.path.exists(mdl.getModelFilename()): 
                raise crf.Model.ModelException("Model exists on disk already (%s), either remove it first or warm-start the training."%mdl.getModelFilename())
            
        mdl.configureLearner(**self.config_learner_kwargs)
        mdl.setBaselineModelList(self._lBaselineModel)
        mdl.saveConfiguration( (self.config_extractor_kwargs, self.config_learner_kwargs) )
        self.traceln("\t - configuration: ", self.config_learner_kwargs )

        self.traceln("- loading training graphs")
        lGraph_trn = DU_GraphClass.loadGraphs(lFilename_trn, bDetach=True, bLabelled=True, iVerbose=1)
        self.traceln(" %d graphs loaded"%len(lGraph_trn))

        if self.nbClass is None:
            traceln("Unknown number of expected classes: cannot check if the training set covers all of them.")
        else:
            #for this check, we load the Y once...
            self.checkLabelCoverage(mdl.get_lY(lGraph_trn))

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
        _     , lFilename_tst = self.listMaxTimestampFile(lsTstColDir, self.sXmlFilenamePattern)
        
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
        _     , lFilename = self.listMaxTimestampFile(lsColDir, self.sXmlFilenamePattern)
        
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

    def checkLabelCoverage(self, lY):
        #check that all classes are represented in the dataset
        #it is done again in train but we do that here also because the extractor can take along time, 
        #   and we may discover afterwards it was a useless dataset.
        aLabelCount, _ = np.histogram( np.hstack(lY) , self.nbClass+1)
        traceln("   Labels count: ", aLabelCount, " (%d graphs)"%len(lY))
        if np.min(aLabelCount) == 0:
            sMsg = "*** ERROR *** Label(s) not observed in data."
            traceln( sMsg+" Label(s): %s"% np.where(aLabelCount[:] == 0)[0] )
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
        sConfigFile = os.path.join(self.sModelDir, self.sModelName+".py")
        self.traceln("  Configuration file: %s"%sConfigFile)
        self.traceln("Evaluating with collection(s):", lsTrnColDir)
        self.traceln("-"*50)
        
        #list the train files
        ts_trn, lFilename_trn = self.listMaxTimestampFile(lsTrnColDir, self.sXmlFilenamePattern)
        self.traceln("       %d train documents" % len(lFilename_trn))
        
        splitter = ShuffleSplit(n_splits, test_size, random_state)
        
        if bStoreOnDisk:
            
            fnCrossValidDetails = os.path.join(self.sModelDir, self.sModelName+"_fold_def.pkl")
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
                traceln("--- Fold info stored in : %s"%fnFoldDetails)
                
        return splitter, ts_trn, lFilename_trn

    def _nfold_RunFoldFromDisk(self, iFold, bWarm=False):
        """
        Run the fold iFold
        Store results on disk
        """
        fnFoldDetails = os.path.join(self.sModelDir, self.sModelName+"_fold_%d_def.pkl"%iFold)
        traceln("--- Loading fold info from : %s"% fnFoldDetails)
        oFoldDetails = crf.Model.Model.gzip_cPickle_load(fnFoldDetails)
        (iFold_stored, ts_trn, lFilename_trn, train_index, test_index) = oFoldDetails
        assert iFold_stored == iFold, "Internal error. Inconsistent fold details on disk."
        
        oReport = self._nfold_RunFold(iFold, ts_trn, lFilename_trn, train_index, test_index, bWarm=bWarm)
        
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
            try:
                oReport = crf.Model.Model.gzip_cPickle_load(fnFoldResults)
                
                loReport.append(oReport)
            except:
                traceln("WARNING: fold %d has NOT FINISHED or FAILED"%iFold)

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

    def _nfold_RunFold(self, iFold, ts_trn, lFilename_trn, train_index, test_index, bWarm=False):
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
        
        mdl = self.cModelClass(sFoldModelName, self.sModelDir)
        
        if os.path.exists(mdl.getModelFilename()) and not bWarm: 
            raise crf.Model.ModelException("Model exists on disk already (%s), either remove it first or warm-start the training."%mdl.getModelFilename())
            
        mdl.configureLearner(**self.config_learner_kwargs)
        mdl.setBaselineModelList(self._lBaselineModel)
        mdl.saveConfiguration( (self.config_extractor_kwargs, self.config_learner_kwargs) )
        self.traceln("\t - configuration: ", self.config_learner_kwargs )

        self.traceln("- loading training graphs")
        lGraph_trn = self.cGraphClass.loadGraphs(lFoldFilename_trn, bDetach=True, bLabelled=True, iVerbose=1)
        self.traceln(" %d graphs loaded"%len(lGraph_trn))

        if self.nbClass is None:
            traceln("Unknown number of expected classes: cannot check if the training set covers all of them.")
        else:
            #for this check, we load the Y once...
            self.checkLabelCoverage(mdl.get_lY(lGraph_trn))
            
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
        
        if lFoldFilename_tst:
            self.traceln("- loading test graphs")
            lGraph_tst = self.cGraphClass.loadGraphs(lFoldFilename_tst, bDetach=True, bLabelled=True, iVerbose=1)
            self.traceln(" %d graphs loaded"%len(lGraph_tst))
    
            oReport = mdl.test(lGraph_tst)
        else:
            oReport = None

        fnFoldReport = os.path.join(self.sModelDir, self.sModelName+"_fold_%d_STATS.txt"%iFold)
        with open(fnFoldReport, "w") as fd:
            fd.write(str(oReport))
        
        return oReport
    
    def nfold_Eval(self, lsTrnColDir, n_splits=3, test_size=0.25, random_state=None):
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
            oReport = self._nfold_RunFold(i+1, ts_trn, lFilename_trn, train_index, test_index)
            traceln(oReport)
            loTstRpt.append(oReport)
        
        return loTstRpt

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
    

if __name__ == "__main__":

    version = "v.01"
    usage, description, parser = DU_CRF_Task.getBasicTrnTstRunOptionParser(sys.argv[0], version)

    parser.print_help()
    
    traceln("\nThis module should not be run as command line. It does nothing. (And did nothing!)")