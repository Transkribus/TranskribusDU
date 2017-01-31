# -*- coding: utf-8 -*-

import os, glob
from optparse import OptionParser

from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV

import sys, os
import numpy as np

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from common.trace import traceln
from tasks import _checkFindColDir, _exit

from crf.Graph_MultiPageXml import Graph_MultiPageXml
from crf.NodeType_PageXml   import NodeType_PageXml
from tasks.DU_CRF_Task import DU_CRF_Task



from common.trace import traceln

import crf.Model
from crf.BaselineModel import BaselineModel
from xml_formats.PageXml import MultiPageXml
import crf.FeatureDefinition
#from crf.FeatureDefinition_PageXml_std import FeatureDefinition_PageXml_StandardOnes
from crf.FeatureDefinition_PageXml_FeatSelect import FeatureDefinition_PageXml_FeatSelect

import pdb

class DU_BL_Task(DU_CRF_Task):
    cModelClass          = BaselineModel
    cFeatureDefinition   = FeatureDefinition_PageXml_FeatSelect

    sMetadata_Creator = "XRCE RXCE- v-0.000001"
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

        #mdl.configureLearner(**self.config_learner_kwargs)
        mdl.setBaselineModelList(self._lBaselineModel[0])
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
            lY = [g.buildLabelMatrix() for g in lGraph_trn]
            lY_flat = np.hstack(lY)
            fe.fitTranformers(lGraph_trn,lY_flat)
            fe.cleanTransformers()
            mdl.setTranformers(fe.getTransformers())
            mdl.saveTransformers()
        self.traceln(" done")

        self.traceln("- training model...")

        #TODO Now do the connection in the BaselineModel with the Graph
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



# ===============================================================================================================
#DEFINING THE CLASS OF GRAPH WE USE
DU_GRAPH = Graph_MultiPageXml
nt = NodeType_PageXml("TR"                   #some short prefix because labels below are prefixed with it
                      , ['catch-word', 'header', 'heading', 'marginalia', 'page-number']   #EXACTLY as in GT data!!!!
                      , []      #no ignored label/ One of those above or nothing, otherwise Exception!!
                      , True    #no label means OTHER
                      )
nt.setXpathExpr( (".//pc:TextRegion"        #how to find the nodes
                  , "./pc:TextEquiv")       #how to get their text
               )
DU_GRAPH.addNodeType(nt)
# ===============================================================================================================


class DU_BL_V1(DU_BL_Task):
    """
    We will do a CRF model for a DU task
    , working on a MultiPageXMl document at TextRegion level
    , with the below labels
    """

    #=== CONFIGURATION ====================================================================
    def __init__(self, sModelName, sModelDir, sComment=None):
        DU_CRF_Task.__init__(self
                             , sModelName, sModelDir
                             , DU_GRAPH
                             , dFeatureConfig = {
                                    'n_tfidf_node'    : 500
                                  , 't_ngrams_node'   : (2,4)
                                  , 'b_tfidf_node_lc' : False
                                  , 'n_tfidf_edge'    : 250
                                  , 't_ngrams_edge'   : (2,4)
                                  , 'b_tfidf_edge_lc' : False
                              }
                             , dLearnerConfig = {
                                    'C'                : .1

                                 , 'njobs'            : 4
                                 , 'inference_cache'  : 50
                                , 'tol'              : .1
                                , 'save_every'       : 50     #save every 50 iterations,for warm start
                                 , 'max_iter'         : 250
                                 }
                             , sComment=sComment
                             )
        self.addBaseline_LogisticRegression()    #use a LR model as baseline

    #=== END OF CONFIGURATION =============================================================


if __name__ == "__main__":





    version = "v.01"
    usage, description, parser = DU_BL_Task.getBasicTrnTstRunOptionParser(sys.argv[0], version)

    # ---
    #parse the command line
    (options, args) = parser.parse_args()
    # ---
    try:
        sModelDir, sModelName = args
    except Exception as e:
        _exit(usage, 1, e)

    doer = DU_BL_V1(sModelName, sModelDir)

    if options.rm:
        doer.rm()
        sys.exit(0)

    traceln("- classes: ", DU_GRAPH.getLabelNameList())


    #Add the "col" subdir if needed
    lTrn, lTst, lRun = [_checkFindColDir(lsDir) for lsDir in [options.lTrn, options.lTst, options.lRun]]

    if lTrn:
        doer.train_save_test(lTrn, lTst, options.warm)

    elif lTst:
        doer.load()
        tstReport = doer.test(lTst)
        traceln(tstReport)

    if lRun:
        doer.load()
        lsOutputFilename = doer.predict(lRun)
        traceln("Done, see in:\n  %s"%lsOutputFilename)


