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

from DU_BL_Task import DU_BL_Task




# ===============================================================================================================
#DEFINING THE CLASS OF GRAPH WE USE
DU_GRAPH = Graph_MultiPageXml()
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


dFeatureConfig_Baseline = {'n_tfidf_node': 500
    , 't_ngrams_node': (2, 4)
    , 'b_tfidf_node_lc': False
    , 'n_tfidf_edge': 250
    , 't_ngrams_edge': (2, 4)
    , 'b_tfidf_edge_lc': False,
                           }


dFeatureConfig_FeatSelect = {'n_tfidf_node': 500
    , 't_ngrams_node': (2, 4)
    , 'b_tfidf_node_lc': False
    , 'n_tfidf_edge': 250
    , 't_ngrams_edge': (2, 4)
    , 'b_tfidf_edge_lc': False
    , 'feat_select':'chi2'
                             }


dLearnerConfig={  'C': .1
                            , 'njobs': 4
                            , 'inference_cache': 50
                            , 'tol': .1
                            , 'save_every': 50  # save every 50 iterations,for warm start
                            , 'max_iter': 250
                        }


    # === CONFIGURATION ====================================================================




class DU_BL_V1(DU_BL_Task):

    def __init__(self, sModelName, sModelDir, feat_select=None,sComment=None):
        if feat_select=='chi2':
            DU_CRF_Task.__init__(self, sModelName, sModelDir,
                        DU_GRAPH,
                        dFeatureConfig=dFeatureConfig_FeatSelect,
                        dLearnerConfig=dLearnerConfig,
                        sComment=sComment
                        )
        else:
            DU_CRF_Task.__init__(self, sModelName, sModelDir,
                                DU_GRAPH,
                                dFeatureConfig=dFeatureConfig_Baseline,
                                dLearnerConfig=dLearnerConfig,
                                sComment=sComment
                                )
        self.addBaseline_LogisticRegression()    #use a LR model as baseline

    #
    # #=== CONFIGURATION ====================================================================
    # def __init__(self, sModelName, sModelDir, sComment=None):
    #     DU_CRF_Task.__init__(self
    #                          , sModelName, sModelDir
    #                          , DU_GRAPH
    #                          , dFeatureConfig = {
    #                                 'n_tfidf_node'    : 500
    #                               , 't_ngrams_node'   : (2,4)
    #                               , 'b_tfidf_node_lc' : False
    #                               , 'n_tfidf_edge'    : 250
    #                               , 't_ngrams_edge'   : (2,4)
    #                               , 'b_tfidf_edge_lc' : False
    #                           }
    #                          , dLearnerConfig = {
    #                                 'C'                : .1
    #
    #                              , 'njobs'            : 4
    #                              , 'inference_cache'  : 50
    #                             , 'tol'              : .1
    #                             , 'save_every'       : 50     #save every 50 iterations,for warm start
    #                              , 'max_iter'         : 250
    #                              }
    #                          , sComment=sComment
    #                          )
    #

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

    doer = DU_BL_V1(sModelName, sModelDir,feat_select='mi_rr')

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
        for x in tstReport:
            traceln(x)


    if lRun:
        doer.load()
        lsOutputFilename = doer.predict(lRun)
        traceln("Done, see in:\n  %s"%lsOutputFilename)


