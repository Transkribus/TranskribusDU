# -*- coding: utf-8 -*-

"""
    Example DU task for Dodge, using the logit textual feature extractor
    
    Copyright Xerox(C) 2017 JL. Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""
import sys, os

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from common.trace import traceln
from tasks import _checkFindColDir, _exit

from graph.Graph_MultiPageXml import Graph_MultiPageXml
from graph.NodeType_PageXml   import NodeType_PageXml_type_NestedText
from tasks.DU_Task_Factory import DU_Task_Factory
from tasks.DU_CRF_Task import DU_CRF_Task
from graph.FeatureDefinition_PageXml_logit_v2 import FeatureDefinition_PageXml_LogitExtractorV2

# ===============================================================================================================

lLabels =  ['TOC-entry'         #0
            , 'caption'
            , 'catch-word'
                         , 'footer'
                         , 'footnote'                #4
                         , 'footnote-continued'
                         , 'header'             #6
						 , 'heading'          #7
                         , 'marginalia'
                         , 'page-number'    #9
                         , 'paragraph'    #10
                         , 'signature-mark']   
lIgnoredLabels = None

nbClass = len(lLabels)

"""
if you play with a toy collection, which does not have all expected classes, you can reduce those.
"""
lActuallySeen = [4, 6, 7, 9, 10]
#lActuallySeen = [4, 6]
"""
                0-            TOC-entry    5940 occurences       (   2%)  (   2%)
                1-              caption     707 occurences       (   0%)  (   0%)
                2-           catch-word     201 occurences       (   0%)  (   0%)
                3-               footer      11 occurences       (   0%)  (   0%)
                4-             footnote   36942 occurences       (  11%)  (  11%)
                5-   footnote-continued    1890 occurences       (   1%)  (   1%)
                6-               header   15910 occurences       (   5%)  (   5%)
                7-              heading   18032 occurences       (   6%)  (   6%)
                8-           marginalia    4292 occurences       (   1%)  (   1%)
                9-          page-number   40236 occurences       (  12%)  (  12%)
               10-            paragraph  194927 occurences       (  60%)  (  60%)
               11-       signature-mark    4894 occurences       (   2%)  (   2%)
"""
lActuallySeen = None
if lActuallySeen:
    traceln("REDUCING THE CLASSES TO THOSE SEEN IN TRAINING")
    lIgnoredLabels  = [lLabels[i] for i in range(len(lLabels)) if i not in lActuallySeen]
    lLabels         = [lLabels[i] for i in lActuallySeen ]
    traceln(len(lLabels)          , lLabels)
    traceln(len(lIgnoredLabels)   , lIgnoredLabels)
    nbClass = len(lLabels) + 1  #because the ignored labels will become OTHER

    #DEFINING THE CLASS OF GRAPH WE USE
    DU_GRAPH = Graph_MultiPageXml
    nt = NodeType_PageXml_type_NestedText("gtb"                   #some short prefix because labels below are prefixed with it
                          , lLabels
                          , lIgnoredLabels
                              , True    #no label means OTHER
                              )
else:
    #DEFINING THE CLASS OF GRAPH WE USE
    DU_GRAPH = Graph_MultiPageXml
    nt = NodeType_PageXml_type_NestedText("gtb"                   #some short prefix because labels below are prefixed with it
                          , lLabels
                          , lIgnoredLabels
                      , False    #no label means OTHER
                      )
nt.setXpathExpr( (".//pc:TextRegion"        #how to find the nodes
                  , "./pc:TextEquiv")       #how to get their text
               )
DU_GRAPH.addNodeType(nt)

"""
The constraints must be a list of tuples like ( <operator>, <NodeType>, <states>, <negated> )
where:
- operator is one of 'XOR' 'XOROUT' 'ATMOSTONE' 'OR' 'OROUT' 'ANDOUT' 'IMPLY'
- states is a list of unary state names, 1 per involved unary. If the states are all the same, you can pass it directly as a single string.
- negated is a list of boolean indicated if the unary must be negated. Again, if all values are the same, pass a single boolean value instead of a list 
"""
if False:
    DU_GRAPH.setPageConstraint( [    ('ATMOSTONE', nt, 'pnum' , False)    #0 or 1 catch_word per page
                                   , ('ATMOSTONE', nt, 'title'    , False)    #0 or 1 heading pare page
                                 ] )

# ===============================================================================================================

 
class DU_GTBooks(DU_CRF_Task):
    """
    We will do a CRF model for a DU task
    , working on a DS XML document at BLOCK level
    , with the below labels 
    """
    sXmlFilenamePattern = "*.mpxml"

#     #In case you want to change the Logistic Regression gird search parameters...
#     dGridSearch_LR_conf  = {'C':[0.01, 0.1, 1.0, 10.0] }   #Grid search parameters for LR baseline method training
#     dGridSearch_LR_n_jobs = 4                              #Grid search: number of jobs
    
    #=== CONFIGURATION ====================================================================
    def __init__(self, sModelName, sModelDir, sComment=None, C=None, tol=None, njobs=None, max_iter=None, inference_cache=None): 
        #NOTE: we might get a list in C tol max_iter inference_cache  (in case of gridsearch)
        
        DU_CRF_Task.__init__(self
                             , sModelName, sModelDir
                             , DU_GRAPH
                             , dFeatureConfig = {
                                    'nbClass'    : nbClass
                                  , 't_ngrams_node'   : (2,4)
                                  , 'b_node_lc' : False    
                                  , 't_ngrams_edge'   : (2,4)
                                  , 'b_edge_lc' : False    
                                  , 'n_jobs'      : 5         #n_jobs when fitting the internal Logit feat extractor model by grid search
                              }
                             , dLearnerConfig = {
                                   'C'                : .1   if C               is None else C
                                 , 'njobs'            : 5    if njobs           is None else njobs
                                 , 'inference_cache'  : 50   if inference_cache is None else inference_cache
                                 #, 'tol'              : .1
                                 , 'tol'              : .05  if tol             is None else tol
                                 , 'save_every'       : 50     #save every 50 iterations,for warm start
                                 , 'max_iter'         : 1000 if max_iter        is None else max_iter
                                 }
                             , sComment=sComment
                             , cFeatureDefinition=FeatureDefinition_PageXml_LogitExtractorV2
                             )
        
        self.setNbClass(nbClass)     #so that we check if all classes are represented in the training set
        
        self.bsln_mdl = self.addBaseline_LogisticRegression()    #use a LR model trained by GridSearch as baseline
    #=== END OF CONFIGURATION =============================================================


if __name__ == "__main__":

    version = "v.01"
    usage, description, parser = DU_Task_Factory.getStandardOptionsParser(sys.argv[0], version)

    # --- 
    #parse the command line
    (options, args) = parser.parse_args()

    # --- 
    try:
        sModelDir, sModelName = args
    except Exception as e:
        traceln("Specify a model folder and a model name!")
        _exit(usage, 1, e)
        
    doer = DU_GTBooks(sModelName, sModelDir,
                      C                 = options.crf_C,
                      tol               = options.crf_tol,
                      njobs             = options.crf_njobs,
                      max_iter          = options.max_iter,
                      inference_cache   = options.crf_inference_cache)
    
    if options.rm:
        doer.rm()
        sys.exit(0)
    
    traceln("- classes: ", DU_GRAPH.getLabelNameList())
    
    if options.best_params:
        dBestParams = doer.getModelClass().loadBestParams(sModelDir, options.best_params) 
        doer.setLearnerConfiguration(dBestParams)
        
    lTrn, lTst, lRun, lFold = [_checkFindColDir(lsDir) for lsDir in [options.lTrn, options.lTst, options.lRun, options.lFold]] 

    if options.iFoldInitNum or options.iFoldRunNum or options.bFoldFinish:
        if options.iFoldInitNum:
            """
            initialization of a cross-validation
            """
            splitter, ts_trn, lFilename_trn = doer._nfold_Init(lFold, options.iFoldInitNum, bStoreOnDisk=True)
        elif options.iFoldRunNum:
            """
            Run one fold
            """
            oReport = doer._nfold_RunFoldFromDisk(options.iFoldRunNum, options.warm)
            traceln(oReport)
        elif options.bFoldFinish:
            tstReport = doer._nfold_Finish()
            traceln(tstReport)
        else:
            assert False, "Internal error"    
        #no more processing!!
        exit(0)
        #-------------------
        
    if lFold:
        loTstRpt = doer.nfold_Eval(lFold, 3, .25, None)
        import graph.GraphModel
        sReportPickleFilename = os.path.join(sModelDir, sModelName + "__report.txt")
        traceln("Results are in %s"%sReportPickleFilename)
        graph.GraphModel.GraphModel.gzip_cPickle_dump(sReportPickleFilename, loTstRpt)
    elif lTrn:
        doer.train_save_test(lTrn, lTst, options.warm)
        try:    traceln("Baseline best estimator: %s"%doer.bsln_mdl.best_params_)   #for GridSearch
        except: pass
        traceln(" --- CRF Model ---")
        traceln(doer.getModel().getModelInfo())
    elif lTst:
        doer.load()
        tstReport = doer.test(lTst)
        traceln(tstReport)
    
    if lRun:
        doer.load()
        lsOutputFilename = doer.predict(lRun)
        traceln("Done, see in:\n  %s"%lsOutputFilename)
    
