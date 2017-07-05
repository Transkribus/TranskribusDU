# -*- coding: utf-8 -*-

"""
    Example DU task for plan dataset, using the logit textual feature extractor
    
    Copyright Xerox(C) 2017 JL. Meunier

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
    from the European Union's Horizon 2020 research and innovation programme 
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

from crf.Graph_DSXml import Graph_DSXml
from crf.NodeType_DSXml   import NodeType_DS
from DU_CRF_Task import DU_CRF_Task
from crf.FeatureDefinition_PageXml_logit_v2 import FeatureDefinition_PageXml_LogitExtractorV2

# ===============================================================================================================
#DEFINING THE CLASS OF GRAPH WE USE
DU_GRAPH = Graph_DSXml
nt = NodeType_DS("Ddg"                   #some short prefix because labels below are prefixed with it
                      , ['title', 'pnum']   #EXACTLY as in GT data!!!!
                      , []      #no ignored label/ One of those above or nothing, otherwise Exception!!
                      , True    #no label means OTHER
                      )
nt.setXpathExpr( ".//BLOCK"        #how to find the nodes
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

 
class DU_Dodge_d(DU_CRF_Task):
    """
    We will do a CRF model for a DU task
    , working on a DS XML document at BLOCK level
    , with the below labels 
    """
    sXmlFilenamePattern = "*_ds.xml"
    
    #=== CONFIGURATION ====================================================================
    def __init__(self, sModelName, sModelDir, sComment=None, C=None, tol=None, njobs=None, max_iter=None, inference_cache=None): 
        #NOTE: we might get a list in C tol max_iter inference_cache  (in case of gridsearch)
        
        DU_CRF_Task.__init__(self
                             , sModelName, sModelDir
                             , DU_GRAPH
                             , dFeatureConfig = {
                                    'nbClass'    : 3
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
                                 , 'tol'              : .05  if tol             is None else tol
                                 , 'save_every'       : 50     #save every 50 iterations,for warm start
                                 , 'max_iter'         : 1000 if max_iter        is None else max_iter
                                 }
                             , sComment=sComment
                             , cFeatureDefinition=FeatureDefinition_PageXml_LogitExtractorV2
                             )
        
        self.setNbClass(3)     #so that we check if all classes are represented in the training set
        
        self.bsln_mdl = self.addBaseline_LogisticRegression()    #use a LR model trained by GridSearch as baseline
    #=== END OF CONFIGURATION =============================================================


if __name__ == "__main__":

    version = "v.01"
    usage, description, parser = DU_CRF_Task.getBasicTrnTstRunOptionParser(sys.argv[0], version)

    # --- 
    #parse the command line
    (options, args) = parser.parse_args()
    # --- 
    try:
        sModelDir, sModelName = args
    except Exception as e:
        traceln("Specify a model folder and a model name!")
        _exit(usage, 1, e)
        
    doer = DU_Dodge_d(sModelName, sModelDir,
                      C                 = options.crf_C,
                      tol               = options.crf_tol,
                      njobs             = options.crf_njobs,
                      max_iter          = options.crf_max_iter,
                      inference_cache   = options.crf_inference_cache)
    
    if options.rm:
        doer.rm()
        sys.exit(0)
    
    traceln("- classes: ", DU_GRAPH.getLabelNameList())
    
    if options.best_params:
        dBestParams = doer.getModelClass().loadBestParams(sModelDir, options.best_params) 
        doer.setLearnerConfiguration(dBestParams)
        
    #Add the "out" subdir if needed
    lTrn, lTst, lRun, lFold = [_checkFindColDir(lsDir, "out") for lsDir in [options.lTrn, options.lTst, options.lRun, options.lFold]] 

    if options.iFoldInitNum or options.iFoldRunNum or options.bFoldFinish:
        if options.iFoldInitNum:
            """
            initialization of a cross-validation
            """
            splitter, ts_trn, lFilename_trn = doer._nfold_Init(lFold, options.iFoldInitNum, test_size=0.25, random_state=None, bStoreOnDisk=True)
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
        import crf.Model
        sReportPickleFilename = os.path.join(sModelDir, sModelName + "__report.txt")
        traceln("Results are in %s"%sReportPickleFilename)
        crf.Model.Model.gzip_cPickle_dump(sReportPickleFilename, loTstRpt)
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
    
