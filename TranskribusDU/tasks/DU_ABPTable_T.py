# -*- coding: utf-8 -*-

"""
    Example DU task for ABP Table that uses the Multi-Type CRF
    
    Copyright Xerox(C) 2017 H. Déjean, JL Meunier

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

from crf.Graph_Multi_SinglePageXml import Graph_MultiSinglePageXml
from crf.NodeType_PageXml   import NodeType_PageXml_type_woText
from tasks.DU_CRF_Task import DU_CRF_Task
from crf.FeatureDefinition_PageXml_std_noText_v3 import FeatureDefinition_T_PageXml_StandardOnes_noText_v3




 
class DU_ABPTable_TypedCRF(DU_CRF_Task):
    """
    We will do a typed CRF model for a DU task
    , with the below labels 
    """
    sXmlFilenamePattern = "*.mpxml"
    
    sLabeledXmlFilenamePattern = "*.mpxml"

    sLabeledXmlFilenameEXT = ".mpxml"

    #=== CONFIGURATION ====================================================================
    @classmethod
    def getConfiguredGraphClass(cls):
        """
        In this class method, we must return a configured graph class
        """

    # ===============================================================================================================
    #DEFINING THE CLASS OF GRAPH WE USE
    DU_GRAPH = Graph_MultiSinglePageXml

    lLabels1 = ['RB', 'RI', 'RE', 'RS','RO']
    lIgnoredLabels1 = None
    # """
    # if you play with a toy collection, which does not have all expected classes, you can reduce those.
    # """
    # 
    # lActuallySeen = None
    # if lActuallySeen:
    #     print "REDUCING THE CLASSES TO THOSE SEEN IN TRAINING"
    #     lIgnoredLabels  = [lLabels[i] for i in range(len(lLabels)) if i not in lActuallySeen]
    #     lLabels         = [lLabels[i] for i in lActuallySeen ]
    #     print len(lLabels)          , lLabels
    #     print len(lIgnoredLabels)   , lIgnoredLabels
    #     nbClass = len(lLabels) + 1  #because the ignored labels will become OTHER
    
    nt1 = NodeType_PageXml_type_woText("text"                   #some short prefix because labels below are prefixed with it
                          , lLabels1
                          , lIgnoredLabels1
                          , False    #no label means OTHER
                          , BBoxDeltaFun=lambda v: max(v * 0.066, min(5, v/3))  #we reduce overlap in this way
                          )
    nt1.setXpathExpr( (".//pc:TextLine"        #how to find the nodes
                      , "./pc:TextEquiv")       #how to get their text
                   )
    DU_GRAPH.addNodeType(nt1)
    
    nt2 = NodeType_PageXml_type_woText("sprtr"                   #some short prefix because labels below are prefixed with it
                          , ['SI', 'SO']
                          , None
                          , False    #no label means OTHER
                          , BBoxDeltaFun=lambda v: max(v * 0.066, min(5, v/3))  #we reduce overlap in this way
                          )
    nt2.setXpathExpr( (".//pc:SeparatorRegion"  #how to find the nodes
                      , "./pc:TextEquiv")       #how to get their text  (no text in fact)
                   )
    DU_GRAPH.addNodeType(nt2)    


    #=== CONFIGURATION ====================================================================
    def __init__(self, sModelName, sModelDir, sComment=None, C=None, tol=None, njobs=None, max_iter=None, inference_cache=None): 
        
        #another way to specify the graph class
        # defining a  getConfiguredGraphClass is preferred
        self.configureGraphClass(self.DU_GRAPH)

        DU_CRF_Task.__init__(self
                     , sModelName, sModelDir
                     , dLearnerConfig = {
                                   'C'                : .1   if C               is None else C
                                 , 'njobs'            : 8    if njobs           is None else njobs
                                 , 'inference_cache'  : 50   if inference_cache is None else inference_cache
                                 #, 'tol'              : .1
                                 , 'tol'              : .05  if tol             is None else tol
                                 , 'save_every'       : 50     #save every 50 iterations,for warm start
                                 , 'max_iter'         : 1000 if max_iter        is None else max_iter
                         }
                     , sComment=sComment
                     , cFeatureDefinition=FeatureDefinition_T_PageXml_StandardOnes_noText_v3
                     , dFeatureConfig = {
                         #config for the extractor of nodes of each type
                         "text": None,    
                         "sprtr": None,
                         #config for the extractor of edges of each type
                         "text_text": None,    
                         "text_sprtr": None,    
                         "sprtr_text": None,    
                         "sprtr_sprtr": None    
                         }
                     )
        
        traceln("- classes: ", self.DU_GRAPH.getLabelNameList())

        self.bsln_mdl = self.addBaseline_LogisticRegression()    #use a LR model trained by GridSearch as baseline
    
    #=== END OF CONFIGURATION =============================================================

  
    def predict(self, lsColDir,sDocId):
        """
        Return the list of produced files
        """
#         self.sXmlFilenamePattern = "*.a_mpxml"
        return DU_CRF_Task.predict(self, lsColDir,sDocId)

           
# ----------------------------------------------------------------------------
def main(sModelDir, sModelName, options):
    doer = DU_ABPTable_TypedCRF(sModelName, sModelDir,
                      C                 = options.crf_C,
                      tol               = options.crf_tol,
                      njobs             = options.crf_njobs,
                      max_iter          = options.max_iter,
                      inference_cache   = options.crf_inference_cache)
    
    
    if options.docid:
        sDocId=options.docid
    else:
        sDocId=None
    if options.rm:
        doer.rm()
        return

    lTrn, lTst, lRun, lFold = [_checkFindColDir(lsDir) for lsDir in [options.lTrn, options.lTst, options.lRun, options.lFold]] 
#     if options.bAnnotate:
#         doer.annotateDocument(lTrn)
#         traceln('annotation done')    
#         sys.exit(0)
    
    ## use. a_mpxml files
    doer.sXmlFilenamePattern = doer.sLabeledXmlFilenamePattern


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
        lsOutputFilename = doer.predict(lRun,sDocId)
        traceln("Done, see in:\n  %s"%lsOutputFilename)
    

# ----------------------------------------------------------------------------
if __name__ == "__main__":

    version = "v.01"
    usage, description, parser = DU_CRF_Task.getBasicTrnTstRunOptionParser(sys.argv[0], version)
    parser.add_option("--docid", dest='docid',  action="store",default=None,  help="only process docid")    
    # --- 
    #parse the command line
    (options, args) = parser.parse_args()
    
    # --- 
    try:
        sModelDir, sModelName = args
    except Exception as e:
        traceln("Specify a model folder and a model name!")
        _exit(usage, 1, e)
    
    main(sModelDir, sModelName, options)
