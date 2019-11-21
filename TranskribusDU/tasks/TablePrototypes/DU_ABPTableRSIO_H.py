# -*- coding: utf-8 -*-

"""
    Example DU task for ABP Table: doing jointly row and header/data
    
    Copyright Naver Labs Europe(C) 2018 H. Déjean, JL Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""




import sys, os
from lxml import etree

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from common.trace import traceln
from tasks import _checkFindColDir, _exit


#from crf.Graph_Multi_SinglePageXml import Graph_MultiSinglePageXml
from crf.factorial.FactorialGraph_MultiPageXml import FactorialGraph_MultiPageXml
from crf.factorial.FactorialGraph_MultiPageXml_Scaffold import FactorialGraph_MultiPageXml_Scaffold

from crf.NodeType_PageXml   import NodeType_PageXml_type_woText
#from tasks.DU_CRF_Task import DU_CRF_Task
from tasks.DU_FactorialCRF_Task import DU_FactorialCRF_Task

#from crf.FeatureDefinition_PageXml_std_noText import FeatureDefinition_PageXml_StandardOnes_noText
from crf.FeatureDefinition_PageXml_std_noText import FeatureDefinition_PageXml_StandardOnes_noText


class NodeType_BIESO_to_SIO_and_CHDO(NodeType_PageXml_type_woText):
    """
    Convert BIESO labeling to SIO
    """
    
    def parseDocNodeLabel(self, graph_node, defaultCls=None):
        """
        Parse and set the graph node label and return its class index
        raise a ValueError if the label is missing while bOther was not True, or if the label is neither a valid one nor an ignored one
        """
        sLabel = self.sDefaultLabel
        domnode = graph_node.node
        
        sXmlLabel = domnode.get(self.sLabelAttr)
        
        sXmlLabel = {'B':'S',
                     'I':'I',
                     'E':'I',
                     'S':'S',
                     'O':'O',
                     'CH':'CH',
                     'D':'D'}[sXmlLabel]
        try:
            sLabel = self.dXmlLabel2Label[sXmlLabel]
        except KeyError:
            #not a label of interest
            try:
                self.checkIsIgnored(sXmlLabel)
                #if self.lsXmlIgnoredLabel and sXmlLabel not in self.lsXmlIgnoredLabel: 
            except:
                raise ValueError("Invalid label '%s'"
                                 " (from @%s or @%s) in node %s"%(sXmlLabel,
                                                           self.sLabelAttr,
                                                           self.sDefaultLabel,
                                                           etree.tostring(domnode)))
        
        return sLabel

 
class DU_ABPTableRH(DU_FactorialCRF_Task):
    """
    We will do a CRF model for a DU task
    , with the below labels 
    """
    sXmlFilenamePattern = "*.mpxml"
    
    #sLabeledXmlFilenamePattern = "*.a_mpxml"
    #WHY THIS ? sLabeledXmlFilenamePattern = "*.mpxml"

    #WHY THIS ? sLabeledXmlFilenameEXT = ".mpxml"

    bScaffold = None
    
    #=== CONFIGURATION ====================================================================
    @classmethod
    def getConfiguredGraphClass(cls):
        """
        In this class method, we must return a configured graph class
        """
        
        lLabelsSIO_R            = ['S', 'I', 'O']  #O?
        lLabels_COLUMN_HEADER   = ['CH', 'D', 'O',]
        
#         """
#         if you play with a toy collection, which does not have all expected classes, you can reduce those.
#         """
#         
#         lActuallySeen = None
#         if lActuallySeen:
#             print( "REDUCING THE CLASSES TO THOSE SEEN IN TRAINING")
#             lIgnoredLabels  = [lLabelsR[i] for i in range(len(lLabelsR)) if i not in lActuallySeen]
#             lLabels         = [lLabelsR[i] for i in lActuallySeen ]
#             print( len(lLabelsR)          , lLabelsR)
#             print( len(lIgnoredLabels)   , lIgnoredLabels)
        
        #DEFINING THE CLASS OF GRAPH WE USE
        if cls.bScaffold is None: raise Exception("Internal error")
        if cls.bScaffold:
            DU_GRAPH = FactorialGraph_MultiPageXml_Scaffold
        else:
            DU_GRAPH = FactorialGraph_MultiPageXml
        
        # ROW
        ntR = NodeType_BIESO_to_SIO_and_CHDO("row"
                              , lLabelsSIO_R
                              , None
                              , False    #no label means OTHER
                              , BBoxDeltaFun=lambda v: max(v * 0.066, min(5, v/3))  #we reduce overlap in this way
                              )
        ntR.setLabelAttribute("DU_row")
        ntR.setXpathExpr( (".//pc:TextLine"        #how to find the nodes
                          , "./pc:TextEquiv")       #how to get their text
                       )
        DU_GRAPH.addNodeType(ntR)
        
        # HEADER
        ntH = NodeType_BIESO_to_SIO_and_CHDO("hdr"
                              , lLabels_COLUMN_HEADER
                              , None
                              , False    #no label means OTHER
                              , BBoxDeltaFun=lambda v: max(v * 0.066, min(5, v/3))  #we reduce overlap in this way
                              )
        ntH.setLabelAttribute("DU_header")
        ntH.setXpathExpr( (".//pc:TextLine"        #how to find the nodes
                          , "./pc:TextEquiv")       #how to get their text
                       )
        DU_GRAPH.addNodeType(ntH)        
        
        
        return DU_GRAPH
        
    def __init__(self, sModelName, sModelDir, sComment=None,
                 C=None, tol=None, njobs=None, max_iter=None,
                 inference_cache=None,
                 bScaffold=False): 
        
        DU_ABPTableRH.bScaffold = bScaffold
        
        DU_FactorialCRF_Task.__init__(self
                     , sModelName, sModelDir
                     , dFeatureConfig = {  }
                     , dLearnerConfig = {
                                   'C'                : .1   if C               is None else C
                                 , 'njobs'            : 4    if njobs           is None else njobs
                                 , 'inference_cache'  : 50   if inference_cache is None else inference_cache
                                 #, 'tol'              : .1
                                 , 'tol'              : .05  if tol             is None else tol
                                 , 'save_every'       : 50     #save every 50 iterations,for warm start
                                 , 'max_iter'         : 10   if max_iter        is None else max_iter
                         }
                     , sComment=sComment
                     #,cFeatureDefinition=FeatureDefinition_PageXml_StandardOnes_noText
                     ,cFeatureDefinition=FeatureDefinition_PageXml_StandardOnes_noText
                     )
        
        #self.setNbClass(3)     #so that we check if all classes are represented in the training set
        
#         if options.bBaseline:
#             self.bsln_mdl = self.addBaseline_LogisticRegression()    #use a LR model trained by GridSearch as baseline
    #=== END OF CONFIGURATION =============================================================

  
    def predict(self, lsColDir):
        """
        Return the list of produced files
        """
        self.sXmlFilenamePattern = "*.mpxml"
        return DU_FactorialCRF_Task.predict(self, lsColDir)
        
    def runForExternalMLMethod(self, lsColDir, storeX, applyY, bRevertEdges=False):
        """
        Return the list of produced files
        """
        self.sXmlFilenamePattern = "*.mpxml"
        return DU_FactorialCRF_Task.runForExternalMLMethod(self, lsColDir, storeX, applyY, bRevertEdges)
              

# ----------------------------------------------------------------------------

def main(sModelDir, sModelName, options):
    doer = DU_ABPTableRH(sModelName, sModelDir,
                      C                 = options.crf_C,
                      tol               = options.crf_tol,
                      njobs             = options.crf_njobs,
                      max_iter          = options.max_iter,
                      inference_cache   = options.crf_inference_cache,
                      bScaffold         = options.bScaffold)
    
    if options.rm:
        doer.rm()
        return

    lTrn, lTst, lRun, lFold = [_checkFindColDir(lsDir, bAbsolute=False) for lsDir in [options.lTrn, options.lTst, options.lRun, options.lFold]] 
#     if options.bAnnotate:
#         doer.annotateDocument(lTrn)
#         traceln('annotation done')    
#         sys.exit(0)
    
    
    traceln("- classes: ", doer.getGraphClass().getLabelNameList())
    
    ## use. a_mpxml files
    #doer.sXmlFilenamePattern = doer.sLabeledXmlFilenamePattern


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
            oReport = doer._nfold_RunFoldFromDisk(options.iFoldRunNum, options.warm, options.pkl)
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
        loTstRpt = doer.nfold_Eval(lFold, 3, .25, None, options.pkl)
        import graph.GraphModel
        sReportPickleFilename = os.path.join(sModelDir, sModelName + "__report.txt")
        traceln("Results are in %s"%sReportPickleFilename)
        graph.GraphModel.GraphModel.gzip_cPickle_dump(sReportPickleFilename, loTstRpt)
    elif lTrn:
        doer.train_save_test(lTrn, lTst, options.warm, options.pkl)
        try:    traceln("Baseline best estimator: %s"%doer.bsln_mdl.best_params_)   #for GridSearch
        except: pass
        traceln(" --- CRF Model ---")
        traceln(doer.getModel().getModelInfo())
    elif lTst:
        doer.load()
        tstReport = doer.test(lTst)
        traceln(tstReport)
        if options.bDetailedReport:
            traceln(tstReport.getDetailledReport())
            sReportPickleFilename = os.path.join(sModelDir, sModelName + "__detailled_report.txt")
            graph.GraphModel.GraphModel.gzip_cPickle_dump(sReportPickleFilename, tstReport)
    
    if lRun:
        if options.storeX or options.applyY:
            try: doer.load() 
            except: pass    #we only need the transformer
            lsOutputFilename = doer.runForExternalMLMethod(lRun, options.storeX, options.applyY, options.bRevertEdges)
        else:
            doer.load()
            lsOutputFilename = doer.predict(lRun)
            
        traceln("Done, see in:\n  %s"%lsOutputFilename)
    
        
# ----------------------------------------------------------------------------
if __name__ == "__main__":

    version = "v.01"
    usage, description, parser = DU_FactorialCRF_Task.getBasicTrnTstRunOptionParser(sys.argv[0], version)
#     parser.add_option("--annotate", dest='bAnnotate',  action="store_true",default=False,  help="Annotate the textlines with BIES labels")    

    #FOR GCN
    parser.add_option("--revertEdges", dest='bRevertEdges',  action="store_true", help="Revert the direction of the edges") 
    parser.add_option("--detail", dest='bDetailedReport',  action="store_true", default=False,help="Display detailled reporting (score per document)") 
    parser.add_option("--baseline", dest='bBaseline',  action="store_true", default=False, help="report baseline method") 
    parser.add_option("--scaffold", dest='bScaffold',  action="store_true", default=False, help="scaffold factorial") 

            
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