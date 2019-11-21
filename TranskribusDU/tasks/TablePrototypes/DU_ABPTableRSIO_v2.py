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

import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from common.trace import traceln
from tasks import _checkFindColDir, _exit

import graph.GraphModel
#from crf.Graph_Multi_SinglePageXml import Graph_MultiSinglePageXml
from crf.Graph_MultiPageXml import Graph_MultiContinousPageXml

from tasks.DU_CRF_Task import DU_CRF_Task

from crf.NodeType_PageXml   import NodeType_PageXml_type_woText

#from crf.FeatureDefinition_PageXml_std_noText import FeatureDefinition_PageXml_StandardOnes_noText
from crf.FeatureDefinition import FeatureDefinition
from crf.Transformer import RobustStandardScaler as StandardScaler
from crf.Transformer import Transformer
from crf.Transformer_PageXml import NodeTransformerXYWH_v2, Node1HotFeatures
from crf.Transformer_PageXml import Edge1HotFeatures, EdgeBooleanFeatures_v2, EdgeNumericalSelector
from crf.PageNumberSimpleSequenciality import PageNumberSimpleSequenciality


#------------------------------------------------------------------------------------------------------
class NodeTransformerNeighbors_v2(Transformer):
    """
    Characterising the neighborough
    """
    def transform(self, lNode):
#         a = np.empty( ( len(lNode), 5 ) , dtype=np.float64)
#         for i, blk in enumerate(lNode): a[i, :] = [blk.x1, blk.y2, blk.x2-blk.x1, blk.y2-blk.y1, blk.fontsize]        #--- 2 3 4 5 6 
        a = np.zeros( ( len(lNode), 5+5 ) , dtype=np.float64)
        for i, blk in enumerate(lNode): 
            ax1, ay1, apnum = blk.x1, blk.y1, blk.pnum
            #number of horizontal/vertical/crosspage neighbors
            nbH, nbV = len(blk.lHNeighbor), len(blk.lVNeighbor)
            if nbH == 0:
                a[i, 0:3] = 1.0
            else:
                nbHafter = sum(1 for _b in blk.lHNeighbor  if _b.x1 > ax1)
                v = (2.0*nbHafter)/nbH - 1            #-1 to +1
                a[i, 1:5] = (1.0 if nbHafter == 0   else 0.0,
                             1.0 if nbHafter == nbH else 0.0,
                             v,
                             v*v)
            if nbV == 0:
                a[i, 5:8] = 1.0
            else:
                nbVafter = sum(1 for _b in blk.lVNeighbor  if _b.y1 > ay1)
                v = (2.0*nbVafter)/nbV - 1            #-1 to +1
                a[i, 6:10] = (1.0 if nbVafter == 0   else 0.0,
                              1.0 if nbVafter == nbV else 0.0,
                              v,
                              v*v)
        return a


class My_FeatureDefinition_v2(FeatureDefinition):

    
    def __init__(self): 
        FeatureDefinition.__init__(self)
        
#         self.n_tfidf_node, self.t_ngrams_node, self.b_tfidf_node_lc = n_tfidf_node, t_ngrams_node, b_tfidf_node_lc
#         self.n_tfidf_edge, self.t_ngrams_edge, self.b_tfidf_edge_lc = n_tfidf_edge, t_ngrams_edge, b_tfidf_edge_lc

#         tdifNodeTextVectorizer = TfidfVectorizer(lowercase=self.b_tfidf_node_lc, max_features=self.n_tfidf_node
#                                                                                   , analyzer = 'char', ngram_range=self.t_ngrams_node #(2,6)
#                                                                                   , dtype=np.float64)
        
        node_transformer = FeatureUnion( [  #CAREFUL IF YOU CHANGE THIS - see cleanTransformers method!!!!
                                    ("xywh", Pipeline([
                                                         ('selector', NodeTransformerXYWH_v2()),
                                                         ('xywh', StandardScaler(copy=False, with_mean=True, with_std=True))  #use in-place scaling
                                                         ])
                                       )
                                    , ("neighbors", NodeTransformerNeighbors_v2())
                                    , ("1hot", Node1HotFeatures())  #does the 1-hot encoding directly
                                      ])
    
        lEdgeFeature = [  #CAREFUL IF YOU CHANGE THIS - see cleanTransformers method!!!!
                                      ("1hot", Edge1HotFeatures(PageNumberSimpleSequenciality()))
                                    , ('boolean', EdgeBooleanFeatures_v2())
                                    , ("numerical", Pipeline([
                                                         ('selector', EdgeNumericalSelector()),
                                                         ('numerical', StandardScaler(copy=False, with_mean=True, with_std=True))  #use in-place scaling
                                                         ])
                                        )
                        ]
                        
        edge_transformer = FeatureUnion( lEdgeFeature )
          
        #return _node_transformer, _edge_transformer, tdifNodeTextVectorizer
        self._node_transformer = node_transformer
        self._edge_transformer = edge_transformer
        self.tfidfNodeTextVectorizer = None #tdifNodeTextVectorizer

   
class NodeType_BIESO_to_SIO(NodeType_PageXml_type_woText):
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
                     'O':'O'}[sXmlLabel]
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

 
class DU_ABPTableR(DU_CRF_Task):
    """
    We will do a CRF model for a DU task
    , with the below labels 
    """
    sXmlFilenamePattern = "*[0-9].mpxml"
    
    #sLabeledXmlFilenamePattern = "*.a_mpxml"
    #WHY THIS ? sLabeledXmlFilenamePattern = "*.mpxml"

    #WHY THIS ? sLabeledXmlFilenameEXT = ".mpxml"

    DU_GRAPH = Graph_MultiContinousPageXml

    #=== CONFIGURATION ====================================================================
    @classmethod
    def getConfiguredGraphClass(cls):
        """
        In this class method, we must return a configured graph class
        """
        
        lLabelsSIO_R  = ['S', 'I', 'O']  #O?
        
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
        DU_GRAPH = Graph_MultiContinousPageXml
        
        # ROW
        ntR = NodeType_BIESO_to_SIO("row"
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
        
        return DU_GRAPH
        
    def __init__(self, sModelName, sModelDir, sComment=None, C=None, tol=None, njobs=None, max_iter=None, inference_cache=None): 

        DU_CRF_Task.__init__(self
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
                     ,cFeatureDefinition=My_FeatureDefinition_v2
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
        return DU_CRF_Task.predict(self, lsColDir)
        
    def runForExternalMLMethod(self, lsColDir, storeX, applyY, bRevertEdges=False):
        """
        Return the list of produced files
        """
        self.sXmlFilenamePattern = "*.mpxml"
        return DU_CRF_Task.runForExternalMLMethod(self, lsColDir, storeX, applyY, bRevertEdges)
              

# ----------------------------------------------------------------------------

def main(sModelDir, sModelName, options):
    doer = DU_ABPTableR(sModelName, sModelDir,
                      C                 = options.crf_C,
                      tol               = options.crf_tol,
                      njobs             = options.crf_njobs,
                      max_iter          = options.max_iter,
                      inference_cache   = options.crf_inference_cache)
    
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
    usage, description, parser = DU_CRF_Task.getBasicTrnTstRunOptionParser(sys.argv[0], version)
#     parser.add_option("--annotate", dest='bAnnotate',  action="store_true",default=False,  help="Annotate the textlines with BIES labels")    

    #FOR GCN
    parser.add_option("--revertEdges", dest='bRevertEdges',  action="store_true", help="Revert the direction of the edges") 
    parser.add_option("--detail", dest='bDetailedReport',  action="store_true", default=False,help="Display detailled reporting (score per document)") 
    parser.add_option("--baseline", dest='bBaseline',  action="store_true", default=False, help="report baseline method") 
            
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
