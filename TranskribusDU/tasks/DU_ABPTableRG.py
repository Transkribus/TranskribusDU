# -*- coding: utf-8 -*-

"""
    DU task for ABP Table: doing jointly row BIESO and horizontal grid lines
    
    Copyright Naver Labs Europe(C) 2018 JL Meunier

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
from __future__ import absolute_import
from __future__ import  print_function
from __future__ import unicode_literals

import sys, os
import math
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
from tasks.DU_CRF_Task import DU_CRF_Task
from crf.Edge import Edge, SamePageEdge
from crf.Graph_MultiPageXml import Graph_MultiPageXml
from crf.NodeType_PageXml   import NodeType_PageXml_type_woText

#from crf.FeatureDefinition_PageXml_std_noText import FeatureDefinition_PageXml_StandardOnes_noText
from crf.FeatureDefinition import FeatureDefinition
from crf.Transformer import Transformer, TransformerListByType
from crf.Transformer import EmptySafe_QuantileTransformer as QuantileTransformer
from crf.Transformer_PageXml import NodeTransformerXYWH_v2, NodeTransformerNeighbors, Node1HotFeatures
from crf.Transformer_PageXml import Edge1HotFeatures, EdgeBooleanFeatures_v2, EdgeNumericalSelector
from crf.PageNumberSimpleSequenciality import PageNumberSimpleSequenciality

from tasks.DU_ABPTableGrid import GridAnnotator 

class GraphGrid(Graph_MultiPageXml):
    """
    We specialize the class of graph because the computation of edges is quite specific
    """

    # Grid stuff
    #Dynamically add a grid
    iGridStep_H = 33  #odd number is better
    iGridStep_V = 33  #odd number is better
    # Some grid line will be O or I simply because they are too short.
    fMinPageCoverage = 0.5  # minimum proportion of the page crossed by a grid line
                            # we want to ignore col- and row- spans
    iGridVisibility = 2  # a grid line sees N neighbours below

    _lClassicNodeType = None
    @classmethod
    def setClassicNodeTypeList(cls, lNodeType):
        """
        determine which type of node goes thru the classical way for determining
        the edges (vertical or horizontal overlap, with occlusion, etc.) 
        """
        cls._lClassicNodeType = lNodeType
    
    def parseXmlFile(self, sFilename, iVerbose=0):
        """
        Load that document as a CRF Graph.
        Also set the self.doc variable!
        
        Return a CRF Graph object
        """
        self.doc = etree.parse(sFilename)
        self.lNode, self.lEdge = list(), list()
        self.lNodeBlock     = []  # text node
        self.lNodeGridLine  = []  # grid line node
        
        root = self.doc.getroot()
        
        doer = GridAnnotator(self.iGridStep_H, self.iGridStep_V)
        
        #map the groundtruth table separators, if any, to our grid
        ltlHlV = doer.get_grid_GT_index_from_DOM(root, self.fMinPageCoverage)
        
        #create DOM node reflecting the grid 
        # we add GridSeparator elements. Groundtruth ones have type="1"
        #doer.add_grid_to_DOM(root, ltlHlV)        

        lClassicType = [nt for nt in self.getNodeTypeList() if nt in self._lClassicNodeType]
        lSpecialType = [nt for nt in self.getNodeTypeList() if nt not in self._lClassicNodeType]
        for pnum, page, domNdPage in self._iter_Page_DomNode(self.doc):
            #now that we have the page, let's create the node for each type!
            lClassicPageNode = [nd for nodeType in lClassicType for nd in nodeType._iter_GraphNode(self.doc, domNdPage, page) ]
            lSpecialPageNode = [nd for nodeType in lSpecialType for nd in nodeType._iter_GraphNode(self.doc, domNdPage, page) ]

            self.lNode.extend(lClassicPageNode)  # e.g. the TextLine objects
            self.lNodeBlock.extend(lClassicPageNode)
            
            self.lNode.extend(lSpecialPageNode)  # e.g. the grid lines!
            self.lNodeGridLine.extend(lSpecialPageNode)
            
            #no previous page to consider (for cross-page links...) => None
            lClassicPageEdge = Edge.computeEdges(None, lClassicPageNode)
            self.lEdge.extend(lClassicPageEdge)
            
            # Now, compute edges between special and classic objects...
            lSpecialPageEdge = self.computeSpecialEdges(lClassicPageNode,
                                                        lSpecialPageNode)
            self.lEdge.extend(lSpecialPageEdge)
            
            #if iVerbose>=2: traceln("\tPage %5d    %6d nodes    %7d edges"%(pnum, len(lPageNode), len(lPageEdge)))
            if iVerbose>=2:
                traceln("\tPage %5d"%(pnum))
                traceln("\t   block: %6d nodes    %7d edges (to block)" %(pnum, len(lClassicPageNode), len(lClassicPageEdge)))
                traceln("\t   line: %6d nodes    %7d edges (from block)"%(pnum, len(lSpecialPageNode), len(lSpecialPageEdge)))
            
        if iVerbose: traceln("\t\t (%d nodes,  %d edges)"%(len(self.lNode), len(self.lEdge)) )
        
        return self
 
    @classmethod
    def computeSpecialEdges(cls, lClassicPageNode, lSpecialPageNode):
        """
        return a list of edges
        """
        raise Exception("Specialize this method")


class Edge_BL(Edge):
    """Edge block-to-Line"""
    pass

class Edge_LL(Edge):
    """Edge line-to-Line"""
    pass

class GraphGrid_H(GraphGrid):
    """
    Only horizontal grid lines
    """
    
    def getNodeListByType(self, iTyp):
        if iTyp == 0:
            return self.lNodeBlock
        else:
            return self.lNodeGridLine
        
    def getEdgeListByType(self, typA, typB):
        if typA == 0:
            if typB == 0:
                return (e for e in self.lEdge if isinstance(e, SamePageEdge))
            else:
                return (e for e in self.lEdge if isinstance(e, Edge_BL))
        else:
            if typB == 0:
                return []
            else:
                return (e for e in self.lEdge if isinstance(e, Edge_LL))
        

    @classmethod
    def computeSpecialEdges(cls, lClassicPageNode, lSpecialPageNode):
        """
        Compute:
        - edges between each block and the grid line above/across/below the block
        - edges between grid lines
        return a list of edges
        """
        # indexing the grid lines
        dGridLineByIndex = {GridAnnotator.snapToGridIndex(nd.y1, cls.iGridStep_V):nd for nd in lSpecialPageNode}
        
        for nd in lSpecialPageNode:
            #print(nd, dGridLineByIndex[GridAnnotator.snapToGridIndex(nd.y1, cls.iGridStep_V)])
            assert dGridLineByIndex[GridAnnotator.snapToGridIndex(nd.y1, cls.iGridStep_V)] == nd, "internal error inconsistent grid"
        
        # block to grid line edges
        lEdge = []
        fLenNorm = float(cls.iGridStep_V * cls.iGridVisibility)
        for ndBlock in lClassicPageNode:
            ### print("---- ", ndBlock)
            # i1 = GridAnnotator.snapToGridIndex(nd.x1, cls.iGridStep_V)
            # i2 = GridAnnotator.snapToGridIndex(nd.x2, cls.iGridStep_V)
            i1 = int(math.floor(ndBlock.y1 / float(cls.iGridStep_V)))
            i2 = int(math.ceil (ndBlock.y2 / float(cls.iGridStep_V)))
            assert i2 >= i1

            yavg = (ndBlock.y1 + ndBlock.y2)/2.0
            
            #Also make visible the iGridVisibility-1 previous grid lines, if any
            for i in range(max(0, i1 - cls.iGridVisibility + 1), i1+1):
                edge = Edge_BL(ndBlock, dGridLineByIndex[i])
                edge.len = (yavg - i * cls.iGridStep_V) / fLenNorm
                edge._gridtype = -1
                lEdge.append(edge)
                ### print(ndBlock.y1, i, edge.len)
            
            for i in range(max(0, i1+1), max(0, i2)):
                ndLine = dGridLineByIndex[i]
                edge = Edge_BL(ndBlock, ndLine)
                edge.len = (yavg  - i * cls.iGridStep_V) / fLenNorm
                edge._gridtype = 0 # grid line is crossing the block
                assert ndBlock.y1 < i*cls.iGridStep_V
                assert i*cls.iGridStep_V < ndBlock.y2
                ### print(ndBlock.y1, ndBlock.y2, i, edge.len)
                lEdge.append(edge)
            
            for i in range(max(0, i2), i2 + cls.iGridVisibility):
                try:
                    edge = Edge_BL(ndBlock, dGridLineByIndex[i])
                except KeyError:
                    break  #  out of the grid
                edge.len = (yavg - i * cls.iGridStep_V) / fLenNorm
                edge._gridtype = +1
                lEdge.append(edge)
                ### print(ndBlock.y2, i, edge.len)
                
        # grid line to grid line edges
        n = len(dGridLineByIndex)
        for i in range(n):
            A = dGridLineByIndex[i]
            for j in range(i+1, min(n, i+cls.iGridVisibility+1)):
                B = dGridLineByIndex[j]
                edge = Edge_LL(A, B)
                edge.len = (j - i)
                lEdge.append(edge) 
        
        return lEdge



#------------------------------------------------------------------------------------------------------
class GridLine_NodeTransformer(Transformer):
    """
    features of a grid line:
    - horizontal or vertical.
    """
    def transform(self, lNode):
        #We allocate TWO more columns to store in it the tfidf and idf computed at document level.
        #a = np.zeros( ( len(lNode), 10 ) , dtype=np.float64)  # 4 possible orientations: 0, 1, 2, 3
        a = np.zeros( ( len(lNode), 2 ) , dtype=np.float64)  # 4 possible orientations: 0, 1, 2, 3
        
        for i, blk in enumerate(lNode):
            if abs(blk.x2 - blk.x1) > abs(blk.y1 - blk.y2):
                a[i,0] = 1.0
            else:
                a[i,1] = 1.0
            
        return a

class Block2GridLine_EdgeTransformer(Transformer):
    """
    features of a block to grid line edge:
    - below, crossing, above
    """
    def transform(self, edge):
        a = np.zeros( ( len(edge), 3 + 3 + 3) , dtype=np.float64)  # 4 possible orientations: 0, 1, 2, 3
        
        for i, edge in enumerate(edge):
            z = 1 + edge._gridtype   # _gridtype is -1 or 0 or 1
            a[i, z] = 1.0 
            a[i, 3 + z] = edge.len   # normalised on [0, 1] edge length
            a[i, 6 + z] = edge.len * edge.len
             
        return a

class GridLine2GridLine_EdgeTransformer(Transformer):
    """
    features of a block to grid line edge:
    - below, crossing, above
    """
    def transform(self, edge):
        a = np.zeros( ( len(edge), GraphGrid_H.iGridVisibility ) , dtype=np.float64)  # 4 possible orientations: 0, 1, 2, 3
        
        for i, edge in enumerate(edge):
            a[i, edge.len - 1] = 1.0   # edge length (number of steps)
            
        return a

class My_FeatureDefinition(FeatureDefinition):
    """
    Multitype version:
    so the node_transformer actually is a list of node_transformer of length n_class
       the edge_transformer actually is a list of node_transformer of length n_class^2
       
    We also inherit from FeatureDefinition_T !!!
    """ 
    n_QUANTILES = 16
       
    def __init__(self, **kwargs):
        """
        set _node_transformer, _edge_transformer, tdifNodeTextVectorizer
        """
        FeatureDefinition.__init__(self)

        nbTypes = self._getTypeNumber(kwargs)
        
        block_transformer = FeatureUnion( [  #CAREFUL IF YOU CHANGE THIS - see cleanTransformers method!!!!
                                    ("xywh", Pipeline([
                                                         ('selector', NodeTransformerXYWH_v2()),
                                                         #v1 ('xywh', StandardScaler(copy=False, with_mean=True, with_std=True))  #use in-place scaling
                                                         ('xywh', QuantileTransformer(n_quantiles=self.n_QUANTILES, copy=False))  #use in-place scaling
                                                         ])
                                       )
                                    , ("neighbors", Pipeline([
                                                         ('selector', NodeTransformerNeighbors()),
                                                         #v1 ('neighbors', StandardScaler(copy=False, with_mean=True, with_std=True))  #use in-place scaling
                                                         ('neighbors', QuantileTransformer(n_quantiles=self.n_QUANTILES, copy=False))  #use in-place scaling
                                                         ])
                                       )
                                    , ("1hot", Pipeline([
                                                         ('1hot', Node1HotFeatures())  #does the 1-hot encoding directly
                                                         ])
                                       )
                                      ])
        grid_line_transformer = GridLine_NodeTransformer()
        
        self._node_transformer = TransformerListByType([block_transformer, grid_line_transformer]) 
        
        edge_BB_transformer = FeatureUnion( [  #CAREFUL IF YOU CHANGE THIS - see cleanTransformers method!!!!
                                      ("1hot", Pipeline([
                                                         ('1hot', Edge1HotFeatures(PageNumberSimpleSequenciality()))
                                                         ])
                                        )
                                    , ("boolean", Pipeline([
                                                         ('boolean', EdgeBooleanFeatures_v2())
                                                         ])
                                        )
                                    , ("numerical", Pipeline([
                                                         ('selector', EdgeNumericalSelector()),
                                                         #v1 ('numerical', StandardScaler(copy=False, with_mean=True, with_std=True))  #use in-place scaling
                                                         ('numerical', QuantileTransformer(n_quantiles=self.n_QUANTILES, copy=False))  #use in-place scaling
                                                         ])
                                        )
                                          ] )
        edge_BL_transformer = Block2GridLine_EdgeTransformer()
        edge_LL_transformer = GridLine2GridLine_EdgeTransformer()
        self._edge_transformer = TransformerListByType([edge_BB_transformer,
                                                  edge_BL_transformer,
                                                  edge_BL_transformer,  # useless but required
                                                  edge_LL_transformer 
                                                  ])
          
        self.tfidfNodeTextVectorizer = None #tdifNodeTextVectorizer

    def fitTranformers(self, lGraph,lY=None):
        """
        Fit the transformers using the graphs, but TYPE BY TYPE !!!
        return True
        """
        self._node_transformer[0].fit([nd for g in lGraph for nd in g.getNodeListByType(0)])
        self._node_transformer[1].fit([nd for g in lGraph for nd in g.getNodeListByType(1)])
        
        self._edge_transformer[0].fit([e for g in lGraph for e in g.getEdgeListByType(0, 0)])
        self._edge_transformer[1].fit([e for g in lGraph for e in g.getEdgeListByType(0, 1)])
        #self._edge_transformer[2].fit([e for g in lGraph for e in g.getEdgeListByType(1, 0)])
        #self._edge_transformer[3].fit([e for g in lGraph for e in g.getEdgeListByType(1, 1)])
        
        return True


class DU_ABPTableRG(DU_CRF_Task):
    """
    We will do a CRF model for a DU task
    , with the below labels 
    """
    sXmlFilenamePattern = "*.mpxml"
    iGridVisibility = None
    
    #=== CONFIGURATION ====================================================================
    @classmethod
    def getConfiguredGraphClass(cls):
        """
        In this class method, we must return a configured graph class
        """
        
        # Textline labels
        #  Begin Inside End Single Other
        lLabels_BIESO  = ['B', 'I', 'E', 'S', 'O'] 

        # Grid lines: 
        #  Border Ignore Separator Outside
        lLabels_BISO_Grid  = ['B', 'I', 'S', 'O']
       
        #DEFINING THE CLASS OF GRAPH WE USE
        DU_GRAPH = GraphGrid_H
        
        if cls.iGridVisibility is None:
            traceln(" - grid visibility is %d" % DU_GRAPH.iGridVisibility)
        else:
            traceln(" - set grid visibility to %d" % cls.iGridVisibility)
            DU_GRAPH.iGridVisibility = cls.iGridVisibility
        
        # ROW
        ntR = NodeType_PageXml_type_woText("row"
                              , lLabels_BIESO
                              , None
                              , False
                              , BBoxDeltaFun=lambda v: max(v * 0.066, min(5, v/3))  #we reduce overlap in this way
                              )
        ntR.setLabelAttribute("DU_row")
        ntR.setXpathExpr( (".//pc:TextLine"        #how to find the nodes
                          , "./pc:TextEquiv")       #how to get their text
                       )
        DU_GRAPH.addNodeType(ntR)
        
        # HEADER
        ntGH = NodeType_PageXml_type_woText("gh"
                              , lLabels_BISO_Grid
                              , None
                              , False
                              , None        # equiv. to: BBoxDeltaFun=lambda _: 0
                              )
        ntGH.setLabelAttribute("type")
        ntGH.setXpathExpr( ('.//pc:GridSeparator[@orient="0"]'        #how to find the nodes
                          , "./pc:TextEquiv")       #how to get their text
                       )
        DU_GRAPH.addNodeType(ntGH)        
        
        DU_GRAPH.setClassicNodeTypeList( [ntR ])
        
        return DU_GRAPH
        
    def __init__(self, sModelName, sModelDir, 
                 iGridVisibility = None,
                 sComment=None,
                 C=None, tol=None, njobs=None, max_iter=None,
                 inference_cache=None): 

        DU_ABPTableRG.iGridVisibility = iGridVisibility
        
        DU_CRF_Task.__init__(self
                     , sModelName, sModelDir
                     , dFeatureConfig = {'row_row':{}, 'row_gh':{},
                                         'gh_row':{}, 'gh_gh':{},
                                         'gh':{}, 'row':{}}
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
                     ,cFeatureDefinition=My_FeatureDefinition
                     )
        
        
#         if options.bBaseline:
#             self.bsln_mdl = self.addBaseline_LogisticRegression()    #use a LR model trained by GridSearch as baseline
    #=== END OF CONFIGURATION =============================================================

  
#     def predict(self, lsColDir):
#         """
#         Return the list of produced files
#         """
#         self.sXmlFilenamePattern = "*.mpxml"
#         return DU_CRF_Task.predict(self, lsColDir)
#         
#     def runForExternalMLMethod(self, lsColDir, storeX, applyY, bRevertEdges=False):
#         """
#         Return the list of produced files
#         """
#         self.sXmlFilenamePattern = "*.mpxml"
#         return DU_CRF_Task.runForExternalMLMethod(self, lsColDir, storeX, applyY, bRevertEdges)
              

# ----------------------------------------------------------------------------

def main(sModelDir, sModelName, options):
    doer = DU_ABPTableRG(sModelName, sModelDir, 
                         iGridVisibility   = options.iGridVisibility,
                         C                 = options.crf_C,
                         tol               = options.crf_tol,
                         njobs             = options.crf_njobs,
                         max_iter          = options.crf_max_iter,
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
            splitter, ts_trn, lFilename_trn = doer._nfold_Init(lFold, options.iFoldInitNum, test_size=0.25, random_state=None, bStoreOnDisk=True)
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
        import crf.Model
        sReportPickleFilename = os.path.join(sModelDir, sModelName + "__report.txt")
        traceln("Results are in %s"%sReportPickleFilename)
        crf.Model.Model.gzip_cPickle_dump(sReportPickleFilename, loTstRpt)
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
            crf.Model.Model.gzip_cPickle_dump(sReportPickleFilename, tstReport)
    
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
    parser.add_option("--seeline", dest='iGridVisibility',  action="store", type=int, default=2, help="seeline: how many next grid lines does one line see?") 

            
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