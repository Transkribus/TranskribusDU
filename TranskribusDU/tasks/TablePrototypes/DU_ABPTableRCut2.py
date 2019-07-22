# -*- coding: utf-8 -*-

"""
    DU task for ABP Table: doing jointly row BIESO and horizontal cuts
    
    block2line edges do not cross another block.
    
    The cut are based on baselines of text blocks.
    
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




import sys, os
import math
from lxml import etree
import collections

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
import graph.GraphModel
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

from tasks.DU_ABPTableCutAnnotator import BaselineCutAnnotator 

class GraphCut2(Graph_MultiPageXml):
    """
    We specialize the class of graph because the computation of edges is quite specific
    """

    #Cut stuff
    #iModulo          = 1  # map the coordinate to this modulo
    fMinPageCoverage = 0.5  # minimal coverage to consider a GT table separator
    #iLineVisibility  = 2  # a cut line sees N neighbours below
    iBlockVisibility = 3*7*13  # a block sees neighbouring cut lines at N pixels
    
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
        self.lNodeCutLine  = []  # cut line node
        
        root = self.doc.getroot()
        
        doer = BaselineCutAnnotator()
        #doer.setModulo(self.iModulo)  # this is optional
        
        #load the groundtruth table separators, if any, per page (1 in tABP)
        ltlYlX = doer.get_separator_YX_from_DOM(root, self.fMinPageCoverage)
        for (lHi, lVi) in ltlYlX:
            traceln(" - found %d horizontal,  %d vertical  GT separators" % (len(lHi), len(lVi)))

        #create DOM node reflecting the cuts 
        #first clean (just in case!)
        n = doer.remove_cuts_from_dom(root) 
        if n > 0: 
            traceln(" - removed %d pre-existing cut lines" % n)
        
        # if GT, then we have labelled cut lines in DOM
        _ltlYCutXCut = doer.add_cut_to_DOM(root, ltlYlX=ltlYlX)  

        lClassicType = [nt for nt in self.getNodeTypeList() if nt     in self._lClassicNodeType]
        lSpecialType = [nt for nt in self.getNodeTypeList() if nt not in self._lClassicNodeType]
        
        for (pnum, page, domNdPage) in self._iter_Page_DomNode(self.doc):
            #now that we have the page, let's create the node for each type!
            lClassicPageNode = [nd for nodeType in lClassicType for nd in nodeType._iter_GraphNode(self.doc, domNdPage, page) ]
            lSpecialPageNode = [nd for nodeType in lSpecialType for nd in nodeType._iter_GraphNode(self.doc, domNdPage, page) ]

            self.lNode.extend(lClassicPageNode)  # e.g. the TextLine objects
            self.lNodeBlock.extend(lClassicPageNode)
            
            self.lNode.extend(lSpecialPageNode)  # e.g. the cut lines!
            self.lNodeCutLine.extend(lSpecialPageNode)
            
            #no previous page to consider (for cross-page links...) => None
            lClassicPageEdge = Edge.computeEdges(None, lClassicPageNode)
            self.lEdge.extend(lClassicPageEdge)
            
            # Now, compute edges between special and classic objects...
            lSpecialPageEdge = self.computeSpecialEdges(lClassicPageNode,
                                                        lSpecialPageNode,
                                                        doer.bCutIsBeforeText)
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
    def __init__(self, A, B, typ):
        """
        An edge from a block A to a cut line B
        typ is
        -1 = upward
        0 = crossing
        +1 = downward
        2 = crossing, with cut line exactly at A baseline
        """
        Edge.__init__(self, A, B)
        self._type      = typ 

class Edge_LL(Edge):
    """Edge line-to-Line"""
    pass

class GraphCut2_H(GraphCut2):
    """
    Only horizontal cut lines
    """
    
    def __init__(self):
        self.showClassParam()
        
    @classmethod
    def showClassParam(cls):
        try:
            cls.bParamShownOnce
            assert cls.bParamShownOnce == True
        except:
            #traceln("  - iModulo : "            , cls.iModulo)
            traceln("  - block_see_line : "     , cls.iBlockVisibility)
            #traceln("  - line_see_line  : "     , cls.iLineVisibility)
            traceln("  - fMinPageCoverage : "   , cls.fMinPageCoverage)
            cls.bParamShownOnce = True
    
    def getNodeListByType(self, iTyp):
        if iTyp == 0:
            return self.lNodeBlock
        else:
            return self.lNodeCutLine
        
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
    def computeSpecialEdges(self, lClassicPageNode, lSpecialPageNode,
                            bCutIsBeforeText):
        """
        Compute:
        - edges between each block and the cut line above/across/below the block
        - edges between cut lines
        return a list of edges
        """
       
        #augment the block with the coordinate of its baseline central point
        for blk in lClassicPageNode:
            try:
                x,y = BaselineCutAnnotator.getDomBaselineXY(blk.node)
                blk.x_bslne = x
                blk.y_bslne = y
            except IndexError:
                traceln("** WARNING: no Baseline in ", blk.domid)
                traceln("** Using x2 and y2 instead... :-/")
                blk.x_bslne = int(round(blk.x2))
                blk.y_bslne = int(round(blk.y2))
                
                
        
        for cutBlk in lSpecialPageNode:
            assert cutBlk.y1 == cutBlk.y2, "Update code: horizontal cut only for now"
            cutBlk.y1 = int(round(cutBlk.y1))  #DeltaFun make float
            cutBlk.y2 = cutBlk.y1

        #block to cut line edges
        lEdge = []
        for blk in lClassicPageNode:
            for cutBlk in lSpecialPageNode:
                if blk.y1 <= cutBlk.y1 and cutBlk.y1 <= blk.y2:
                    #crossing
                    typ = 2 if blk.y_bslne == cutBlk.y1 else 0
                    edge = Edge_BL(blk, cutBlk, typ)
                    edge.len = (blk.y_bslne - cutBlk.y1) / abs(blk.y1 - blk.y2)
                    lEdge.append(edge)                    
                elif abs(blk.y_bslne - cutBlk.y1) <= self.iBlockVisibility:
                    typ = -1 if blk.y_bslne > cutBlk.y1 else +1
                    edge = Edge_BL(blk, cutBlk, typ)
                    edge.len = abs(blk.y_bslne - cutBlk.y1) / self.iBlockVisibility
                    lEdge.append(edge)                    
        
        #sort those edge from top to bottom
        lEdge.sort(key=lambda o: o.B.y1)  # o.B.y1 == o.B.y2 by construction
        
        #now filter those edges
        n0 = len(lEdge)
        if False:
            print("--- before filtering: %d edges" % len(lEdge))
            lSortedEdge = sorted(lEdge, key=lambda x: x.A.domid)
            for edge in lSortedEdge:
                print("Block domid=%s y1=%s y2=%s yg=%s"%(edge.A.domid, edge.A.y1, edge.A.y2, edge.A.y_bslne)
                        + "  %s line %s "%(["↑", "x", "↓", "-"][1+edge._type],                                       
                                       edge.B.y1)
                        + "domid=%s y1=%s  " %(edge.B.domid, edge.B.y1)
                        +str(id(edge))
                    )
        lEdge = self._filterBadEdge(lEdge, lSpecialPageNode, bCutIsBeforeText)
        traceln(" - filtering: removed %d edges due to obstruction." % (n0-len(lEdge)))
        if False:
            print("--- After filtering: %d edges" % len(lEdge))
            lSortedEdge = sorted(lEdge, key=lambda x: x.A.domid)
            print(len(lSortedEdge))
            for edge in lSortedEdge:
                print("Block domid=%s y1=%s y2=%s yg=%s"%(edge.A.domid, edge.A.y1, edge.A.y2, edge.A.y_bslne)
                        + "  %s line %s "%(["↑", "x", "↓", "-"][1+edge._type],                                       
                                       edge.B.y1)
                        + "domid=%s y1=%s  " %(edge.B.domid, edge.B.y1)
                        +str(id(edge))
                    )

#         # grid line to grid line edges
#         n = len(dGridLineByIndex)
#         for i in range(n):
#             A = dGridLineByIndex[i]
#             for j in range(i+1, min(n, i+cls.iGridVisibility+1)):
#                 edge = Edge_LL(A, dGridLineByIndex[j])
#                 edge.len = (j - i)
#                 lEdge.append(edge) 
        
        return lEdge


    @classmethod
    def _filterBadEdge(cls, lEdge, lCutLine, bCutIsBeforeText, fRatio=0.25):
        """
        We get 
        - a list of block2Line edges
        - a sorted list of cut line 
        But some block should not be connected to a line due to obstruction by 
        another blocks.
        We filter out those edges...
        return a sub-list of lEdge
        """
        lKeepEdge = []
        
        def _xoverlapSrcSrc(edge, lEdge):
            """
            does the source node of edge overlap with the source node of any 
            edge of the list?
            """
            A = edge.A
            for _edge in lEdge:
                if A.significantXOverlap(_edge.A, fRatio): return True
            return False

        def _yoverlapSrcSrc(edge, lEdge):
            """
            does the source node of edge overlap with the source node of any 
            edge of the list?
            """
            A = edge.A
            for _edge in lEdge:
                if A.significantYOverlap(_edge.A, fRatio): return True
            return False
        
        #there are two ways for dealing with lines crossed by a block
        # - either it prevents another block to link to the line (assuming an x-overlap)
        # - or not (historical way)
        # THIS IS THE "MODERN" way!!
        
        #check carefully the inequality below...
        if bCutIsBeforeText == True:
            keep1 = set([1])
            keep2 = set([-1, 2])  # downward and exact
        else:
            keep1 = set([1, 2])
            keep2 = set([-1])
        
        #take each line in turn
        for ndLine in lCutLine:

            #--- process downward edges
            #TODO: index!
            lDownwardAndXingEdge = [edge for edge in lEdge \
                              if edge._type in keep1 and edge.B == ndLine]
            if lDownwardAndXingEdge:
                #sort edge by source block from closest to line block to farthest
                lDownwardAndXingEdge.sort(key=lambda o: ndLine.y1 - o.A.y_bslne)
                
                lKeepDownwardEdge = [lDownwardAndXingEdge.pop(0)]
                
                #now keep all edges whose source does not overlap vertically with 
                #  the source of an edge that is kept
                for edge in lDownwardAndXingEdge:
                    if not _xoverlapSrcSrc(edge, lKeepDownwardEdge):
                        lKeepDownwardEdge.append(edge)
                lKeepEdge.extend(lKeepDownwardEdge)

            #NOTHING to do for crossing edges: they should be in the list!
#             #--- keep all crossing edges
#             #TODO: index!
#             lCrossingEdge = [edge for edge in lEdge \
#                               if edge._type == 0 and edge.B == ndLine]
#             
#             lKeepEdge.extend(lCrossingEdge)                
            
            #--- process upward edges
            #TODO: index!
            lUpwarAndXingdEdge = [edge for edge in lEdge \
                              if edge._type in keep2 and edge.B == ndLine]
            if lUpwarAndXingdEdge:
                #sort edge by source block from closest to line -block to farthest
                lUpwarAndXingdEdge.sort(key=lambda o: o.A.y_bslne - ndLine.y2)
                
                lKeepUpwardEdge = [lUpwarAndXingdEdge.pop(0)]
                
                #now keep all edges whose source does not overlap vertically with 
                #  the source of an edge that is kept
                for edge in lUpwarAndXingdEdge:
                    if not _xoverlapSrcSrc(edge, lKeepUpwardEdge):
                        lKeepUpwardEdge.append(edge)
                # now we keep only the edges, excluding the crossing ones
                # (already included!!)
                lKeepEdge.extend(edge for edge in lKeepUpwardEdge)
                
        #--- and include the crossing ones (that are discarded
        lKeepEdge.extend(edge for edge in lEdge if edge._type == 0)
            
        return lKeepEdge


#------------------------------------------------------------------------------------------------------
class GridLine_NodeTransformer_v2(Transformer):
    """
    features of a grid line:
    - horizontal or vertical.
    """
    def transform(self, lNode):
        #We allocate TWO more columns to store in it the tfidf and idf computed at document level.
        #a = np.zeros( ( len(lNode), 10 ) , dtype=np.float64)  # 4 possible orientations: 0, 1, 2, 3
        a = np.zeros( ( len(lNode), 6 ) , dtype=np.float64)  # 4 possible orientations: 0, 1, 2, 3
        
        for i, blk in enumerate(lNode):
            page = blk.page
            if abs(blk.x2 - blk.x1) > abs(blk.y1 - blk.y2):
                #horizontal
                v = 2*blk.y1/float(page.h) - 1  # to range -1, +1
                a[i,0:3] = (1.0, v, v*v)
            else:
                #vertical
                v = 2*blk.x1/float(page.w) - 1  # to range -1, +1
                a[i,3:6] = (1.0, v, v*v)
        return a


class Block2CutLine_EdgeTransformer(Transformer):
    """
    features of a block to grid line edge:
    - below, crossing, above
    
    typ  is -1, 0, 1, 2
    
    """
    def transform(self, edge):
        a = np.zeros( ( len(edge), 3 + 3 + 3 + 3) , dtype=np.float64)  # 4 possible orientations: 0, 1, 2, 3
        
        for i, edge in enumerate(edge):
            z = 1 + edge._type   # _type is -1 or 0 or 1 or 2
            a[i, z] = 1.0 
            a[i, 4 + z] = edge.len   # normalised on [0, 1] edge length
            a[i, 8 + z] = edge.len * edge.len
             
        return a

class GridLine2GridLine_EdgeTransformer(Transformer):  # ***** USELESS *****
    """
    features of a block to grid line edge:
    - below, crossing, above
    """
    def transform(self, edge):
        a = np.zeros( ( len(edge), 1 ) , dtype=np.float64) 
        
        for i, edge in enumerate(edge):
            a[i, 0] = 1.0   # edge length (number of steps)
            
        return a

class My_FeatureDefinition_v2(FeatureDefinition):
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
        grid_line_transformer = GridLine_NodeTransformer_v2()
        
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
        edge_BL_transformer = Block2CutLine_EdgeTransformer()
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


class DU_ABPTableRCut(DU_CRF_Task):
    """
    We will do a CRF model for a DU task
    , with the below labels 
    """
    sXmlFilenamePattern = "*[0-9].mpxml"
    
    iBlockVisibility    = None
    
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
        lLabels_SO_Cut  = ['S', 'O']
       
        #DEFINING THE CLASS OF GRAPH WE USE
        DU_GRAPH = GraphCut2_H
        
        DU_GRAPH.iBlockVisibility   = cls.iBlockVisibility
        
        # ROW
        ntR = NodeType_PageXml_type_woText("row"
                              , lLabels_BIESO
                              , None
                              , False
                              , BBoxDeltaFun=lambda v: max(v * 0.066, min(5, v/3))
                              )
        ntR.setLabelAttribute("DU_row")
        ntR.setXpathExpr( (".//pc:TextLine"        #how to find the nodes
                          , "./pc:TextEquiv")       #how to get their text
                       )
        DU_GRAPH.addNodeType(ntR)
        
        # HEADER
        ntCutH = NodeType_PageXml_type_woText("ch"
                              , lLabels_SO_Cut
                              , None
                              , False
                              , None        # equiv. to: BBoxDeltaFun=lambda _: 0
                              )
        ntCutH.setLabelAttribute("type")
        ntCutH.setXpathExpr( ('.//pc:CutSeparator[@orient="0"]'        #how to find the nodes
                          , "./pc:TextEquiv")       #how to get their text
                       )
        DU_GRAPH.addNodeType(ntCutH)        
        
        DU_GRAPH.setClassicNodeTypeList( [ntR ])
        
        return DU_GRAPH
        
    def __init__(self, sModelName, sModelDir, 
                 iBlockVisibility = None,
                 sComment = None,
                 C=None, tol=None, njobs=None, max_iter=None,
                 inference_cache=None): 
        
        DU_ABPTableRCut.iBlockVisibility = iBlockVisibility
        
        DU_CRF_Task.__init__(self
                     , sModelName, sModelDir
                     , dFeatureConfig = {'row_row':{}, 'row_ch':{},
                                         'ch_row':{}, 'ch_ch':{},
                                         'ch':{}, 'row':{}}
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
    doer = DU_ABPTableRCut(sModelName, sModelDir, 
                        iBlockVisibility  = options.iBlockVisibility,
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
    parser.add_option("--detail", dest='bDetailedReport',  action="store_true", default=False,help="Display detailed reporting (score per document)") 
    parser.add_option("--baseline", dest='bBaseline',  action="store_true", default=False, help="report baseline method") 
    parser.add_option("--block_see_line", dest='iBlockVisibility',  action="store",
                      type=int, default=273,
                      help="seeblock2line: how far in pixel can a block see a cut line?") 
            
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
