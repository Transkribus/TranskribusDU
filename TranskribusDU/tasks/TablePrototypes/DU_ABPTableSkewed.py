# -*- coding: utf-8 -*-

"""
    DU task for ABP Table: 
        doing jointly row EIO and near horizontal cuts SIO
    
    block2line edges do not cross another block.
    
    The cut are based on baselines of text blocks, with some positive or negative inclination.

    - the labels of cuts are SIO 
    
    Copyright Naver Labs Europe(C) 2018 JL Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""




import sys, os
import math
from lxml import etree
from collections import Counter
from ast import literal_eval

import numpy as np
import shapely.geometry as geom
import shapely.ops

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from common.trace import traceln
from tasks import _checkFindColDir, _exit
from tasks.DU_CRF_Task import DU_CRF_Task
from tasks.DU_Table.DU_ABPTableSkewed_CutAnnotator import SkewedCutAnnotator

from xml_formats.PageXml import MultiPageXml, PageXml

import graph.GraphModel
from graph.Block import Block
from graph.Edge import Edge, SamePageEdge, HorizontalEdge, VerticalEdge
from graph.Graph_MultiPageXml import Graph_MultiPageXml
from graph.NodeType_PageXml import NodeType_PageXml_type

#from graph.FeatureDefinition_PageXml_std_noText import FeatureDefinition_PageXml_StandardOnes_noText
from graph.FeatureDefinition import FeatureDefinition
from graph.Transformer import Transformer, TransformerListByType, SparseToDense
from graph.Transformer import EmptySafe_QuantileTransformer as QuantileTransformer
from graph.Transformer_PageXml import NodeTransformerXYWH_v2, NodeTransformerNeighbors, Node1HotFeatures_noText,\
    NodeTransformerText, NodeTransformerTextLen, EdgeNumericalSelector_v2
from graph.Transformer_PageXml import Edge1HotFeatures_noText, EdgeBooleanFeatures_v2, EdgeNumericalSelector_noText
from graph.PageNumberSimpleSequenciality import PageNumberSimpleSequenciality

from util.Shape import ShapeLoader

class GraphSkewedCut(Graph_MultiPageXml):
    """
    We specialize the class of graph because the computation of edges is quite specific
    
    Here we consider horizontal and near-horizontal cuts
    """
    bCutAbove = False  # the cut line is above the "support" text
    lRadAngle = None
    
    #Cut stuff
    #iModulo          = 1  # map the coordinate to this modulo
    fMinPageCoverage = 0.5  # minimal coverage to consider a GT table separator
    # fCutHeight       = 25   # height of a cutting ribbon
    # For NAF to get 91% GT recall with same recall on ABP 98% (moving from 105 to 108% cuts)
    fCutHeight       = 10   # height of a cutting ribbon
    
    # BAAAAD iLineVisibility  = 5 * 11  # a cut line sees other cut line up to N pixels downward
    iLineVisibility  = 3700 // 7  # (528) a cut line sees other cut line up to N pixels downward
    iBlockVisibility = 3*7*13  # (273) a block sees neighbouring cut lines at N pixels
    
    _lClassicNodeType = None
    
    # when loading a text, we create a shapely shape using the function below.
    shaper_fun = ShapeLoader.node_to_Point
    
    @classmethod
    def setClassicNodeTypeList(cls, lNodeType):
        """
        determine which type of node goes thru the classical way for determining
        the edges (vertical or horizontal overlap, with occlusion, etc.) 
        """
        cls._lClassicNodeType = lNodeType
    
    def parseDocFile(self, sFilename, iVerbose=0):
        """
        Load that document as a CRF Graph.
        Also set the self.doc variable!
        
        CAUTION: DOES NOT WORK WITH MULTI-PAGE DOCUMENTS...
        
        Return a CRF Graph object
        """
        traceln(" ----- FILE %s ------" % sFilename)
        self.doc = etree.parse(sFilename)
        self.lNode, self.lEdge = list(), list()
        self.lNodeBlock     = []  # text node
        self.lNodeCutLine  = []  # cut line node
        
        doer = SkewedCutAnnotator(self.bCutAbove, lAngle=self.lRadAngle)
        domid = 0
        for (pnum, page, domNdPage) in self._iter_Page_DocNode(self.doc):
            traceln(" --- page %s - constructing separator candidates" % pnum)
            #load the page objects and the GT partition (defined by the table) if any
            loBaseline, dsetTableByRow = doer.loadPage(domNdPage, shaper_fun=self.shaper_fun)
            traceln(" - found %d objects on page" % (len(loBaseline)))
            if loBaseline: traceln("\t - shaped as %s" % type(loBaseline[0]))

            # find almost-horizontal cuts and tag them if GT is available
            loHCut = doer.findHCut(domNdPage, loBaseline, dsetTableByRow, self.fCutHeight, iVerbose)  

            #create DOM node reflecting the cuts 
            #first clean (just in case!)
            n = doer.remove_cuts_from_dom(domNdPage) 
            if n > 0: 
                traceln(" - removed %d pre-existing cut lines" % n)
        
            # if GT, then we have labelled cut lines in DOM
            domid = doer.add_Hcut_to_Page(domNdPage, loHCut, domid)

        lClassicType = [nt for nt in self.getNodeTypeList() if nt     in self._lClassicNodeType]
        lSpecialType = [nt for nt in self.getNodeTypeList() if nt not in self._lClassicNodeType]
        
        for (pnum, page, domNdPage) in self._iter_Page_DocNode(self.doc):
            traceln(" --- page %s - constructing the graph" % pnum)
            #now that we have the page, let's create the node for each type!
            lClassicPageNode = [nd for nodeType in lClassicType for nd in nodeType._iter_GraphNode(self.doc, domNdPage, page) ]
            lSpecialPageNode = [nd for nodeType in lSpecialType for nd in nodeType._iter_GraphNode(self.doc, domNdPage, page) ]

            self.lNode.extend(lClassicPageNode)  # e.g. the TextLine objects
            self.lNodeBlock.extend(lClassicPageNode)
            
            self.lNode.extend(lSpecialPageNode)  # e.g. the cut lines!
            self.lNodeCutLine.extend(lSpecialPageNode)
            
            #no previous page to consider (for cross-page links...) => None
            lClassicPageEdge = Edge.computeEdges(None, lClassicPageNode, self.iGraphMode)
            self.lEdge.extend(lClassicPageEdge)
            
            # Now, compute edges between special and classic objects...
            lSpecialPageEdge = self.computeSpecialEdges(lClassicPageNode,
                                                        lSpecialPageNode)
            self.lEdge.extend(lSpecialPageEdge)
            
            #if iVerbose>=2: traceln("\tPage %5d    %6d nodes    %7d edges"%(pnum, len(lPageNode), len(lPageEdge)))
            if iVerbose>=2:
                traceln("\tPage %5d"%(pnum))
                traceln("\t   block: %6d nodes    %7d edges (to block)" %(len(lClassicPageNode), len(lClassicPageEdge)))
                traceln("\t   line: %6d nodes    %7d edges (from block or line)"%(len(lSpecialPageNode), len(lSpecialPageEdge)))
                c = Counter(type(o).__name__ for o in lSpecialPageEdge)
                l = list(c.items())
                l.sort()
                traceln("\t\t", l)
        if iVerbose: traceln("\t\t (%d nodes,  %d edges)"%(len(self.lNode), len(self.lEdge)) )
        
        return self

    def addParsedLabelToDom(self):
        """
        while parsing the pages, we may have updated the standard BIESO labels
        we store the possibly new label in the DOM
        """ 
        for nd in self.lNode:
            nd.type.setDocNodeLabel(nd, self._dLabelByCls[ nd.cls ])
        
    def addEdgeToDoc(self):
        """
        To display the grpah conveniently we add new Edge elements
        """
        import random
        (pnum, page, ndPage) = next(self._iter_Page_DocNode(self.doc))
        w = int(ndPage.get("imageWidth"))

        nn = 1 + len([e for e in self.lEdge if type(e) not in [HorizontalEdge, VerticalEdge, Edge_BL]])
        ii = 0
        for edge in self.lEdge:
            if type(edge) in [HorizontalEdge, VerticalEdge]:
                A, B = edge.A.shape.centroid, edge.B.shape.centroid
            elif type(edge) in [Edge_BL]:
                A = edge.A.shape.centroid
                # not readable                 _pt, B = shapely.ops.nearest_points(A, edge.B.shape)
                _pt, B = shapely.ops.nearest_points(edge.A.shape, edge.B.shape)
            else:
                ii += 1
                x = 1 + ii * (w/nn)
                pt = geom.Point(x, 0)
                A, _ = shapely.ops.nearest_points(edge.A.shape, pt)
                B, _ = shapely.ops.nearest_points(edge.B.shape, pt)
            ndSep = MultiPageXml.createPageXmlNode("Edge")
            ndSep.set("DU_type", type(edge).__name__)
            ndPage.append(ndSep)
            MultiPageXml.setPoints(ndSep, [(A.x, A.y), (B.x, B.y)])
        return
            
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

class GraphSkewedCut_H(GraphSkewedCut):
    """
    Only horizontal cut lines
    """
    
    def __init__(self):
        self.showClassParam()
        
    @classmethod
    def showClassParam(cls):
        """
        show the class parameters
        return whether or not they were shown
        """
        try:
            cls.bParamShownOnce
            return False
        except:
            #traceln("  - iModulo : "            , cls.iModulo)
            traceln("  - block_see_line : "     , cls.iBlockVisibility)
            traceln("  - line_see_line  : "     , cls.iLineVisibility)
            traceln("  - cut height     : "     , cls.fCutHeight)
            traceln("  - cut above      : "     , cls.bCutAbove)
            traceln("  - angles         : "     , [math.degrees(v) for v in cls.lRadAngle])
            traceln("  - fMinPageCoverage : "   , cls.fMinPageCoverage)
            traceln("  - Textual features : "   , cls.bTxt)
            cls.bParamShownOnce = True
            return True
    
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
    def computeSpecialEdges(self, lClassicPageNode, lSpecialPageNode):
        """
        Compute:
        - edges between each block and the cut line above/across/below the block
        - edges between cut lines
        return a list of edges
        """
        #augment the block with the coordinate of its baseline central point
        for blk in lClassicPageNode:
            try:
                pt = blk.shape.centroid
                blk.x_bslne = pt.x
                blk.y_bslne = pt.y
            except IndexError:
                traceln("** WARNING: no Baseline in ", blk.domid)
                traceln("** Using BB instead... :-/")
                blk.x_bslne = (blk.x1+blk.x2) / 2
                blk.y_bslne = (blk.y1+blk.y2) / 2
            blk._in_edge_up = 0     # count of incoming edge from upper lines
            blk._in_edge_down = 0   # count of incoming edge from downward lines

        #block to cut line edges
        # no _type=0 because they are valid cut (never crossing any block)
        lEdge = []
        for cutBlk in lSpecialPageNode:
            #equation of the line
            # y = A x + B
            A = (cutBlk.y2 - cutBlk.y1) / (cutBlk.x2 - cutBlk.x1)
            B =  cutBlk.y1 - A * cutBlk.x1
            oCut = cutBlk.shape
            for blk in lClassicPageNode:
                dist =  oCut.distance(blk.shape)
                if dist <= self.iBlockVisibility:
                    edge = Edge_BL(blk, cutBlk)  # Block _to_ Cut !!
                    # experiments show that abs helps
                    # edge.len = (blk.y_bslne - cutBlk.y1) / self.iBlockVisibility
                    edge.len = dist / self.iBlockVisibility
                    y = A * blk.x_bslne + B     # y of the point on cut line
                    # edge._type = -1 if blk.y_bslne > y else (+1 if blk.y_bslne < y else 0)
                    # shapely can give as distance a very small number while y == 0
                    edge._type = -1 if blk.y_bslne >= y else +1
                    assert edge._type != 0, (str(oCut), list(blk.shape.coords), oCut.distance(blk.shape.centroid), str(blk.shape.centroid))
                    lEdge.append(edge)                    
        
        #now filter those edges
        n0 = len(lEdge)
        #lEdge = self._filterBadEdge(lEdge, lClassicPageNode, lSpecialPageNode)
        lEdge = self._filterBadEdge(lEdge, lSpecialPageNode)
        
        traceln(" - filtering: removed %d edges due to obstruction." % (n0-len(lEdge)))

        # add a counter of incoming edge to nodes, for features eng.
        for edge in lEdge:
            if edge._type > 0:
                edge.A._in_edge_up += 1
            else:
                edge.A._in_edge_down += 1
        
        # Cut line to Cut line edges
        n0 = len(lEdge)
        if self.iLineVisibility > 0:
            for i, A in enumerate(lSpecialPageNode):
                for B in lSpecialPageNode[i+1:]:
                    dist = A.shape.distance(B.shape)
                    if dist <= self.iLineVisibility:
                        edge = Edge_LL(A, B)
                        edge.len = dist / self.iLineVisibility
                        lEdge.append(edge)
            traceln(" - edge_LL: added %d edges." % (len(lEdge)-n0))
        
        return lEdge


    @classmethod
    def _filterBadEdge(cls, lEdge, lCutLine, fRatio=0.25):
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
        

        def isTargetLineVisible_X(edge, lEdge, fThreshold=0.9):
            """
            can the source node of the edge see the target node line?
            we say no if some other block obstructs half or more of the view
            """
            a1, a2 = edge.A.x1, edge.A.x2
            w = a2 - a1
            minVisibility = w * fThreshold
            for _edge in lEdge:
                # we want a visibility window of at least 1/4 of the object A
                b1, b2 = _edge.A.x1, _edge.A.x2
                vis = min(w, max(0, b1 - a1) + max(0, a2 - b2))
                if vis <= minVisibility: return False
            return True

        #there are two ways for dealing with lines crossed by a block
        # - either it prevents another block to link to the line (assuming an x-overlap)
        # - or not (historical way)
        # THIS IS THE "MODERN" way!!
        
        #take each line in turn
        for ndLine in lCutLine:
            #--- process downward edges
            #TODO: index!
            lDownwardAndXingEdge = [edge for edge in lEdge \
                              if edge._type > 0 and edge.B == ndLine]
            if lDownwardAndXingEdge:
                #sort edge by source block from closest to line block to farthest
                #lDownwardAndXingEdge.sort(key=lambda o: ndLine.y1 - o.A.y_bslne)
                lDownwardAndXingEdge.sort(key=lambda o: ndLine.shape.distance(o.A.shape))
                
                lKeepDownwardEdge = [lDownwardAndXingEdge.pop(0)]
                
                #now keep all edges whose source does not overlap vertically with 
                #  the source of an edge that is kept
                for edge in lDownwardAndXingEdge:
                    if isTargetLineVisible_X(edge, lKeepDownwardEdge):
                        lKeepDownwardEdge.append(edge)
                lKeepEdge.extend(lKeepDownwardEdge)

            #--- process upward edges
            #TODO: index!
            lUpwarAndXingdEdge = [edge for edge in lEdge \
                              if edge._type < 0 and edge.B == ndLine]
            if lUpwarAndXingdEdge:
                #sort edge by source block from closest to line -block to farthest
                #lUpwarAndXingdEdge.sort(key=lambda o: o.A.y_bslne - ndLine.y2)
                lUpwarAndXingdEdge.sort(key=lambda o: ndLine.shape.distance(o.A.shape))
                lKeepUpwardEdge = [lUpwarAndXingdEdge.pop(0)]
                
                #now keep all edges whose source does not overlap vertically with 
                #  the source of an edge that is kept
                for edge in lUpwarAndXingdEdge:
                    if isTargetLineVisible_X(edge, lKeepUpwardEdge):
                        lKeepUpwardEdge.append(edge)
                        
                # now we keep only the edges, excluding the crossing ones
                # (already included!!)
                lKeepEdge.extend(edge for edge in lKeepUpwardEdge)
                
            #--- and include the crossing ones (that are discarded
        return lKeepEdge


#------------------------------------------------------------------------------------------------------
class SupportBlock_NodeTransformer(Transformer):
    """
    aspects related to the "support" notion of a block versus a cut line  
    """
    def transform(self, lNode):
#         a = np.empty( ( len(lNode), 5 ) , dtype=np.float64)
#         for i, blk in enumerate(lNode): a[i, :] = [blk.x1, blk.y2, blk.x2-blk.x1, blk.y2-blk.y1, blk.fontsize]        #--- 2 3 4 5 6 
        a = np.empty( ( len(lNode), 2 ) , dtype=np.float64)
        for i, blk in enumerate(lNode): 
            a[i, :] = (blk._in_edge_up, blk._in_edge_down)
        return a

#------------------------------------------------------------------------------------------------------
class CutLine_NodeTransformer_v3(Transformer):
    """
    features of a Cut line:
    - horizontal or vertical.
    """
    def transform(self, lNode):
        #We allocate TWO more columns to store in it the tfidf and idf computed at document level.
        #a = np.zeros( ( len(lNode), 10 ) , dtype=np.float64)  # 4 possible orientations: 0, 1, 2, 3
        N = 6
        a = np.zeros( ( len(lNode), N ) , dtype=np.float64)  # 4 possible orientations: 0, 1, 2, 3
        
        for i, blk in enumerate(lNode):
            page = blk.page
            assert abs(blk.x2 - blk.x1) > abs(blk.y1 - blk.y2)
                #horizontal
            v = (blk.y1+blk.y2)/float(page.h) - 1  # to range -1, +1
            a[i,:] = (1.0, v, v*v
                          , blk.angle, blk.angle_freq, blk.angle_cumul_freq)
#             else:
#                 #vertical
#                 v = 2*blk.x1/float(page.w) - 1  # to range -1, +1
#                 a[i, N:] = (1.0, v, v*v
#                           ,blk.angle, blk.angle_freq, blk.angle_cumul_freq) 
        # traceln("CutLine_NodeTransformer_v3", a[:min(100, len(lNode)),])
        return a

class CutLine_NodeTransformer_qty(Transformer):
    """
    features of a Cut line:
    - horizontal or vertical.
    """
    def transform(self, lNode):
        #We allocate TWO more columns to store in it the tfidf and idf computed at document level.
        #a = np.zeros( ( len(lNode), 10 ) , dtype=np.float64)  # 4 possible orientations: 0, 1, 2, 3
        N = 1
        a = np.zeros( ( len(lNode), 2*N ) , dtype=np.float64)  # 4 possible orientations: 0, 1, 2, 3
        
        for i, blk in enumerate(lNode):
            assert abs(blk.x2 - blk.x1) > abs(blk.y1 - blk.y2)
            #horizontal
            a[i,:] = (len(blk.set_support))
        return a


#------------------------------------------------------------------------------------------------------
class Block2CutLine_EdgeTransformer(Transformer):
    """
    features of a block to Cut line edge:
    - below, crossing, above
    """
    def transform(self, lEdge):
        N = 8
        a = np.zeros( ( len(lEdge), 2 * N) , dtype=np.float64)  
        for i, edge in enumerate(lEdge):
            z = 0 if edge._type < 0 else N  # _type is -1 or 1 
            blk = edge.A
            page = blk.page
            w = float(page.w) #  h = float(page.h)
            x = (blk.x1 + blk.x2) / w - 1  # [-1, +1]
            a[i, z:z+N] = (1.0
                           , edge.len
                           , edge.len*edge.len
                           , edge.B.angle_freq
                           , edge.B.angle_cumul_freq
                           , 1.0 if edge.A.du_index in edge.B.set_support else 0.0
                           , x, x * x
                           )
#             print(a[i,:].tolist())
        # traceln("Block2CutLine_EdgeTransformer", a[:min(100, len(lEdge)),])
        return a

class Block2CutLine_EdgeTransformer_qtty(Transformer):
    def transform(self, lEdge):
        N = 3
        a = np.zeros( ( len(lEdge), 2 * N) , dtype=np.float64)  
        for i, edge in enumerate(lEdge):
            z = 0 if edge._type < 0 else N  # _type is -1 or 1 
            a[i, z:z+N] = (len(edge.B.set_support)
                           , edge.A._in_edge_up
                           , edge.A._in_edge_down
                           )
#             print(a[i,:].tolist())
        # traceln("Block2CutLine_EdgeTransformer", a[:min(100, len(lEdge)),])
        return a

class Block2CutLine_FakeEdgeTransformer(Transformer):
    """
    a fake transformer that return as many features as the union of real ones above
    """
    def transform(self, lEdge):
        assert not(lEdge)
        return np.zeros( ( len(lEdge), 2*8 + 2*3) , dtype=np.float64)


class CutLine2CutLine_EdgeTransformer(Transformer):  # ***** USELESS *****
    """
    features of a block to Cut line edge:
    - below, crossing, above
    """
# BEST SO FAR
#     def transform(self, lEdge):
#         a = np.zeros( ( len(lEdge), 4 ) , dtype=np.float64) 
#         for i, edge in enumerate(lEdge):
#             a[i,:] = (1, edge.len, edge.len * edge.len, int(edge.len==0))
#         # traceln("CutLine2CutLine_EdgeTransformer", a[:min(100, len(lEdge)),])
#         return a

# WORSE
#     def transform(self, lEdge):
#         a = np.zeros( ( len(lEdge),  12) , dtype=np.float64) 
#         for i, edge in enumerate(lEdge):
#             dAngle = (edge.A.angle - edge.B.angle) / 5  # we won't go beyond +-5 degrees.
#             iSameSupport = int(len(edge.B.set_support.intersection(edge.A.set_support)) > 0)
#             iCrosses = int(edge.A.shape.crosses(edge.B.shape))
#             a[i,:] = (1
#                       , edge.len, edge.len * edge.len, int(edge.len==0), int(edge.len < 5)
#                       , dAngle, dAngle * dAngle, int(abs(dAngle) < 0.1), int(abs(dAngle) < 0.1)
#                       , iSameSupport
#                       , iCrosses
#                       , (1-iSameSupport) * iCrosses # not same support but crossing each other
#                       )
#         return a
    
    def transform(self, lEdge):
        a = np.zeros( ( len(lEdge), 7 ) , dtype=np.float64) 
        for i, edge in enumerate(lEdge):
            dAngle = (edge.A.angle - edge.B.angle) / 5  # we won't go beyond +-5 degrees.
            iSameSupport = int(len(edge.B.set_support.intersection(edge.A.set_support)) > 0)
            iCrosses = int(edge.A.shape.crosses(edge.B.shape))
            a[i,:] = (1, edge.len, edge.len * edge.len
                      , dAngle, dAngle * dAngle
                    , iSameSupport
                    , iCrosses
                      )
        # traceln("CutLine2CutLine_EdgeTransformer", a[:min(100, len(lEdge)),])
        return a



class My_FeatureDefinition_v3_base(FeatureDefinition):
    n_QUANTILES = 16
    n_QUANTILES_sml = 8
    
    def __init__(self, **kwargs):
        """
        set _node_transformer, _edge_transformer, tdifNodeTextVectorizer
        """
        FeatureDefinition.__init__(self)
        self._node_transformer     = None
        self._edge_transformer     = None
        self._node_text_vectorizer = None #tdifNodeTextVectorizer

    def fitTranformers(self, lGraph,lY=None):
        """
        Fit the transformers using the graphs, but TYPE BY TYPE !!!
        return True
        """
        self._node_transformer[0].fit([nd for g in lGraph for nd in g.getNodeListByType(0)])
        self._node_transformer[1].fit([nd for g in lGraph for nd in g.getNodeListByType(1)])
        
        self._edge_transformer[0].fit([e for g in lGraph for e in g.getEdgeListByType(0, 0)])
        self._edge_transformer[1].fit([e for g in lGraph for e in g.getEdgeListByType(0, 1)])
        self._edge_transformer[2].fit([e for g in lGraph for e in g.getEdgeListByType(1, 0)])
        self._edge_transformer[3].fit([e for g in lGraph for e in g.getEdgeListByType(1, 1)])
        
        return True
    
class My_FeatureDefinition_v3(My_FeatureDefinition_v3_base):
    """
    Multitype version:
    so the node_transformer actually is a list of node_transformer of length n_class
       the edge_transformer actually is a list of node_transformer of length n_class^2
       
    We also inherit from FeatureDefinition_T !!!
    """ 
       
    def __init__(self, **kwargs):
        """
        set _node_transformer, _edge_transformer, tdifNodeTextVectorizer
        """
        My_FeatureDefinition_v3_base.__init__(self)

        nbTypes = self._getTypeNumber(kwargs)
        
        block_transformer = FeatureUnion( [  #CAREFUL IF YOU CHANGE THIS - see cleanTransformers method!!!!
                                    ("xywh", Pipeline([
                                                         ('selector', NodeTransformerXYWH_v2()),
                                                         #v1 ('xywh', StandardScaler(copy=False, with_mean=True, with_std=True))  #use in-place scaling
                                                         ('xywh', QuantileTransformer(n_quantiles=self.n_QUANTILES, copy=False))  #use in-place scaling
                                                         ])
                                       )
                                    , ("edge_cnt", Pipeline([
                                                         ('selector', SupportBlock_NodeTransformer()),
                                                         #v1 ('xywh', StandardScaler(copy=False, with_mean=True, with_std=True))  #use in-place scaling
                                                         ('edge_cnt', QuantileTransformer(n_quantiles=self.n_QUANTILES_sml, copy=False))  #use in-place scaling
                                                         ])
                                       )
                                    , ("neighbors", Pipeline([
                                                         ('selector', NodeTransformerNeighbors()),
                                                         #v1 ('neighbors', StandardScaler(copy=False, with_mean=True, with_std=True))  #use in-place scaling
                                                         ('neighbors', QuantileTransformer(n_quantiles=self.n_QUANTILES, copy=False))  #use in-place scaling
                                                         ])
                                       )
                                    , ("1hot", Pipeline([
                                                         ('1hot', Node1HotFeatures_noText())  #does the 1-hot encoding directly
                                                         ])
                                       )
                                      ])
        
        Cut_line_transformer = FeatureUnion( [
                                      ("std", CutLine_NodeTransformer_v3())
                                    , ("qty", Pipeline([
                                                         ('selector', CutLine_NodeTransformer_qty()),
                                                         ('quantile', QuantileTransformer(n_quantiles=self.n_QUANTILES_sml, copy=False))  #use in-place scaling
                                                         ])
                                        )
                                      ])
        
        self._node_transformer = TransformerListByType([block_transformer, Cut_line_transformer]) 
        
        edge_BB_transformer = FeatureUnion( [  #CAREFUL IF YOU CHANGE THIS - see cleanTransformers method!!!!
                                      ("1hot", Pipeline([
                                                         ('1hot', Edge1HotFeatures_noText(PageNumberSimpleSequenciality()))
                                                         ])
                                        )
                                    , ("boolean", Pipeline([
                                                         ('boolean', EdgeBooleanFeatures_v2())
                                                         ])
                                        )
                                    , ("numerical", Pipeline([
                                                         ('selector', EdgeNumericalSelector_noText()),
                                                         #v1 ('numerical', StandardScaler(copy=False, with_mean=True, with_std=True))  #use in-place scaling
                                                         ('numerical', QuantileTransformer(n_quantiles=self.n_QUANTILES, copy=False))  #use in-place scaling
                                                         ])
                                        )
                                          ] )
        #edge_BL_transformer = Block2CutLine_EdgeTransformer()
        edge_BL_transformer = FeatureUnion( [
                                      ("std", Block2CutLine_EdgeTransformer())
                                    , ("qty", Pipeline([
                                                         ('selector', Block2CutLine_EdgeTransformer_qtty()),
                                                         ('quantile', QuantileTransformer(n_quantiles=self.n_QUANTILES_sml, copy=False))  #use in-place scaling
                                                         ])
                                        )
                                      ])
        
        edge_LL_transformer = CutLine2CutLine_EdgeTransformer()
        self._edge_transformer = TransformerListByType([edge_BB_transformer,
                                                  edge_BL_transformer,
                                                  # edge_BL_transformer,  # useless but required
                                                  Block2CutLine_FakeEdgeTransformer(), # fit is called with [], so the Pipeline explodes
                                                  edge_LL_transformer 
                                                  ])
          


gTBL = str.maketrans("0123456789", "NNNNNNNNNN")
def My_FeatureDefinition_v3_txt_preprocess(s):
    """
    Normalization of the etxt before extracting ngrams
    """
    return s.lower().translate(gTBL)


class My_FeatureDefinition_v3_txt(My_FeatureDefinition_v3_base):
    """
    Multitype version:
    so the node_transformer actually is a list of node_transformer of length n_class
       the edge_transformer actually is a list of node_transformer of length n_class^2
       
    We also inherit from FeatureDefinition_T !!!
    """ 
    t_ngrams_range          = (2, 4)
    n_ngrams                = 1000
    
    # pre-processing of text before extracting ngrams
    def __init__(self, **kwargs):
        """
        set _node_transformer, _edge_transformer, tdifNodeTextVectorizer
        """
        My_FeatureDefinition_v3_base.__init__(self) 

        nbTypes = self._getTypeNumber(kwargs)

        # since we have a preprocessor, lowercase and strip_accents options are disabled
        self._node_text_vectorizer = CountVectorizer(  analyzer = 'char'
                                                     # AttributeError: Can't pickle local object 'My_FeatureDefinition_v3_txt.__init__.<locals>.<lambda>'
                                                     # , preprocessor = lambda x: x.lower().translate(self.TBL)
                                                     , preprocessor = My_FeatureDefinition_v3_txt_preprocess
                                                     , max_features = self.n_ngrams
                                                     , ngram_range  = self.t_ngrams_range #(2,6)
                                                     , dtype=np.float64)

        block_transformer = FeatureUnion( [  #CAREFUL IF YOU CHANGE THIS - see cleanTransformers method!!!!
                                      ("text", Pipeline([
                                                         ('selector', NodeTransformerText())
                                                       , ('vecto', self._node_text_vectorizer) #we can use it separately from the pipleline once fitted
                                                       , ('todense', SparseToDense())  #pystruct needs an array, not a sparse matrix
                                                       ])
                                     )
                                    , 
                                    ("textlen", Pipeline([
                                                         ('selector', NodeTransformerTextLen()),
                                                         ('textlen', QuantileTransformer(n_quantiles=self.n_QUANTILES, copy=False))  #use in-place scaling
                                                         ])
                                       )
                                    , ("xywh", Pipeline([
                                                         ('selector', NodeTransformerXYWH_v2()),
                                                         #v1 ('xywh', StandardScaler(copy=False, with_mean=True, with_std=True))  #use in-place scaling
                                                         ('xywh', QuantileTransformer(n_quantiles=self.n_QUANTILES, copy=False))  #use in-place scaling
                                                         ])
                                       )
                                    , ("edge_cnt", Pipeline([
                                                         ('selector', SupportBlock_NodeTransformer()),
                                                         #v1 ('xywh', StandardScaler(copy=False, with_mean=True, with_std=True))  #use in-place scaling
                                                         ('edge_cnt', QuantileTransformer(n_quantiles=self.n_QUANTILES_sml, copy=False))  #use in-place scaling
                                                         ])
                                       )
                                    , ("neighbors", Pipeline([
                                                         ('selector', NodeTransformerNeighbors()),
                                                         #v1 ('neighbors', StandardScaler(copy=False, with_mean=True, with_std=True))  #use in-place scaling
                                                         ('neighbors', QuantileTransformer(n_quantiles=self.n_QUANTILES, copy=False))  #use in-place scaling
                                                         ])
                                       )
                                    , ("1hot", Pipeline([
                                                         ('1hot', Node1HotFeatures_noText())  #does the 1-hot encoding directly
                                                         ])
                                       )
                                      ])
        
        Cut_line_transformer = FeatureUnion( [
                                      ("std", CutLine_NodeTransformer_v3())
                                    , ("qty", Pipeline([
                                                         ('selector', CutLine_NodeTransformer_qty()),
                                                         ('quantile', QuantileTransformer(n_quantiles=self.n_QUANTILES_sml, copy=False))  #use in-place scaling
                                                         ])
                                        )
                                      ])
        
        self._node_transformer = TransformerListByType([block_transformer, Cut_line_transformer]) 
        
        edge_BB_transformer = FeatureUnion( [  #CAREFUL IF YOU CHANGE THIS - see cleanTransformers method!!!!
                                      ("1hot", Pipeline([
                                                         ('1hot', Edge1HotFeatures_noText(PageNumberSimpleSequenciality()))
                                                         ])
                                        )
                                    , ("boolean", Pipeline([
                                                         ('boolean', EdgeBooleanFeatures_v2())
                                                         ])
                                        )
                                    , ("numerical", Pipeline([
                                                         ('selector', EdgeNumericalSelector_v2()),
                                                         #v1 ('numerical', StandardScaler(copy=False, with_mean=True, with_std=True))  #use in-place scaling
                                                         ('numerical', QuantileTransformer(n_quantiles=self.n_QUANTILES, copy=False))  #use in-place scaling
                                                         ])
                                        )
                                          ] )
        #edge_BL_transformer = Block2CutLine_EdgeTransformer()
        edge_BL_transformer = FeatureUnion( [
                                      ("std", Block2CutLine_EdgeTransformer())
                                    , ("qty", Pipeline([
                                                         ('selector', Block2CutLine_EdgeTransformer_qtty()),
                                                         ('quantile', QuantileTransformer(n_quantiles=self.n_QUANTILES_sml, copy=False))  #use in-place scaling
                                                         ])
                                        )
                                      ])
        
        edge_LL_transformer = CutLine2CutLine_EdgeTransformer()
        self._edge_transformer = TransformerListByType([edge_BB_transformer,
                                                  edge_BL_transformer,
                                                  # edge_BL_transformer,  # useless but required
                                                  Block2CutLine_FakeEdgeTransformer(), # fit is called with [], so the Pipeline explodes
                                                  edge_LL_transformer 
                                                  ])
          
        
    def cleanTransformers(self):
        """
        the TFIDF transformers are keeping the stop words => huge pickled file!!!
         
        Here the fix is a bit rough. There are better ways....
        JL
        """
        self._node_transformer[0].transformer_list[0][1].steps[1][1].stop_words_ = None   #is 1st in the union...
#         for i in [2, 3, 4, 5, 6, 7]:
#             self._edge_transformer.transformer_list[i][1].steps[1][1].stop_words_ = None   #are 3rd and 4th in the union....
        return self._node_transformer, self._edge_transformer        


def test_preprocess(capsys):
    
    with capsys.disabled():
        print("toto")
        tbl = str.maketrans("0123456789", "NNNNNNNNNN")
        fun = lambda x: x.lower().translate(tbl)
        assert "abc" == fun("abc")
        assert "abc" == fun("ABC")
        assert "abcdé" == fun("ABCdé")
        assert "tüv" == fun("tÜv")
        assert "tüv NN " == fun("tÜv 12 ")
        assert "" == fun("")
        assert "N" == fun("1")
        assert "NN" == fun("23")
        assert "j't'aime moi non plus. dites NN!!" == fun("J't'aime MOI NON PlUs. Dites 33!!")
        assert "" == fun("")
        assert "" == fun("")
        assert "" == fun("")
    

class NodeType_PageXml_Cut_Shape(NodeType_PageXml_type):
    """
    we specialize it because our cuts are near horizontal
    """
    def _iter_GraphNode(self, doc, domNdPage, page):
        """
        Get the DOM, the DOM page node, the page object

        iterator on the DOM, that returns nodes  (of class Block)
        """    
        #--- XPATH contexts
        assert self.sxpNode, "CONFIG ERROR: need an xpath expression to enumerate elements corresponding to graph nodes"
        lNdBlock = domNdPage.xpath(self.sxpNode, namespaces=self.dNS) #all relevant nodes of the page

        for ndBlock in lNdBlock:
            domid = ndBlock.get("id")
            sText = ""
            
            #now we need to infer the bounding box of that object
            (x1, y1), (x2, y2) = PageXml.getPointList(ndBlock)  #the polygon
            
            orientation = 0 
            classIndex = 0   #is computed later on

            #and create a Block
            # we pass the coordinates, not x1,y1,w,h !!
            cutBlk = Block(page, ((x1, y1), (x2, y2)), sText, orientation, classIndex, self, ndBlock, domid=domid)
            
            # Create the shapely shape
            cutBlk.shape = geom.LineString([(x1, y1), (x2, y2)])
            cutBlk.angle = float(ndBlock.get("DU_angle"))
            cutBlk.angle_freq       = float(ndBlock.get("DU_angle_freq"))
            cutBlk.angle_cumul_freq = float(ndBlock.get("DU_angle_cumul_freq"))
            cutBlk.set_support      = literal_eval(ndBlock.get("DU_set_support"))
            
            yield cutBlk
            
        return        


# ----------------------------------------------------------------------------

def main(TableSkewedRowCut_CLASS, sModelDir, sModelName, options):
    """
    TableSkewedRowCut_CLASS must be a class inheriting from DU_Graph_CRF
    """
    lDegAngle = [float(s) for s in options.lsAngle.split(",")]
    lRadAngle = [math.radians(v) for v in lDegAngle]
    
    doer = TableSkewedRowCut_CLASS(sModelName, sModelDir, 
                        iBlockVisibility  = options.iBlockVisibility,
                        iLineVisibility   = options.iLineVisibility,
                        fCutHeight        = options.fCutHeight,
                        bCutAbove         = options.bCutAbove,
                        lRadAngle         = lRadAngle,
                        bTxt              = options.bTxt,
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
        try:    traceln("Baseline best estimator: %s"%doer.bsln_mdl.best_params_)   #for CutSearch
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
    
       
def main_command_line(TableSkewedRowCut_CLASS):        
    version = "v.01"
    usage, description, parser = DU_CRF_Task.getBasicTrnTstRunOptionParser(sys.argv[0], version)
#     parser.add_option("--annotate", dest='bAnnotate',  action="store_true",default=False,  help="Annotate the textlines with BIES labels")    

    #FOR GCN
    parser.add_option("--revertEdges", dest='bRevertEdges',  action="store_true", help="Revert the direction of the edges") 
    parser.add_option("--detail", dest='bDetailedReport',  action="store_true", default=False,help="Display detailed reporting (score per document)") 
    parser.add_option("--baseline", dest='bBaseline',  action="store_true", default=False, help="report baseline method") 
    parser.add_option("--line_see_line", dest='iLineVisibility',  action="store",
                      type=int, default=GraphSkewedCut.iLineVisibility,
                      help="seeline2line: how far in pixel can a line see another cut line?") 
    parser.add_option("--block_see_line", dest='iBlockVisibility',  action="store",
                      type=int, default=GraphSkewedCut.iBlockVisibility,
                      help="seeblock2line: how far in pixel can a block see a cut line?") 
    parser.add_option("--height", dest="fCutHeight", default=GraphSkewedCut.fCutHeight
                      , action="store", type=float, help="Minimal height of a cut") 
    parser.add_option("--cut-above", dest='bCutAbove',  action="store_true", default=False
                        ,help="Each object defines one or several cuts above it (instead of below as by default)") 
    parser.add_option("--angle", dest='lsAngle'
                      ,  action="store", type="string", default="-1,0,+1"
                        ,help="Allowed cutting angles, in degree, comma-separated") 

    parser.add_option("--graph", dest='bGraph',  action="store_true", help="Store the graph in the XML for displaying it") 
            
    # --- 
    #parse the command line
    (options, args) = parser.parse_args()

    if options.bGraph:
        import os.path
        # hack
        TableSkewedRowCut_CLASS.bCutAbove = options.bCutAbove
        traceln("\t%s.bCutAbove=" % TableSkewedRowCut_CLASS.__name__, TableSkewedRowCut_CLASS.bCutAbove)
        TableSkewedRowCut_CLASS.lRadAngle = [math.radians(v) for v in [float(s) for s in options.lsAngle.split(",")]]
        traceln("\t%s.lRadAngle=" % TableSkewedRowCut_CLASS.__name__, TableSkewedRowCut_CLASS.lRadAngle)
        for sInputFilename in args:
            sp, sf = os.path.split(sInputFilename)
            sOutFilename = os.path.join(sp, "graph-" + sf)
            doer = TableSkewedRowCut_CLASS("debug", "."
                                           , iBlockVisibility=options.iBlockVisibility
                                           , iLineVisibility=options.iLineVisibility
                                           , fCutHeight=options.fCutHeight
                                           , bCutAbove=options.bCutAbove
                                           , lRadAngle=[math.radians(float(s)) for s in options.lsAngle.split(",")])
            o = doer.cGraphClass()
            o.parseDocFile(sInputFilename, 9)
            o.parseDocLabels()
            o.addParsedLabelToDom()
            o.addEdgaddEdgeToDoc         print('Graph edges added to %s'%sOutFilename)
            o.doc.write(sOutFilename, encoding='utf-8',pretty_print=True,xml_declaration=True)
        SkewedCutAnnotator.gtStatReport()
        exit(0)
    
    # --- 
    try:
        sModelDir, sModelName = args
    except Exception as e:
        traceln("Specify a model folder and a model name!")
        _exit(usage, 1, e)
    
    main(TableSkewedRowCut_CLASS, sModelDir, sModelName, options)
    
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    from tasks.DU_ABPTableSkewed_txtBIO_sepSIO import DU_ABPTableSkewedRowCut
    main_command_line(DU_ABPTableSkewedRowCut)
