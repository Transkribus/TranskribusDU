# -*- coding: utf-8 -*-

"""
    DU task for ABP Table: doing jointly row BIESO and horizontal grid lines
    
    block2line edges do not cross another block.
    
    Here we make consistent label when any N grid lines have no block in-between
     each other.
    In that case, those N grid lines must have consistent BISO labels:
    - if one is B, all become B
    - elif one is S, all become S
    - elif one is I, all become I
    - else: they should all be O already 
    
   
    Copyright Naver Labs Europe(C) 2018 JL Meunier


    
    
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
    iGridVisibility = 2   # a grid line sees N neighbours below
    iBlockVisibility = 1  # a block sees N neighbouring grid lines
    
    _lClassicNodeType = None
    
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
        for (lHi, lVi) in ltlHlV:
            traceln(" - found %d horizontal,  %d vertical  GT separators" % (len(lHi), len(lVi)))

        #create DOM node reflecting the grid 
        #first clean (just in case!)
        n = doer.remove_grid_from_dom(root) 
        if n > 0: traceln(" - removed %d existing grid lines" % n)

        # we add GridSeparator elements. Groundtruth ones have type="1"
        n = doer.add_grid_to_DOM(root, ltlHlV)
        traceln(" - added   %d grid lines  %s" % (n,
                                        (self.iGridStep_H, self.iGridStep_V)) )       

        lClassicType = [nt for nt in self.getNodeTypeList() if nt in self._lClassicNodeType]
        lSpecialType = [nt for nt in self.getNodeTypeList() if nt not in self._lClassicNodeType]
        for pnum, page, domNdPage in self._iter_Page_DocNode(self.doc):
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
    def __init__(self):
        traceln("  - iGridStep_H : ", self.iGridStep_H)
        traceln("  - iGridStep_V : ", self.iGridStep_V)
        traceln("  - iGridVisibility  : ", self.iGridVisibility)
        traceln("  - iBlockVisibility : ", self.iBlockVisibility)
        traceln("  - fMinPageCoverage : ", self.fMinPageCoverage)
        
    
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
        fLenNorm = float(cls.iGridStep_V * cls.iBlockVisibility)
        imin, imax = 100, -1
        assert lClassicPageNode, "ERROR: empty page!!??"
        
        for ndBlock in lClassicPageNode:
            ### print("---- ", ndBlock)
            # i1 = GridAnnotator.snapToGridIndex(nd.x1, cls.iGridStep_V)
            # i2 = GridAnnotator.snapToGridIndex(nd.x2, cls.iGridStep_V)
            i1 = int(math.floor(ndBlock.y1 / float(cls.iGridStep_V)))
            i2 = int(math.ceil (ndBlock.y2 / float(cls.iGridStep_V)))
            assert i2 >= i1

            yBlkAvg = (ndBlock.y1 + ndBlock.y2)/2.0
            
            #Also make visible the iBlockVisibility-1 previous grid lines, if any
            for i in range(max(0, i1 - cls.iBlockVisibility + 1), i1+1):
                edge = Edge_BL(ndBlock, dGridLineByIndex[i])
                edge.len = (yBlkAvg - i * cls.iGridStep_V) / fLenNorm
                edge._gridtype = -1
                lEdge.append(edge)
                imin = min(i, imin)
                ### print(ndBlock.y1, i, edge.len)
            
            for i in range(max(0, i1+1), max(0, i2)):
                ndLine = dGridLineByIndex[i]
                edge = Edge_BL(ndBlock, ndLine)
                edge.len = (yBlkAvg  - i * cls.iGridStep_V) / fLenNorm
                edge._gridtype = 0 # grid line is crossing the block
                assert ndBlock.y1 < i*cls.iGridStep_V
                assert i*cls.iGridStep_V < ndBlock.y2
                ### print(ndBlock.y1, ndBlock.y2, i, edge.len)
                lEdge.append(edge)
                imax = max(imax, i)
            
            for i in range(max(0, i2), i2 + cls.iBlockVisibility):
                try:
                    edge = Edge_BL(ndBlock, dGridLineByIndex[i])
                except KeyError:
                    break  #  out of the grid
                edge.len = (yBlkAvg - i * cls.iGridStep_V) / fLenNorm
                edge._gridtype = +1
                lEdge.append(edge)
                imax = max(imax, i)
                ### print(ndBlock.y2, i, edge.len)
                
        #now filter those edges
        n0 = len(lEdge)
        lEdge = cls._filterBadEdge(lEdge, imin, imax, dGridLineByIndex)
        print(" - filtering: removed %d edges due to obstruction." % (len(lEdge) - n0))
        if False:
            print("--- After filtering: %d edges" % len(lEdge))
            lSortedEdge = sorted(lEdge, key=lambda x: x.A.domid)
            for edge in lSortedEdge:
                print("Block domid=%s y1=%s y2=%s"%(edge.A.domid, edge.A.y1, edge.A.y2)
                        + "  %s line %s "%(["↑", "-", "↓"][1+edge._gridtype],                                       
                                       edge.B.y1 / cls.iGridStep_V)
                        + "domid=%s y1=%s" %(edge.B.domid, edge.B.y1)
                    )
        #what differ from previosu version
        cls._makeConsistentLabelForEmptyGridRow(lEdge, lClassicPageNode, dGridLineByIndex)
        
        # grid line to grid line edges
        n = len(dGridLineByIndex)
        for i in range(n):
            A = dGridLineByIndex[i]
            for j in range(i+1, min(n, i+cls.iGridVisibility+1)):
                edge = Edge_LL(A, dGridLineByIndex[j])
                edge.len = (j - i)
                lEdge.append(edge) 
        
        return lEdge


    @classmethod
    def _filterBadEdge(cls, lEdge, imin, imax, dGridLineByIndex, fRatio=0.25):
        """
        We get 
        - a list of block2Line edges
        - the [imin, imax] interval of involved grid line index
        - the dGridLineByIndex dictionary
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
        
        #take each line in turn
        for i in range(imin, imax+1):
            ndLine = dGridLineByIndex[i]

            #--- process downward edges
            #TODO: index!
            lDownwardAndXingEdge = [edge for edge in lEdge \
                              if edge._gridtype >= 0 and edge.B == ndLine]
            if lDownwardAndXingEdge:
                #sort edge by source block from closest to line block to farthest
                lDownwardAndXingEdge.sort(key=lambda o: o.A.y2 - ndLine.y1,
                                          reverse=True)
                
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
#                               if edge._gridtype == 0 and edge.B == ndLine]
#             
#             lKeepEdge.extend(lCrossingEdge)                
            
            #--- process upward edges
            #TODO: index!
            lUpwarAndXingdEdge = [edge for edge in lEdge \
                              if edge._gridtype <= 0 and edge.B == ndLine]
            if lUpwarAndXingdEdge:
                #sort edge by source block from closest to line block to farthest
                lUpwarAndXingdEdge.sort(key=lambda o: ndLine.y2 - o.A.y1,
                                        reverse=True)
                
                lKeepUpwardEdge = [lUpwarAndXingdEdge.pop(0)]
                
                #now keep all edges whose source does not overlap vertically with 
                #  the source of an edge that is kept
                for edge in lUpwarAndXingdEdge:
                    if not _xoverlapSrcSrc(edge, lKeepUpwardEdge):
                        lKeepUpwardEdge.append(edge)
                # now we keep only the edges, excluding the crossing ones
                # (already included!!)
                lKeepEdge.extend(edge for edge in lKeepUpwardEdge \
                                  if edge._gridtype != 0)
                
        return lKeepEdge


    @classmethod
    def _makeConsistentLabelForEmptyGridRow(cls, lEdge, lBlockNode, dGridLineByIndex):
        """
        Here we make consistent label when any N grid lines have no block in-between
         each other.
        In that case, those N grid lines must have consistent BISO labels:
        - if one is B, all become B
        - elif one is S, all become S
        - elif one is I, all become I
        - else: they should all be O already  (or not annotated!)

        lLabels_BISO_Grid  = ['B', 'I', 'S', 'O']
        
        NOTE: I'm favoring safe and clean code to efficient code, for experimenting.
        TODO: optimize!  (if it performs better...)
        
        """
        bDBG = False
        
        #list object in each interval between 2 edges
        dsetObjectsByInterval = collections.defaultdict(set)
        imax = -1
        for ndBlock in lBlockNode:
            ### print("---- ", ndBlock)
            # i1 = GridAnnotator.snapToGridIndex(nd.x1, cls.iGridStep_V)
            # i2 = GridAnnotator.snapToGridIndex(nd.x2, cls.iGridStep_V)
            i1 = int(math.floor(ndBlock.y1 / float(cls.iGridStep_V)))
            i2 = int(math.ceil (ndBlock.y2 / float(cls.iGridStep_V)))
            for i in range(i1, i2):
                dsetObjectsByInterval[i].add(ndBlock)
            imax = max(imax, i2)
        
        # actually the imax is the index of the last positive grid line ('B')
        j = imax
        lj = list(dGridLineByIndex.keys())
        lj.sort(reverse=True)
        for j in lj:
            if dGridLineByIndex[j].node.get('type') == 'B':
                imax = max(imax, j)
                break
            
        
        
        #enumerate empty intervals
        lEmptyIntervalIndex = [i for i in range(0, imax+1) \
                          if bool(dsetObjectsByInterval[i]) == False]
        if bDBG: 
            traceln("nb empty intervals: %d"%len(lEmptyIntervalIndex))
            traceln([(j, dGridLineByIndex[j].domid, dGridLineByIndex[j].node.get('type')) for j in lEmptyIntervalIndex])
        
        #Make consistent labelling (if any labelling!!)
        if lEmptyIntervalIndex:
            k = 0                       #index in lEmptyInterval list
            kmax = len(lEmptyIntervalIndex)
            while k < kmax:
                i = lEmptyIntervalIndex[k]
                dk = 1
                while (k + dk) < kmax and lEmptyIntervalIndex[k+dk] == (i + dk): 
                    dk += 1
                if bDBG:
                    nd = dGridLineByIndex[i]
                    traceln("--- start grid line %s %s (nb=%d ending at %s)   cls=%s" %(nd.domid, i, dk-1,dGridLineByIndex[i+dk-1].domid,  nd.cls))
                
#TO FIX!!!!
#                 #we have a series of consecutive empty interval between i and i+dk (excluded)
#                 lCls = [dGridLineByIndex[j].cls for j in range(i, min(i+dk+1, kmax))]
#                 # we go to i+dk+1 because last boundary line may propagate its label
#                 #the node labels are loaded later on... :-(((
#                 
#                 if   0 in lCls:     # B
#                     iUniformClass = 0 
#                 elif 2 in lCls:     # S
#                     iUniformClass = 2
#                 elif 1 in lCls:     # I
#                     iUniformClass = 1
#                 elif 3 in lCls:     # O
#                     iUniformClass = 3
#                 else:               #unannotated
#                     if bDBG: traceln("No annotation: ", lCls)
#                     iUniformClass = None
#                 
#                 if not iUniformClass is None:
#                     for j in range(i, i+dk):
#                         if bDBG:
#                             nd = dGridLineByIndex[j]
#                             traceln("grid line %s %s made %d from %s"%(nd.domid, j, iUniformClass, nd.cls))
#                        dGridLineByIndex[j].cls = iUniformClass

#WORKAROUND  
                lCls = [dGridLineByIndex[j].node.get('type') for j in range(i, min(i+dk+1, imax+1))]
                # we go to i+dk+1 because last boundary line may propagate its label
                if   'B' in lCls:     # B
                    cUniformClass = 'B' 
                elif 'S' in lCls:     # S
                    cUniformClass = 'S'
                elif 'I' in lCls:     # I
                    cUniformClass = 'I'
                elif 'O' in lCls:     # O
                    cUniformClass = 'O'
                else:               #unannotated
                    if bDBG: traceln("No annotation: ", lCls)
                    cUniformClass = None
                
                if not cUniformClass is None:
                    for j in range(i, i+dk):
                        if bDBG:
                            nd = dGridLineByIndex[j]
                            traceln("grid line %s %s made %s from %s"%(nd.domid, j, cUniformClass, nd.node.get('type')))
                        dGridLineByIndex[j].node.set('type', cUniformClass)
                
                k = k + dk
                                
        return
    
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
        
        print("BETTER FEATURES")
        
        
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


class DU_ABPTableRG4(DU_CRF_Task):
    """
    We will do a CRF model for a DU task
    , with the below labels 
    """
    sXmlFilenamePattern = "*.mpxml"
    
    iGridStep_H         = None
    iGridStep_V         = None
    iGridVisibility     = None
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
        lLabels_BISO_Grid  = ['B', 'I', 'S', 'O']
       
        #DEFINING THE CLASS OF GRAPH WE USE
        DU_GRAPH = GraphGrid_H
        
        DU_GRAPH.iGridStep_H        = cls.iGridStep_H
        DU_GRAPH.iGridStep_V        = cls.iGridStep_V
        DU_GRAPH.iGridVisibility    = cls.iGridVisibility
        DU_GRAPH.iBlockVisibility   = cls.iBlockVisibility
        
        # ROW
        ntR = NodeType_PageXml_type_woText("row"
                              , lLabels_BIESO
                              , None
                              , False
                              
                              #HISTORICAL FUNCTION IS (idiotic I think...):
                              #, BBoxDeltaFun=lambda v: max(v * 0.066, min(5, v/3))
                              
                              , BBoxDeltaFun=lambda v: v / 5.0,  #keep 2/3rd of the box  
                              # we reduce overlap in this way
                              #this function returns the amount by which each border of
                              # a bounding box is "shifted toward its centre"...
                              #     w,h = x2-x1, y2-y1
                              #     dx = self.BBoxDeltaFun(w)
                              #     dy = self.BBoxDeltaFun(h)
                              #     x1,y1, x2,y2 = [ int(round(v)) for v in [x1+dx,y1+dy, x2-dx,y2-dy] ]

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
                 iGridStep_H = None,
                 iGridStep_V = None,
                 iGridVisibility = None,
                 iBlockVisibility = None,
                 sComment=None,
                 C=None, tol=None, njobs=None, max_iter=None,
                 inference_cache=None): 
        DU_ABPTableRG4.iGridStep_H = iGridStep_H
        DU_ABPTableRG4.iGridStep_V = iGridStep_V
        DU_ABPTableRG4.iGridVisibility  = iGridVisibility
        DU_ABPTableRG4.iBlockVisibility = iBlockVisibility
        
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
    doer = DU_ABPTableRG4(sModelName, sModelDir, 
                        iGridStep_H       = options.iGridStep_H,
                        iGridStep_V       = options.iGridStep_V,
                        iGridVisibility   = options.iGridVisibility,
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
    usage, description, parser = DU_CRF_Task.getBasicTrnTstRunOptionParser(sys.argv[0], version)
#     parser.add_option("--annotate", dest='bAnnotate',  action="store_true",default=False,  help="Annotate the textlines with BIES labels")    

    #FOR GCN
    parser.add_option("--revertEdges", dest='bRevertEdges',  action="store_true", help="Revert the direction of the edges") 
    parser.add_option("--detail", dest='bDetailedReport',  action="store_true", default=False,help="Display detailled reporting (score per document)") 
    parser.add_option("--baseline", dest='bBaseline',  action="store_true", default=False, help="report baseline method") 
    parser.add_option("--line_see_line", dest='iGridVisibility',  action="store",
                      type=int, default=2,
                      help="seeline2line: how many next grid lines does one line see?") 
    parser.add_option("--block_see_line", dest='iBlockVisibility',  action="store",
                      type=int, default=2,
                      help="seeblock2line: how many next grid lines does one block see?") 
    parser.add_option("--grid_h", dest='iGridStep_H',  action="store", type=int,
                      default=GraphGrid.iGridStep_H,
                      help="Grid horizontal step") 
    parser.add_option("--grid_v", dest='iGridStep_V',  action="store", type=int,
                      default=GraphGrid.iGridStep_V,
                      help="Grid Vertical step") 

            
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