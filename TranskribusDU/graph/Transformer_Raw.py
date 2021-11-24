# -*- coding: utf-8 -*-

"""
    Node and edge feature transformers to extract features from a graph
    
    No link with DOm or JSON => named GENERIC
    
  
    So, it should work equally well whatever the input format is (XML, JSON) since
    it uses only the node and edge geometric attributes and text
    
    Copyright Naver(C) 2019 JL. Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""

import collections


import numpy as np

from common.trace import traceln

from .FeatureDType  import dtype as Feat_dtype
from .Transformer   import Transformer
from .Edge          import HorizontalEdge, VerticalEdge, SamePageEdge
import graph.Transformer_PageXml as Transformer_PageXml

fALIGNMENT_COEF = 6.0

#------------------------------------------------------------------------------------------------------
# HERE IS WHAT IS UNCHANGED, because ALWAYS COMPUTABEL FROM ANY GRAPH

NodeTransformerText             = Transformer_PageXml.NodeTransformerText 
NodeTransformerTextEnclosed     = Transformer_PageXml.NodeTransformerTextEnclosed 
NodeTransformerNeighborText     = Transformer_PageXml.NodeTransformerNeighborText 
NodeTransformerTextLen          = Transformer_PageXml.NodeTransformerTextLen 
Node1ConstantFeature            = Transformer_PageXml.Node1ConstantFeature 

EdgeTransformerByClassIndex     = Transformer_PageXml.EdgeTransformerByClassIndex 
EdgeTransformerSourceText       = Transformer_PageXml.EdgeTransformerSourceText 
EdgeTransformerTargetText       = Transformer_PageXml.EdgeTransformerTargetText 
EdgeTransformerClassShifter     = Transformer_PageXml.EdgeTransformerClassShifter 


# -----------------------------------------------------------------------------
class NodeTransformerTextSentencePiece(Transformer):
    """
    Node text
    """
    bTextNotShownYet = True

    def __init__(self, sSPModel):
        self.sSPModel = sSPModel
    
    def _delayed_loadSPM(self):
        """
        a loaded SPM model cannot be serialized. 
        So we create it after the transformer is saved... :-/
        """
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.Load(self.sSPModel)  # e.g. "test/test_model.model"
        traceln(" SentencePiece model vocab size = %d" % len(sp))
        self.sp = sp
        self.N  = len(sp)

    def transform(self, lNode):
        # SentencePiece model
        try:
            self.sp
        except:
            self._delayed_loadSPM()

        if self.bTextNotShownYet:
            traceln(" Example of text and SentencePiece (#vocab=%d) subwords: " % len(self.sp)
                    , "\n".join(map(lambda o: str((o.text, self.sp.EncodeAsPieces(o.text))), lNode[0:10])))
            self.bTextNotShownYet = False

            
        nn = len(lNode)
        a = np.zeros((nn, self.N), dtype=Feat_dtype)

        # TODO make it faster
        sp = self.sp
        if True:
            # 1-hot encoding
            for i, n in enumerate(lNode):
                a[i, sp.EncodeAsIds(n.text)] = 1
        else:
            # frequencies
            for i, n in enumerate(lNode):
                a[i, sp.EncodeAsIds(n.text)] += 1
            a = a / a.sum(axis=1).reshape(len(lNode),1)
        return a
    
    def __str__(self):
        try:
            self.sp
        except:
            self._delayed_loadSPM()
        return "- Text_SPM %s <<%s  #vocab=%d>>" % (self.__class__, self.sSPModel, self.N)


#------------------------------------------------------------------------------------------------------
class NodeTransformerXYWH(Transformer):
    """
    In this version:
    -  X / pageW
    - Y / pageH
    """
    def transform(self, lNode):
        a = np.empty( ( len(lNode), 4) , dtype=Feat_dtype)

        page=lNode[0].page

        for i, blk in enumerate(lNode):
            x1,y1,x2,y2 = blk.x1, blk.y1, blk.x2, blk.y2
#             w = abs(x1-x2) 
#             h = abs(y1-y2) 
            x1,x2 = x1/page.w, x2/page.w
            y1,y2 = y1/page.h, y2/page.h
            a[i, :] = [  x1
                       , x2 
                       , y1 
                       , y2 
                       ]
        return a

#------------------------------------------------------------------------------------------------------
class NodeTransformerNeighbors(Transformer):
    """
    Characterising the neighborough
    """
    def transform(self, lNode):
#         a = np.empty( ( len(lNode), 5 ) , dtype=Feat_dtype)
#         for i, blk in enumerate(lNode): a[i, :] = [blk.x1, blk.y2, blk.x2-blk.x1, blk.y2-blk.y1, blk.fontsize]        #--- 2 3 4 5 6 
        a = np.empty( ( len(lNode), 2 + 2 ) , dtype=Feat_dtype)
        for i, blk in enumerate(lNode): 
            ax1, ay1 = blk.x1, blk.y1
            #number of horizontal/vertical/crosspage neighbors
            a[i,0:2] = len(blk.lHNeighbor), len(blk.lVNeighbor)
            #number of horizontal/vertical/crosspage neighbors occuring after this block
            a[i,2:4] = (sum(1 for _b in blk.lHNeighbor  if _b.x1 > ax1), 
                        sum(1 for _b in blk.lVNeighbor  if _b.y1 > ay1))
        return a

#------------------------------------------------------------------------------------------------------
class Node1HotTextFeatures(Transformer):
    """
    we will get a list of block and return a one-hot encoding, directly
    """
    def transform(self, lNode):
        a = np.zeros( ( len(lNode), 6 ) , dtype=Feat_dtype)
        for i, blk in enumerate(lNode): 
            s = blk.text
            a[i,0:7] = ( s.isalnum(),
                         s.isalpha(),
                         s.isdigit(),
                         s.islower(),
                         s.istitle(),
                         s.isupper())
        return a


#------------------------------------------------------------------------------------------------------
def _getMeanLengthByEdgeClass(lEdge, fMinMean=1e-8):
    """
    return a dictionary: edge_class --> mean_length
    
    with fMinMean as minimal mean value
    
    NOTE: this code looks at edge class, not its ancestor class, e.g. SamePageEgde and the like
    """
    dSum = collections.defaultdict(float)
    dCnt = collections.defaultdict(int)
    for e in lEdge:
        dSum[e.__class__] += e.length
        dCnt[e.__class__] += 1
    dMean = { c:max(fMinMean, dSum[c] / dCnt[c]) for c in dSum.keys() }
    del dSum, dCnt
    return dMean

def _getMaxLengthByEdgeClass(lEdge):
    """
    return a dictionary: edge_class --> mean_length
    
    with fMinMean as minimal mean value
    
    NOTE: this code looks at edge class, not its ancestor class, e.g. SamePageEgde and the like
    """
    dMax = collections.defaultdict(float)
    for e in lEdge:
        dMax[e.__class__] = max(e.length, dMax[e.__class__])
    return dMax


class EdgeNumericalSelector(EdgeTransformerClassShifter):
    """
    getting rid of the hand-crafted thresholds
    JLM Nov 2019: simpler and better (normalization must not change with direction for the 2 removed any direction features)
    JLM July 2020: adding x1,y1,x2,y2 of each node
    # Bug found in July 2020 by JLM: last feature was not used...
    """
    nbFEAT = 6-1+8 -1  #lcs
    
    def transform(self, lEdge):
        #no font size a = np.zeros( ( len(lEdge), 5 ) , dtype=Feat_dtype)
#         a = np.zeros( ( len(lEdge), 7 ) , dtype=Feat_dtype)
        a = np.zeros( ( len(lEdge), self._nbEdgeFeat ) , dtype=Feat_dtype)

        dMeanLength = _getMeanLengthByEdgeClass(lEdge)
         
        for i, edge in enumerate(lEdge):
            z = self._dEdgeClassIndexShift[edge.__class__]
            A,B = edge.A, edge.B        
            
            #overlap
            ovr = A.significantOverlap(B, 0)
            try:
                a[i, z+0] = ovr / (A.area() + B.area() - ovr)
            except ZeroDivisionError:
                pass
            
#             na, nb = len(A.text), len(B.text)
#             lcs = lcs_length(A.text,na, B.text,nb)
#             try:
#                 a[i, z+1] =  float( lcs / (na+nb-lcs) )
#             except ZeroDivisionError:
#                 pass
            
            norm_length = edge.length / dMeanLength[edge.__class__]
            
            a[i,z+2:z+5] = (edge.length, norm_length, norm_length*norm_length)
            a[i,z+5:z+5+8] = ( A.x1,A.x2, A.y1,A.y2
                             , B.x1,B.x2, B.y1,B.y2 )                    
        return a  


class EdgeNumericalSelector_noText(EdgeTransformerClassShifter):
    nbFEAT = 5-1+8
    
    def transform(self, lEdge):
        a = np.zeros( ( len(lEdge), self._nbEdgeFeat ) , dtype=Feat_dtype)

        dMeanLength = _getMeanLengthByEdgeClass(lEdge)

        for i, edge in enumerate(lEdge):
            z = self._dEdgeClassIndexShift[edge.__class__]
            A,B = edge.A, edge.B        
            
            #overlap
            ovr = A.significantOverlap(B, 0)
            try:
                a[i, z+0] = ovr / (A.area() + B.area() - ovr)
            except ZeroDivisionError:
                pass
            norm_length = edge.length / dMeanLength[edge.__class__]
            a[i,z+1:z+4] = (edge.length, norm_length, norm_length*norm_length)
            a[i,z+4:z+4+8] = ( A.x1,A.x2, A.y1,A.y2
                             , B.x1,B.x2, B.y1,B.y2 )                    
        return a  



class EdgeNumericalSelector_noText_vPeri(EdgeTransformerClassShifter):
    nbFEAT = 10
    
    def transform(self, lEdge):
        a = np.zeros( ( len(lEdge), self._nbEdgeFeat ) , dtype=Feat_dtype)
        
        for i, edge in enumerate(lEdge):
            z = self._dEdgeClassIndexShift[edge.__class__]
            A,B = edge.A, edge.B        

            # Blank zone characteristics
            # in term of length
            rL = edge.length / edge.getExtremeLength()
            
            # in term of areas
            _x1, _y1, w, h = edge.computeOverlapBB() 
            
            wh  = w*h
            rAreaToUnion   = wh / (wh + A.area() + B.area())
        
            rAreaToUnionBB = wh / (max(A.x2, B.x2) - min(A.x1, B.x1)) / (max(A.y2, B.y2) - min(A.y1, B.y1))
            
            # position of center of overlap
            _m, p1A, p1B, pmA, pmB, p2A, p2B, rO = edge.computeOverlapPositionAndRatio()
            
            a[i,z:z+10] = (  rO, rL                    # 1D
                          , rAreaToUnion,  rAreaToUnionBB   # 2D
                          , p1A, p1B, pmA, pmB, p2A, p2B                          # positional
                          )    
#             assert (abs(a[i]) <= 1.0).all(), a[i]
#         print(a[0:10,:])       
#         print(a[-10:,:])   
#         assert (a <= 1.0).all()    
#         assert (a >= -1.0).all()    
        return a  





