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


# # -----------------------------------------------------------------------------
# class NodeTransformerTextBERT(Transformer):
#     """
#     Node text
#     """
#     bTextNotShownYet = True
# 
#     def __init__(self, sEmbModel='bert-base-multilingual-cased'):
#         self.sEmbModel = sEmbModel
#         # cpu only
#         import flair, torch
#         flair.device = torch.device('cpu')
#          
#         from flair.embeddings import BertEmbeddings
# #         from flair.embeddings import DocumentRNNEmbeddings
# #         from flair.embeddings import BytePairEmbeddings
# 
#         self.emb = BertEmbeddings(self.sEmbModel)
# #         self.document_embeddings = DocumentRNNEmbeddings([bert_embedding])
# #         self.document_embeddings = BytePairEmbeddings('multi',5000,128) #, cache_dir=cache_dir)
#         self.N = self.emb.embedding_length #self.document_embeddings.embedding_length
#     
#     def transform(self, lNode):
#             
#         nn = len(lNode)
#         a = np.zeros((nn, self.N), dtype=Feat_dtype)
#         
#         from flair.data import Sentence
#         import torch
#         for i, n in enumerate(lNode):
#             if len(n.text)>0: #flair is expecting a non empty list of tokens!
#                 s = Sentence(n.text)
#                 self.emb.embed(s)
#                 a[i] = torch.cat([t.get_embedding() for t in s]).reshape(self.N,len(s)).mean(1).cpu().detach().numpy()
# #                 self.document_embeddings.embed(s)
# #                 a[i] = s.get_embedding().cpu().detach().numpy()
#                 del(s) 
#         return a
#     
#     def __str__(self):
#         return "- BERT %s  (%s embedding length=%d)" % (self.__class__,self.emb.__class__, self.N)




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
    - we do not use the page width and height to normalise, but max(x2) and max(y2)
    - width and heights also normalised by mean(w) and mean(h)
    - we do not consider odd/even pages
    
    Added by Nitin
    Updated by JL
    we will get a list of block and need to send back what StandardScaler needs for in-place scaling, a numpy array!.
    So we return a numpy array
    """
    def transform(self, lNode):
#         a = np.empty( ( len(lNode), 5 ) , dtype=Feat_dtype)
#         for i, blk in enumerate(lNode): a[i, :] = [blk.x1, blk.y2, blk.x2-blk.x1, blk.y2-blk.y1, blk.fontsize]        #--- 2 3 4 5 6
        a = np.empty( ( len(lNode), 4+4+4 ) , dtype=Feat_dtype)

        try:
            max_x  = max(o.x2 for o in lNode)
            max_y  = max(o.y2 for o in lNode)
            mean_w = sum(abs(o.x1 - o.x2) for o in lNode) / len(lNode)
            mean_h = sum(abs(o.y1 - o.y2) for o in lNode) / len(lNode)
        except (ValueError, ZeroDivisionError):
            max_x, max_y, mean_w, mean_h = None, None, None, None

        for i, blk in enumerate(lNode):
            x1,y1,x2,y2 = blk.x1, blk.y1, blk.x2, blk.y2
            w = abs(x1-x2) / mean_w
            h = abs(y1-y2) / mean_h
            x1,x2 = x1/max_x, x2/max_x
            y1,y2 = y1/max_y, y2/max_y
            a[i, :] = [  x1, x1*x1
                       , x2, x2*x2
                       , y1, y1*y1
                       , y2, y2*y2
                       ,  w, w * w
                       ,  h, h * h
                       ]
        return a


class NodeTransformerXYWH_vPetitPiton(Transformer):

    nbFEAT = 4+4+4+3  # 15
    
    def transform(self, lNode):
        a = np.empty( ( len(lNode), NodeTransformerXYWH_vPetitPiton.nbFEAT) , dtype=Feat_dtype)

        try:
            min_x, max_x  = min(o.x1 for o in lNode), max(o.x2 for o in lNode)
            min_y, max_y  = min(o.y1 for o in lNode), max(o.y2 for o in lNode)
        except (ValueError, ZeroDivisionError):
            min_x, max_x, min_y, max_y = None, None, None, None

        max_w, max_h = abs(max_x - min_x), abs(max_y - min_y)
        for i, blk in enumerate(lNode):
            x1,y1,x2,y2 = blk.x1, blk.y1, blk.x2, blk.y2
            w = abs(x1-x2) / max_w
            h = abs(y1-y2) / max_h
            x1 = (x1 - min_x) / max_w
            x2 = (x2 - min_x) / max_w
            y1 = (y1 - min_y) / max_h
            y2 = (y2 - min_y) / max_h

            feat = [  x1, x1*x1
                       , x2, x2*x2
                       , y1, y1*y1
                       , y2, y2*y2
                       ,  w, w * w
                       ,  h, h * h
                       , x1*y1      , x2*y2     , w * h
                       ]
            a[i, :] = feat
            # HACK for EdgeNumericalSelector_noText_vPetitPiton
            blk._hack_feat = feat

        return a

class NodeTransformerXYWH_vPeri(Transformer):
    """
    Same as NodeTransformerXYWH_vPetitPiton without the ._hack_feat
    """
    nbFEAT = 4+4+4+3  # 15
    
    def transform(self, lNode):
        a = np.empty( ( len(lNode), NodeTransformerXYWH_vPetitPiton.nbFEAT) , dtype=Feat_dtype)

        try:
            min_x, max_x  = min(o.x1 for o in lNode), max(o.x2 for o in lNode)
            min_y, max_y  = min(o.y1 for o in lNode), max(o.y2 for o in lNode)
        except (ValueError, ZeroDivisionError):
            min_x, max_x, min_y, max_y = None, None, None, None

        max_w, max_h = abs(max_x - min_x), abs(max_y - min_y)
        for i, blk in enumerate(lNode):
            x1,y1,x2,y2 = blk.x1, blk.y1, blk.x2, blk.y2
            w = abs(x1-x2) / max_w
            h = abs(y1-y2) / max_h
            x1 = (x1 - min_x) / max_w
            x2 = (x2 - min_x) / max_w
            y1 = (y1 - min_y) / max_h
            y2 = (y2 - min_y) / max_h

            feat = [  x1, x1*x1
                       , x2, x2*x2
                       , y1, y1*y1
                       , y2, y2*y2
                       ,  w, w * w
                       ,  h, h * h
                       , x1*y1      , x2*y2     , w * h
                       ]
            a[i, :] = feat

        return a


class NodeTransformerXW(Transformer):

    def transform(self, lNode):
#         a = np.empty( ( len(lNode), 5 ) , dtype=Feat_dtype)
#         for i, blk in enumerate(lNode): a[i, :] = [blk.x1, blk.y2, blk.x2-blk.x1, blk.y2-blk.y1, blk.fontsize]        #--- 2 3 4 5 6
        a = np.empty( ( len(lNode), 6 ) , dtype=Feat_dtype)

        try:
            max_x  = max(o.x2 for o in lNode)
            mean_w = sum(abs(o.x1 - o.x2) for o in lNode) / len(lNode)
        except (ValueError, ZeroDivisionError):
            max_x, mean_w = None, None

        for i, blk in enumerate(lNode):
            x1,x2 = blk.x1, blk.x2
            w = abs(x1-x2) / mean_w
            x1,x2 = x1/max_x, x2/max_x
            a[i, :] = [  x1, x1*x1
                       , x2, x2*x2
                       ,  w, w * w
                       ]
        return a


class NodeTransformerYH(Transformer):

    def transform(self, lNode):
#         a = np.empty( ( len(lNode), 5 ) , dtype=Feat_dtype)
#         for i, blk in enumerate(lNode): a[i, :] = [blk.x1, blk.y2, blk.x2-blk.x1, blk.y2-blk.y1, blk.fontsize]        #--- 2 3 4 5 6
        a = np.empty( ( len(lNode), 6 ) , dtype=Feat_dtype)

        try:
            max_y  = max(o.y2 for o in lNode)
            mean_h = sum(abs(o.y1 - o.y2) for o in lNode) / len(lNode)
        except (ValueError, ZeroDivisionError):
            max_y, mean_h = None, None

        for i, blk in enumerate(lNode):
            y1,y2 = blk.y1, blk.y2
            h = abs(y1-y2) / mean_h
            y1,y2 = y1/max_y, y2/max_y
            a[i, :] = [  y1, y1*y1
                       , y2, y2*y2
                       ,  h, h * h
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


#-------------------------------------------------------------------------------------------------------
class NodeTransformerNeighborsAllText(Transformer):
    """
    Collects all the text from the neighbors
    On going ...
    """
    def transform(self, lNode):
        txt_list=[]
        #print('Node Text',lNode.text)
        for _i,blk in enumerate(lNode):
            txt_H = ' '.join(o.text for o in blk.lHNeighbor)
            txt_V = ' '.join(o.text for o in blk.lVNeighbor)
            txt_list.append(' '.join([txt_H, txt_V]))

        return txt_list

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


class Node1ConstantFeature(Transformer):
    """
    we generate one constant feature per node. (1.0)
    """
    def transform(self, lNode):
        return np.ones( ( len(lNode), 1 ) , dtype=Feat_dtype)


class SemanticFeature(Transformer):
    """
        use sSemAttr as feature
    """
    
class NodeSemanticLabels(Transformer): 
    """
    Ground truth semantic labels as features
    
    """
    
    def __init__(self
                 , lLabel=None
                 , labelFun=None):
        #Animesh code by default
        if not lLabel is None: self.lLabel = lLabel
        self.labelFun = NodeSemanticLabels.AnimeshLabelFun if (labelFun is None) else labelFun
    
    @staticmethod
    def AnimeshLabelFun(lLabel, lNode, aa):
        for i, blk in enumerate(lNode):
            try:
#                 s = blk.node.attrib['DU_sem']
                s = blk.getSemAttribute()
            except:
                s = 'IGNORE'
            aa[i, lLabel.index(s)] = 1.0
        return aa
    
    def transform(self, lNode):
        aa = np.zeros( ( len(lNode), len(self.lLlabel) ) , dtype=Feat_dtype)  

        return self.labelFun(self.lLabel, lNode, aa)
   

#------------------------------------------------------------------------------------------------------
class NodeTransformerTextEnclosed_NormedDigits(Transformer):
    """
    we will get a list of block and need to send back what a textual feature extractor (TfidfVectorizer) needs.
    So we return a list of strings  
    SPECIFIC :  1-9 digits normalized to §
    """
    TR = str.maketrans("123456789", "§§§§§§§§§")
    def transform(self, lNode):
        return map(lambda x: "{%s}"%str.translate(x.text, NodeTransformerTextEnclosed.TR), lNode)


#------------------------------------------------------------------------------------------------------

class EdgeBooleanAlignmentFeatures(EdgeTransformerClassShifter):
    """
    we will get a list of edges and return a boolean array, directly
    
    We ignore the page information

    vertical-, horizontal- centered  (at epsilon precision, epsilon typically being 5pt ?)
    left-, top-, right-, bottom- justified  (at epsilon precision)
    """
    nbFEAT = 6 
    
    def transform(self, lEdge):
        #DISC a = np.zeros( ( len(lEdge), 16 ) , dtype=Feat_dtype)
        a = - np.ones( ( len(lEdge), self._nbEdgeFeat ) , dtype=Feat_dtype)
        
        try:
            mean_h_A = sum(abs(o.A.y1 - o.A.y2) for o in lEdge) / len(lEdge)
            mean_h_B = sum(abs(o.B.y1 - o.B.y2) for o in lEdge) / len(lEdge)
            mean_h = (mean_h_A + mean_h_B) / 2
        except (ValueError, ZeroDivisionError):
            mean_h = None
        
        # When to decide of an alignment
        thH = mean_h / fALIGNMENT_COEF
        
        for i, edge in enumerate(lEdge):
            z = self._dEdgeClassIndexShift[edge.__class__]
            
            A,B = edge.A, edge.B        

            a[i,z:z+self.nbFEAT] = ( A.x1 + A.x2 - (B.x1 + B.x2) <= thH, # centering
                                     A.y1 + A.y2 - (B.y1 + B.y2) <= thH, 
                                     abs(A.x1-B.x1) <= thH,              #justified
                                     abs(A.y1-B.y1) <= thH,
                                     abs(A.x2-B.x2) <= thH,
                                     abs(A.y2-B.y2) <= thH
                                     )
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
    nbFEAT = 6-1+8
    
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
            
            #
            na, nb = len(A.text), len(B.text)
            lcs = lcs_length(A.text,na, B.text,nb)
            try:
                a[i, z+1] =  float( lcs / (na+nb-lcs) )
            except ZeroDivisionError:
                pass
            
            #new in READ: the length of a same-page edge
#             if isinstance(edge, SamePageEdge):
#                 if isinstance(edge, HorizontalEdge):
#                     norm_length = edge.length / mean_length_h
#                     #                Horiz.       Vert.         Horiz.                   Vert.  
#                     a[i,z+2:z+8] = (0.0, edge.length, 0.0        , norm_length   , 0.0                     , norm_length*norm_length )
#                 else:
#                     norm_length = edge.length / mean_length_v
#                     a[i,z+2:z+8] = (edge.length, 0.0, norm_length, 0.0           , norm_length*norm_length , 0.0         )
            #norm_length = edge.length / mean_length
            norm_length = edge.length / dMeanLength[edge.__class__]
            
            a[i,z+2:z+5] = (edge.length, norm_length, norm_length*norm_length)
            a[i,z+5:z+5+8] = ( A.x1,A.x2, A.y1,A.y2
                             , B.x1,B.x2, B.y1,B.y2 )                    
        return a  


class EdgeNumericalSelector_noText(EdgeTransformerClassShifter):
    # Bug found in July 2020 by JLM: last feature was not used...
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
            
            #new in READ: the length of a same-page edge
#             if isinstance(edge, SamePageEdge):
#                 if isinstance(edge, VerticalEdge):
#                     norm_length = edge.length / mean_length
#                     #                Horiz.       Vert.         Horiz.                  Vert.
#                     a[i,z+1:z+7] = (0.0, edge.length, 0.0        , norm_length   , 0.0                    , norm_length*norm_length)
#                 else:
#                     
#                     a[i,z+1:z+7] = (edge.length, 0.0, norm_length, 0.0           , norm_length*norm_length , 0.0)
            norm_length = edge.length / dMeanLength[edge.__class__]
            a[i,z+1:z+4] = (edge.length, norm_length, norm_length*norm_length)
            a[i,z+4:z+4+8] = ( A.x1,A.x2, A.y1,A.y2
                             , B.x1,B.x2, B.y1,B.y2 )                    
        return a  


class EdgeNumericalSelector_noText_vPetitPiton(EdgeTransformerClassShifter):
    # Bug found in July 2020 by JLM: last feature was not used...
    nbFEAT = 3+8 + NodeTransformerXYWH_vPetitPiton.nbFEAT * 2
     
    def transform(self, lEdge):
        a = np.zeros( ( len(lEdge), self._nbEdgeFeat ) , dtype=Feat_dtype)
 
        dMaxLength = _getMaxLengthByEdgeClass(lEdge)
 
        for i, edge in enumerate(lEdge):
            z = self._dEdgeClassIndexShift[edge.__class__]
            A,B = edge.A, edge.B        
             
            #overlap
            ovr = A.significantOverlap(B, 0)
            try:
                a[i, z+0] = ovr / (A.area() + B.area() - ovr)
            except ZeroDivisionError:
                pass
             
            norm_length = edge.length / dMaxLength[edge.__class__]
            a[i,z+1:z+3] = (norm_length, norm_length*norm_length)
            a[i,z+3:z+3+NodeTransformerXYWH_vPetitPiton.nbFEAT*2] = A._hack_feat + B._hack_feat                   
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


class EdgeNumericalSelector_noText_AvgNorm_ByClass(EdgeTransformerClassShifter):
    """
    Normalizing by average edge length, by edge class
    """
    nbFEAT = 1
    
    def transform(self, lEdge):
        a = np.zeros( ( len(lEdge), self._nbEdgeFeat ) , dtype=Feat_dtype)
        
        for i, edge in enumerate(lEdge):
            z = self._dEdgeClassIndexShift[edge.__class__]
            a[i,z] = edge.length
        
        a = a / np.average(a, axis=0)
#         print(a[0:5,:])
#         print(a[-5:,:])
        return a  


class EdgeNumericalSelector_noText_NormPage_ByClass(EdgeTransformerClassShifter):
    """
    Normalizing edge length, by edge class, by image width and height
    """
    nbFEAT = 1
    
    def transform(self, lEdge):
        a = np.zeros( ( len(lEdge), self._nbEdgeFeat ) , dtype=Feat_dtype)
        
        dAttr = {HorizontalEdge:"w", VerticalEdge:"h"}
        
        for i, edge in enumerate(lEdge):
            z = self._dEdgeClassIndexShift[edge.__class__]
            pg = edge.A.page
            a[i,z] = edge.length / getattr(pg, dAttr[edge.__class__])
        
#         print(a[0:5,:], pg.w, pg.h)
#         print(a[-5:,:], pg.w, pg.h, [_e.length for _e in lEdge[-5:]])
        return a  

#------------------------------------------------------------------------------------------------------

class EdgeTypeFeature_HV(Transformer):
    """
    Only tells the type of edge: Horizontal or Vertical
    """
    def transform(self, lEdge):
        a = np.zeros( (len(lEdge), 2), dtype=Feat_dtype)
        for i, edge in enumerate(lEdge):
            #-- vertical / horizontal
            if edge.__class__ == HorizontalEdge:
                a[i,0] = 1.0
            else:
                assert edge.__class__ == VerticalEdge
                a[i,1] = 1.0
        return a
    



# -----------------------------------------------------------------------------------------------------------------------------    
def _debug(lO, a):
    for i,o in enumerate(lO):
        print(o)
        print(a[i])
        print()
                
def lcs_length(a,na, b,nb):
    """
    Compute the length of the longest common string. Very fast. JLM March 2016
    
    NOTE: I did not compare against fastlcs...
    """
    #na, nb = len(a), len(b)
    if nb < na: a, na, b, nb = b, nb, a, na
    if na==0: return 0
    na1 = na+1
    curRow  = [0]*na1
    prevRow = [0]*na1
    range1a1 = range(1, na1)
    for i in range(nb):
        bi = b[i]
        prevRow, curRow = curRow, prevRow
        curRow[0] = 0
        curRowj = 0
        for j in range1a1:
            if bi == a[j-1]:
                curRowj = max(1+prevRow[j-1], prevRow[j], curRowj)
            else:
                curRowj = max(prevRow[j], curRowj)
            curRow[j] = curRowj
    return curRowj


class NodeEdgeTransformer(Transformer):
    """
    we will get a list of list of edges ...
    """
    def __init__(self,edge_list_transformer,agg_func='sum'):
        self.agg_func=agg_func
        self.edge_list_transformer=edge_list_transformer

    def transform(self,lNode):
        x_all=[]
        for _i, blk in enumerate(lNode):
            x_edge_node = self.edge_list_transformer.transform(blk.edgeList)
            if self.agg_func=='sum':
                x_node=x_edge_node.sum(axis=0)
            elif self.agg_func=='mean':
                x_node=x_edge_node.mean(axis=0)
            else:
                raise ValueError('Invalid Argument',self.agg_func)
            x_all.append(x_node)
        return np.vstack(x_all)




