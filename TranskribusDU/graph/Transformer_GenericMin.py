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




import numpy as np

from common.trace import traceln

from .FeatureDType  import dtype as Feat_dtype
from .Transformer   import Transformer
from .Edge          import HorizontalEdge, VerticalEdge, SamePageEdge
import graph.Transformer_PageXml as Transformer_PageXml
# import torch
# torch.set_grad_enabled(False)
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
# class NodeTransformerTextroberta(Transformer):
#     """
#     Node text
#     """
#  
#     def __init__(self,model):
#         self.model = model
#         self.N = model.args.encoder_embed_dim #self.model.config.hidden_size
#     def transform(self, lNode):
# #         import torch
# #         torch.set_grad_enabled(False)
#         
#         nn = len(lNode)
#         a = np.zeros((nn, self.N), dtype=np.float32)
#         for i, n in enumerate(lNode):
#             ltokens =  self.model.encode(n.text)
#             last_layer_features = self.model.extract_features(ltokens)[0]
#             if len(n.text)>0: 
#                 a[i] = last_layer_features[0,:]
#         return a
#      
#     def __str__(self):
# #         return "- EMBEDDINGS %s  (embedding length=%d)" % (self.__class__, self.N)
#         return "- EMBEDDINGS (length=%d)" % (self.N)
# 
# 
# 
# 
# # -----------------------------------------------------------------------------
# class NodeTransformerTextBERT(Transformer):
#     """
#     Node text
#     """
#  
#     def __init__(self,tokenizer,model):
#         self.model = model
#         self.tokenizer=tokenizer
#         self.N = self.model.config.hidden_size
#     def transform(self, lNode):
# #         import torch
# #         torch.set_grad_enabled(False)
#  
#         nn = len(lNode)
#         ltokens = self.tokenizer.batch_encode_plus(
#             [n.text for n in lNode], 
#             pad_to_max_length=True,
#             return_tensors='pt',
#             )
#         _, pooled = self.model(ltokens['input_ids'])
#         a = np.zeros((nn, self.N), dtype=np.float32)
#         for i, n in enumerate(lNode):
#             if len(n.text)>0: 
#                 a[i] = pooled[0,:]
#         return a
#      
#     def __str__(self):
# #         return "- EMBEDDINGS %s  (embedding length=%d)" % (self.__class__, self.N)
#         return "- EMBEDDINGS (length=%d)" % (self.N)
# 



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
        x,y normalized by page W/H
    """
    def transform(self, lNode):
#         a = np.empty( ( len(lNode), 5 ) , dtype=Feat_dtype)
#         for i, blk in enumerate(lNode): a[i, :] = [blk.x1, blk.y2, blk.x2-blk.x1, blk.y2-blk.y1, blk.fontsize]        #--- 2 3 4 5 6
        a = np.empty( ( len(lNode), 6+0+0) , dtype=Feat_dtype)

        for i, blk in enumerate(lNode):
            page = blk.page
            x1,y1,x2,y2 = blk.x1, blk.y1, blk.x2, blk.y2
            w = abs(x1-x2) 
            h = abs(y1-y2) 
#             x1,x2 = x1/page.w, x2/page.w
#             y1,y2 = y1/page.h, y2/page.h
            a[i, :] = [  x1#, x1*x1
                       , x2#, x2*x2
                       , y1#, y1*y1
                       , y2#, y2*y2
                       ,  w#, w * w
                       ,  h#, h * h
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
        a = np.empty( ( len(lNode), 2 + 0 ) , dtype=Feat_dtype)
        for i, blk in enumerate(lNode): 
#             ax1, ay1 = blk.x1, blk.y1
            #number of horizontal/vertical/crosspage neighbors
            a[i,0:2] = len(blk.lHNeighbor), len(blk.lVNeighbor)
            #number of horizontal/vertical/crosspage neighbors occuring after this block
#             a[i,2:4] = (sum(1 for _b in blk.lHNeighbor  if _b.x1 > ax1), 
#                         sum(1 for _b in blk.lVNeighbor  if _b.y1 > ay1))
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


class Node1ConstantFeature(Transformer):
    """
    we generate one constant feature per node. (1.0)
    """
    def transform(self, lNode):
        return np.ones( ( len(lNode), 1 ) , dtype=Feat_dtype)


#------------------------------------------------------------------------------------------------------

class EdgeBooleanAlignmentFeatures(EdgeTransformerClassShifter):
    """
    we will get a list of edges and return a boolean array, directly
    
    We ignore the page information

    vertical-, horizontal- centered  (at epsilon precision, epsilon typically being 5pt ?)
    left-, top-, right-, bottom- justified  (at epsilon precision)
    
    
    
    area as reified edge
    
    """
    nbFEAT = 6
    
    def transform(self, lEdge):
        #DISC a = np.zeros( ( len(lEdge), 16 ) , dtype=Feat_dtype)
        a = - np.ones( ( len(lEdge), self._nbEdgeFeat ) , dtype=Feat_dtype)
        
#         try:
#             mean_h_A = sum(abs(o.A.y1 - o.A.y2) for o in lEdge) / len(lEdge)
#             mean_h_B = sum(abs(o.B.y1 - o.B.y2) for o in lEdge) / len(lEdge)
#             mean_h = (mean_h_A + mean_h_B) / 2
#         except (ValueError, ZeroDivisionError):
#             mean_h = None
        
        # When to decide of an alignment
#         thH = mean_h / fALIGNMENT_COEF
        
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



class EdgeNumericalSelector(EdgeTransformerClassShifter):
    """
    getting rid of the hand-crafted thresholds
    """
    nbFEAT = 2
#     nbFEAT = 3  + 2
#     nbFEAT = 3  + 4
#     nbFEAT = 3  + 8
    
    def transform(self, lEdge):
        #no font size a = np.zeros( ( len(lEdge), 5 ) , dtype=Feat_dtype)
#         a = np.zeros( ( len(lEdge), 7 ) , dtype=Feat_dtype)
#         a = np.zeros( ( len(lEdge), self.nbFEAT ) , dtype=Feat_dtype)
        a = np.zeros( ( len(lEdge), self._nbEdgeFeat ) , dtype=Feat_dtype)
#         a = np.ones( ( len(lEdge), self._nbEdgeFeat ) , dtype=Feat_dtype)
#         return a 
        for i, edge in enumerate(lEdge):
            z = self._dEdgeClassIndexShift[edge.__class__]
#             if isinstance(edge,VerticalEdge):
#                 a[i,0] = 0.66
#             else:
#                 a[i,0] = 0.33
                
#             z = self._dEdgeClassIndexShift[edge.__class__]
            A,B = edge.A, edge.B        
            
            
#             #
#             na, nb = len(A.text), len(B.text)
#             lcs = lcs_length(A.text,na, B.text,nb)
#             try:
#                 a[i, z+1] =  float( lcs / (na+nb-lcs) )
#             except ZeroDivisionError:
#                 pass
            
            #new in READ: the length of a same-page edge
            if isinstance(edge, SamePageEdge):
                if isinstance(edge, HorizontalEdge):
                    a[i,z+0] = edge.length/ A.page.h
                else:
                    a[i,z+0] = edge.length / A.page.w
            
#             # page.w page.h  ???
#             
#             A.h = A.y2 - A.y1
#             A.w = A.x2 - A.x1 
#             B.h = B.y2 - B.y1
#             B.w = B.x2 - B.x1              
# #             print(A.h/B.h, A.h/B.w, A.w/B.w, A.w/B.h) 
#             ratio = (# A.h/B.h, A.h/B.w, A.w/B.w, A.w/B.h,
# #                     abs(A.x1 - B.x1) , abs(A.y1 - B.y1) ,  abs(A.x2 - B.x2), abs(A.y2 - B.y2) 
# #                     abs(A.x1 - B.x1) /  A.page.w, abs(A.y1 - B.y1) /  A.page.h,  abs(A.x2 - B.x2) /  A.page.w, abs(A.y2 - B.y2) /  A.page.h
#                      min(A.x1, B.x1) /  A.page.w, min(A.y1 , B.y1) /  A.page.h,  max(A.x2 , B.x2) /  A.page.w, max(A.y2 , B.y2) /  A.page.h 
#                     )
#             #ratio = (  abs(A.x1 - B.x1) /  A.page.w, abs(A.y1 - B.y1) /  A.page.h,  abs(A.x2 - B.x2) /  A.page.w, abs(A.y2 - B.y2) /  A.page.h)
# #             r = (  min( abs(A.x1, B.x1) /  A.page.w, min(A.y1 - B.y1) /  A.page.h,  max(A.x2 - B.x2) /  A.page.w, max(A.y2 - B.y2) /  A.page.h)
# #             ratio =  ( A.x1, A.x2, A.y1, A.y2, B.x1, B.x2, B.y1, B.y2)
# #             ratio = ( A.x1/A.page.w,A.x2/A.page.w, A.y1/A.page.h,A.y2/A.page.h
# #                     , B.x1/A.page.w,B.x2/A.page.w, B.y1/A.page.h,B.y2/A.page.h )
# #             a[i,3:3+8] = ( A.x1/A.page.w,A.x2/A.page.w, A.y1/A.page.h,A.y2/A.page.h
# #                         , B.x1/A.page.w,B.x2/A.page.w, B.y1/A.page.h,B.y2/A.page.h )
# #             a[i,z+1:z+1+len(ratio)] = ratio
#             
#             #X,Y,x2,x2 of the boundingbox(A,B)
        return a  


# class EdgeNumericalSelector_noText(EdgeTransformerClassShifter):
#     nbFEAT = 5
#     
#     def transform(self, lEdge):
#         a = np.zeros( ( len(lEdge), self._nbEdgeFeat ) , dtype=Feat_dtype)
# 
#         try:
#             mean_length = sum(o.length for o in lEdge) / len(lEdge)
#         except ZeroDivisionError:
#             mean_length = None
# 
#         for i, edge in enumerate(lEdge):
#             z = self._dEdgeClassIndexShift[edge.__class__]
#             A,B = edge.A, edge.B        
#             
#             #overlap
#             ovr = A.significantOverlap(B, 0)
#             try:
#                 a[i, z+0] = ovr / (A.area() + B.area() - ovr)
#             except ZeroDivisionError:
#                 pass
#             
#             #new in READ: the length of a same-page edge
#             if isinstance(edge, SamePageEdge):
#                 if isinstance(edge, VerticalEdge):
#                     a[i,z+1:z+7] = (0.0, edge.length, 0.0        , norm_length   , 0.0                    , norm_length*norm_length)
#                     a[i,z+1:z+7] = (edge.length, 0.0, norm_length, 0.0           , norm_length*norm_length , 0.0)
#                     
#         return a  

class EdgeNumericalSelector_noText(EdgeTransformerClassShifter):
    nbFEAT = 2

    def transform(self, lEdge):
        a = np.zeros( ( len(lEdge),  4) , dtype=Feat_dtype)
        #a = np.zeros( ( len(lEdge),  self._nbEdgeFeat) , dtype=Feat_dtype)

        for i, edge in enumerate(lEdge):
            z = self._dEdgeClassIndexShift[edge.__class__]
            A,B = edge.A, edge.B

            #overlap
            ovr = A.significantOverlap(B, 0)
            try:
                a[i, z+0] = ovr / (A.area() + B.area() - ovr)
            except ZeroDivisionError:
                pass

            a[i,z+1:z+2] = (edge.length)
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




