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



import ast
import numpy as np

from .Transformer import Transformer
from .Edge import HorizontalEdge, VerticalEdge, SamePageEdge
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
#         a = np.empty( ( len(lNode), 5 ) , dtype=np.float64)
#         for i, blk in enumerate(lNode): a[i, :] = [blk.x1, blk.y2, blk.x2-blk.x1, blk.y2-blk.y1, blk.fontsize]        #--- 2 3 4 5 6
        a = np.empty( ( len(lNode), 4+4+4 ) , dtype=np.float64)

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

#------------------------------------------------------------------------------------------------------
class NodeTransformerNeighbors(Transformer):
    """
    Characterising the neighborough
    """
    def transform(self, lNode):
#         a = np.empty( ( len(lNode), 5 ) , dtype=np.float64)
#         for i, blk in enumerate(lNode): a[i, :] = [blk.x1, blk.y2, blk.x2-blk.x1, blk.y2-blk.y1, blk.fontsize]        #--- 2 3 4 5 6 
        a = np.empty( ( len(lNode), 2 + 2 ) , dtype=np.float64)
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
        a = np.zeros( ( len(lNode), 6 ) , dtype=np.float64)
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
        return np.ones( ( len(lNode), 1 ) , dtype=np.float64)

class NodeTransformerTWY(Transformer):
    """
    adding the proba distro
    """

    def transform(self, lNode):
        #         a = np.empty( ( len(lNode), 5 ) , dtype=Feat_dtype)
        #         for i, blk in enumerate(lNode): a[i, :] = [blk.x1, blk.y2, blk.x2-blk.x1, blk.y2-blk.y1, blk.fontsize]        #--- 2 3 4 5 6
        assert len(lNode) > 0

        # get nb classes
        NY = len(lNode[0].node.get("DU_Y"))
        X = np.empty((len(lNode), NY), dtype=np.float64)

        for i, nd in enumerate(lNode):
            s = nd.node.get("DU_Y")
            assert s != None, "This SW needs tagged words, with a @DU_Y and %d values" % self.NY
            Y = np.array(ast.literal_eval(s), dtype=np.float)
            X[i] = Y

        return X


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
        #DISC a = np.zeros( ( len(lEdge), 16 ) , dtype=np.float64)
        a = - np.ones( ( len(lEdge), self._nbEdgeFeat ) , dtype=np.float64)
        
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



class EdgeNumericalSelector(EdgeTransformerClassShifter):
    """
    getting rid of the hand-crafted thresholds
    JLM Nov 2019: simpler and better (normalization must not change with direction for the 2 removed any direction features)
    """
    nbFEAT = 6
    
    def transform(self, lEdge):
        #no font size a = np.zeros( ( len(lEdge), 5 ) , dtype=np.float64)
#         a = np.zeros( ( len(lEdge), 7 ) , dtype=np.float64)
        a = np.zeros( ( len(lEdge), self._nbEdgeFeat ) , dtype=np.float64)

        try:
            mean_length_h = sum(o.length for o in lEdge if isinstance(o, HorizontalEdge)) / len(lEdge)
        except ZeroDivisionError:
            mean_length_h = None
        try:
            mean_length_v = sum(o.length for o in lEdge if not isinstance(o, HorizontalEdge)) / len(lEdge)
        except ZeroDivisionError:
            mean_length_v = None
        
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
            if isinstance(edge, SamePageEdge):
                if isinstance(edge, HorizontalEdge):
                    norm_length = edge.length / mean_length_h
                    #                Horiz.       Vert.         Horiz.                   Vert.  
                    a[i,z+2:z+8] = (0.0, edge.length, 0.0        , norm_length   , 0.0                     , norm_length*norm_length )
                else:
                    norm_length = edge.length / mean_length_v
                    a[i,z+2:z+8] = (edge.length, 0.0, norm_length, 0.0           , norm_length*norm_length , 0.0         )
                    
        return a  


class EdgeNumericalSelector_noText(EdgeTransformerClassShifter):
    nbFEAT = 5
    
    def transform(self, lEdge):
        a = np.zeros( ( len(lEdge), self._nbEdgeFeat ) , dtype=np.float64)

        try:
            mean_length = sum(o.length for o in lEdge) / len(lEdge)
        except ZeroDivisionError:
            mean_length = None

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
            if isinstance(edge, SamePageEdge):
                if isinstance(edge, VerticalEdge):
                    norm_length = edge.length / ( mean_length + 1e-8)
                    #                Horiz.       Vert.         Horiz.                  Vert.
                    a[i,z+1:z+7] = (0.0, edge.length, 0.0        , norm_length   , 0.0                    , norm_length*norm_length)
                else:
                    norm_length = edge.length / (mean_length + 1e-8)
                    a[i,z+1:z+7] = (edge.length, 0.0, norm_length, 0.0           , norm_length*norm_length , 0.0)
                    
        return a  


#------------------------------------------------------------------------------------------------------

class EdgeTypeFeature_HV(Transformer):
    """
    Only tells the type of edge: Horizontal or Vertical
    """
    def transform(self, lEdge):
        a = np.zeros( (len(lEdge), 2), dtype=np.float64)
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




