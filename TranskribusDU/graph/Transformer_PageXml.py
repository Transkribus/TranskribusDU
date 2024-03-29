# -*- coding: utf-8 -*-

"""
    Node and edge feature transformers to extract features for PageXml
    
    Copyright Xerox(C) 2016 JL. Meunier
    
    v2 March 2017 JLM


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""




import numpy as np

from .Transformer import Transformer
from .Edge import HorizontalEdge, VerticalEdge, SamePageEdge, CrossPageEdge, CrossMirrorPageEdge

fEPSILON = 10

fEPSILON_v2 = 1.5 /210     #1.5mm to the width of a A4 page

#------------------------------------------------------------------------------------------------------
class NodeTransformerText(Transformer):
    """
    we will get a list of block and need to send back what a textual feature extractor (TfidfVectorizer) needs.
    So we return a list of strings  
    """
    def transform(self, lNode):
        return map(lambda x: x.text, lNode)

#------------------------------------------------------------------------------------------------------
class NodeTransformerTextEnclosed(Transformer):
    """
    we will get a list of block and need to send back what a textual feature extractor (TfidfVectorizer) needs.
    So we return a list of strings  
    """
    def transform(self, lNode):
        return map(lambda x: "{%s}"%x.text, lNode) #start/end characters

#------------------------------------------------------------------------------------------------------
class NodeTransformerNeighborText(Transformer):
    """
    for each node, returning the space-separated text of all its neighbor on the page
    """
    def transform(self, lNode):
        return [" ".join([nd2.text for nd2 in nd1.lNeighbor]) for nd1 in lNode]

#------------------------------------------------------------------------------------------------------
# class NodeTransformerCrossPageNeighborText(Transformer):
#     """
#     for each node, returning the space-separated text of all its neighbor on the previous or next page
#     """
#     def transform(self, lNode):
#         return [" ".join([nd2.text for nd2 in nd1.lCPNeighbor]) for nd1 in lNode]  


#------------------------------------------------------------------------------------------------------
class NodeTransformerTextLen(Transformer):
    """
    we will get a list of block and need to send back what StandardScaler needs for in-place scaling, a numpy array!.
    So we return a numpy array  
    """
    def transform(self, lNode):
        a = np.empty( ( len(lNode), 2 ) , dtype=np.float64)             #--- FEAT #1  text length
        #for i, blk in enumerate(lNode): a[i] = len(blk.text)
        for i, blk in enumerate(lNode):
            a[i,:] = len(blk.text), blk.text.count(' ')
        return a


#------------------------------------------------------------------------------------------------------
class NodeTransformerXYWH(Transformer):
    """
    we will get a list of block and need to send back what StandardScaler needs for in-place scaling, a numpy array!.
    So we return a numpy array  
    """
    def transform(self, lNode):
#         a = np.empty( ( len(lNode), 5 ) , dtype=np.float64)
#         for i, blk in enumerate(lNode): a[i, :] = [blk.x1, blk.y2, blk.x2-blk.x1, blk.y2-blk.y1, blk.fontsize]        #--- 2 3 4 5 6 
        a = np.empty( ( len(lNode), 2+4+4 ) , dtype=np.float64)
        for i, blk in enumerate(lNode): 
            page = blk.page
            x1,y1,x2,y2 = blk.x1, blk.y1, blk.x2, blk.y2
            w,h = float(page.w), float(page.h)
            #Normalize by page with and height to range (-1, +1]
            xn1, yn1, xn2, yn2 = 2*x1/w-1, 2*y1/h-1, 2*x2/w-1, 2*y2/h-1
            #generate X-from-binding
            if page.bEven:
                xnb1, xnb2  = -xn2 , -xn1
            else:
                xnb1, xnb2  =  xn1 , xn2
            a[i, :] = [xnb1, xnb2    , xn1, yn1, xn2-xn1, yn2-yn1    , xn1*xn1, yn1*yn1, xn2*xn2, yn2*yn2] 
        return a

#------------------------------------------------------------------------------------------------------

class NodeTransformerXYWH_NoPage(Transformer):
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
            a[i, :] = [  x1, x2, x1*x1, x2*x2
                       , y1, y2, y1*y1, y2*y2
                       , w , h , w *w , h *h
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
        a = np.empty( ( len(lNode), 3+3 ) , dtype=np.float64)
        for i, blk in enumerate(lNode): 
            ax1, ay1, apnum = blk.x1, blk.y1, blk.pnum
            #number of horizontal/vertical/crosspage neighbors
            a[i,0:3] = len(blk.lHNeighbor), len(blk.lVNeighbor), len(blk.lCPNeighbor)
            #number of horizontal/vertical/crosspage neighbors occuring after this block
            a[i,3:6] = (sum(1 for _b in blk.lHNeighbor  if _b.x1 > ax1), 
                        sum(1 for _b in blk.lVNeighbor  if _b.y1 > ay1), 
                        sum(1 for _b in blk.lCPNeighbor if _b.pnum > apnum))
            #better to give direct info
            a[i,0:3] = a[i,0:3] - a[i,3:6]
        
        #_debug(lNode, a)
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
            txt_block=[]
            #print [" ".join([nd2.text for nd2 in blk.lHNeighbor])]

            for blk_neighbor in blk.lHNeighbor:
                txt_block.append(blk_neighbor.text)
            for blk_neighbor in blk.lVNeighbor:
                txt_block.append(blk_neighbor.text)
            for blk_neighbor in blk.lCPNeighbor:
                txt_block.append(blk_neighbor.text)

            txt_list.append(' '.join(txt_block))

        return txt_list




#------------------------------------------------------------------------------------------------------
class Node1HotFeatures(Transformer):
    """
    we will get a list of block and return a one-hot encoding, directly
    """
    def transform(self, lNode):
        #We allocate TWO more columns to store in it the tfidf and idf computed at document level.
        #a = np.zeros( ( len(lNode), 10 ) , dtype=np.float64)  # 4 possible orientations: 0, 1, 2, 3
        a = np.zeros( ( len(lNode), 7+3+3 ) , dtype=np.float64)  # 4 possible orientations: 0, 1, 2, 3
        for i, blk in enumerate(lNode): 
            s = blk.text
            a[i,0:7] = ( s.isalnum(),
                         s.isalpha(),
                         s.isdigit(),
                         s.islower(),
                         s.istitle(),
                         s.isupper(),
                         blk.pnum%2 == 0)
            #new in READ
            #are we in page 1 or 2 or next ones?
            a[i, 6 +max(1 , min(3, blk.pnum))]      = 1.0  #  a[i, 7-8-9 ]
            #are we in page -2 or -1 or previous ones?
            a[i, 12+max(-2, blk.pnum-blk.page.pagecnt)]  = 1.0  #  a[i, 10-11-12 ]
            #a[i,blk.orientation] = 1.0   

 
        return a

class Node1HotFeatures_noText(Transformer):
    """
    we will get a list of block and return a one-hot encoding, directly
    noText = without _any_ text-related feature
    """
    def transform(self, lNode):
        #We allocate TWO more columns to store in it the tfidf and idf computed at document level.
        #a = np.zeros( ( len(lNode), 10 ) , dtype=np.float64)  # 4 possible orientations: 0, 1, 2, 3
        a = np.zeros( ( len(lNode), 7 ) , dtype=np.float64)  # 4 possible orientations: 0, 1, 2, 3
        
        for i, blk in enumerate(lNode): 
            a[i,0] = int(blk.pnum%2 == 0)
            #new in READ
            #are we in page 1 or 2 or next ones?
            a[i, max(1 , min(3, blk.pnum))]      = 1.0  #  a[i, 1-2-3 ]
            #are we in page -2 or -1 or previous ones?
            a[i, 6+max(-2, blk.pnum-blk.page.pagecnt)]  = 1.0  #  a[i, 4-5-6 ]
            #a[i,blk.orientation] = 1.0   
            
        return a		
		
#------------------------------------------------------------------------------------------------------
#Added by Animesh
class NodeSemanticLabels(Transformer): 
    """
    Ground truth semantic labels as features
    
    JL tried to make it a bit more generic
    By default it does what Animesh wanted
    """
    lLabel=[ 'resolution-marginalia', 'resolution-number', 'resolution-paragraph'
            , 'page-number',  'heading', 'header', 'other'
           , 'IGNORE']
    
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
                s = blk.node.attrib['DU_sem']
            except:
                s = 'IGNORE'
            aa[i, lLabel.index(s)] = 1.0
        return aa
    
    def transform(self, lNode):
        aa = np.zeros( ( len(lNode), len(self.lLlabel) ) , dtype=np.float64)  

        return self.labelFun(self.lLabel, lNode, aa)

# Animesh code
#     def transform(self, lNode):
#         labels = [ 'resolution-marginalia', 'resolution-number', 'resolution-paragraph',
#                    'page-number',  'heading', 'header', 'other', 'IGNORE']
#         aa = np.zeros( ( len(lNode), len(labels) ) , dtype=np.float64)  
# 
#         for i, blk in enumerate(lNode):
#             try:
#                 s = blk.node.attrib['DU_sem']
#             except:
#                 s = 'IGNORE'
#             aa[i, labels.index(s)] = 1.0
#  
#         return aa


class Node1ConstantFeature(Transformer):
    """
    we generate one constant feature per node. (1.0)
    """
    def transform(self, lNode):
        return np.ones( ( len(lNode), 1 ) , dtype=np.float64)

# #------------------------------------------------------------------------------------------------------
# class NodePNumFeatures(Transformer):
#     """
#     does it look like following a plan number pattern after normalization??
#     """
#     def transform(self, lNode):
#         a = np.zeros( ( len(lNode), len(lCRE) ) , dtype=np.float64)
#         
#         for i, blk in enumerate(lNode): 
#             s, sconf = normalizeNumber(blk.text, blk.sconf)
#             a[i, :] = [1.0 if cre.match(s, 0) else 0.0 for cre in lCRE]
# 
#         return a

#------------------------------------------------------------------------------------------------------
    
# --- EDGES

class EdgeTransformerByClassIndex(Transformer):
    """
    We are interested only in 1 class of edge
    """
    def __init__(self, n, bMirrorPage=True, bMultiPage=True):
        Transformer.__init__(self)
        if bMirrorPage:
            self._edgeClass = [HorizontalEdge, VerticalEdge, CrossPageEdge, CrossMirrorPageEdge][n]
        else:
            if bMultiPage:
                self._edgeClass = [HorizontalEdge, VerticalEdge, CrossPageEdge][n]
            else:   
                self._edgeClass = [HorizontalEdge, VerticalEdge][n]

        self.n = n
        self.bMirrorPage = bMirrorPage
        self.bMultiPage = bMultiPage

class EdgeTransformerSourceText(EdgeTransformerByClassIndex):
    """
    we will get a list of edges and need to send back what a textual feature extractor (TfidfVectorizer) needs.
    So we return a list of strings, of the source node of the edge
    """
    def transform(self, lEdge):
        #return map(lambda x: x.A.text, lEdge)
#         return map(lambda x: "{%s}"%x.A.text, lEdge)
        return map(lambda edge: "{%s}"%edge.A.text if isinstance(edge, self._edgeClass) else "_", lEdge)

#------------------------------------------------------------------------------------------------------
class EdgeTransformerTargetText(EdgeTransformerByClassIndex):
    """
    we will get a list of edges and need to send back what a textual feature extractor (TfidfVectorizer) needs.
    So we return a list of strings, of the source node of the edge
    """
    def transform(self, lEdge):
        #return map(lambda x: x.B.text, lEdge)
#         return map(lambda x: "{%s}"%x.B.text, lEdge)
        return map(lambda edge: "{%s}"%edge.B.text if isinstance(edge, self._edgeClass) else "_", lEdge)

#------------------------------------------------------------------------------------------------------

class EdgeTransformerClassShifter(Transformer):
    """
    We assign one range of feature per Edge class
    """
    
    nbFEAT = None   #must be specialized!!!
    
    lDefaultEdgeClass = [HorizontalEdge, VerticalEdge, CrossPageEdge]
    
    def __init__(self, bMirrorPage=True, bMultiPage=True):
        Transformer.__init__(self)
        if bMirrorPage:
            lEdgeClass = self.lDefaultEdgeClass + [CrossMirrorPageEdge] #this makes a copy
        else:
            if bMultiPage:
                lEdgeClass = list(self.lDefaultEdgeClass)
            else:
                lEdgeClass = list(self.lDefaultEdgeClass[0:2])

            
        self._nbEdgeFeat = self.nbFEAT * len(lEdgeClass)
        self._dEdgeClassIndexShift = { cls:i*self.nbFEAT for i,cls in enumerate(lEdgeClass) }
        self.bMirrorPage = bMirrorPage
        self.bMultiPage  = bMultiPage

    @classmethod
    def setDefaultEdgeClass(cls, lEdgeClass):
        """
        to set the list of edge classes the feature extractor will see
        """
        cls.lDefaultEdgeClass = lEdgeClass
        
    def fit(self, x, y=None):
        return self


class Edge1HotFeatures(EdgeTransformerClassShifter):
    """
    we will get a list of edges and return a boolean array, directly

    above/below, left/right, neither but on same page
    same or consecutive pages    
    vertical-, horizontal- centered  (at epsilon precision, epsilon typically being 5pt ?)
    left-, top-, right-, bottom- justified  (at epsilon precision)
    TODO sequentiality of content
    TODO crossing ruling-line

    """
    
    nbFEAT = 18
    
    def __init__(self, pageNumSequenciality, bMirrorPage=True):
        EdgeTransformerClassShifter.__init__(self, bMirrorPage)
        self.sqnc = pageNumSequenciality
        self.pageNumSequenciality = pageNumSequenciality

    def transform(self, lEdge):
        #a = np.zeros( ( len(lEdge), 3 + 17*3 ) , dtype=np.float64)
        a = np.zeros( ( len(lEdge), self._nbEdgeFeat), dtype=np.float64)
        for i, edge in enumerate(lEdge):
            #-- vertical / horizontal / virtual / cross-page / not-neighbor
            z = self._dEdgeClassIndexShift[edge.__class__]
            a[i, z] = 1.0
            
            A,B = edge.A, edge.B        
            
            #sequenciality, either None, or increasing or decreasing
            sA, sB = A.text, B.text
            if sA == sB: a[i, z + 1] = 1.0
            if self.sqnc.isPossibleSequence(sA, sB):
                fInSequence = 1.0
            elif self.sqnc.isPossibleSequence(sB, sA): 
                fInSequence = -1.0
            else:
                fInSequence = 0.0
                
            if A.pnum == B.pnum:
                a[i, z + 2] = fInSequence          #-1, 0, +1
                a[i, z + 3] = abs(fInSequence)     # 0 or 1
            else:
                a[i, z + 4] = fInSequence          #-1, 0, +1
                a[i, z + 5] = abs(fInSequence)     # 0 or 1
            
            a[i, z +  6:z + self.nbFEAT] = (sA.isalnum(),
                                            sA.isalpha(),
                                            sA.isdigit(),
                                            sA.islower(),
                                            sA.istitle(),
                                            sA.isupper(),
                                            sB.isalnum(),
                                            sB.isalpha(),
                                            sB.isdigit(),
                                            sB.islower(),
                                            sB.istitle(),
                                            sB.isupper()
                                )

        return a
    

#------------------------------------------------------------------------------------------------------
class EdgeBooleanFeatures(EdgeTransformerClassShifter):
    """
    we will get a list of edges and return a boolean array, directly

    vertical-, horizontal- centered  (at epsilon precision, epsilon typically being 5pt ?)
    left-, top-, right-, bottom- justified  (at epsilon precision)
    """
    nbFEAT = 6

    def __init__(self, bMirrorPage=True):
        EdgeTransformerClassShifter.__init__(self, bMirrorPage)

    def transform(self, lEdge):
        #DISC a = np.zeros( ( len(lEdge), 16 ) , dtype=np.float64)
        a = - np.ones( ( len(lEdge), self._nbEdgeFeat ) , dtype=np.float64)
        for i, edge in enumerate(lEdge):
            z = self._dEdgeClassIndexShift[edge.__class__]
            
            A,B = edge.A, edge.B        
            thH = fEPSILON_v2 * (A.page.w+B.page.w) / 2.0
            # thV = fEPSILON_v2 * (A.page.h+B.page.h) / 2.0            

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
                if isinstance(edge, VerticalEdge):
                    norm_length = edge.length / float(edge.A.page.h)
                    norm_length2 = norm_length * norm_length
                    #                Horiz.       Vert.         Horiz.       Vert.  
                    a[i,z+2:z+6] = (0.0        , norm_length   , 0.0          , norm_length2 )
                else:
                    norm_length = edge.length / float(edge.A.page.w)
                    norm_length2 = norm_length * norm_length
                    a[i,z+2:z+6] = (norm_length, 0.0           , norm_length2 , 0.0         )
                    
        return a  


class EdgeNumericalSelector_noText(EdgeTransformerClassShifter):
    nbFEAT = 5
    
    def transform(self, lEdge):
        a = np.zeros( ( len(lEdge), self._nbEdgeFeat ) , dtype=np.float64)
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
                    norm_length = edge.length / float(edge.A.page.h)
                    norm_length2 = norm_length * norm_length
                    #                Horiz.       Vert.         Horiz.          Vert.
                    a[i,z+1:z+5] = (0.0        , norm_length   , 0.0          , norm_length2)
                else:
                    norm_length = edge.length / float(edge.A.page.w)
                    norm_length2 = norm_length * norm_length
                    a[i,z+1:z+5] = (norm_length, 0.0           , norm_length2 , 0.0)
                    
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




