# -*- coding: utf-8 -*-

"""
    Node and edge feature transformers to extract features for PageXml
    
    Copyright Xerox(C) 2016 JL. Meunier
    
    v2 March 2017 JLM

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
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""
import numpy as np

from Transformer import Transformer
from Edge import HorizontalEdge, VerticalEdge, SamePageEdge, CrossPageEdge

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
# class NodeTransformerNeighborText(Transformer):
#     """
#     for each node, returning the space-separated text of all its neighbor on the page
#     """
#     def transform(self, lNode):
#         return [" ".join([nd2.text for nd2 in nd1.lNeighbor]) for nd1 in lNode]  

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
        a = np.empty( ( len(lNode), 2+4+2+4 ) , dtype=np.float64)
        for i, blk in enumerate(lNode): 
            page = blk.page
            x1,y1,x2,y2 = blk.x1, blk.y1, blk.x2, blk.y2
            w,h = float(page.w), float(page.h)
            #Normalize by page with and height
            xn1, yn1, xn2, yn2 = x1/w, y1/h, x2/w, y2/h
            #generate X-from-binding
            if page.bEven:
                xb1, xb2    = w - x2    , w - x1
                xnb1, xnb2  = 1.0 - xn2 , 1.0 - xn1
            else:
                xb1, xb2    = x1    , x2
                xnb1, xnb2  = xn1   , xn2
            a[i, :] = [xb1, xb2     , x1, y2, x2-x1, y2-y1   , xnb1, xnb2   , xn1, yn2, xn2-xn1, yn2-yn1] 
        return a

class NodeTransformerXYWH_v2(Transformer):
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
class NodeTransformerNeighbors(Transformer):
    """
    Characterising the neighborough
    """
    def transform(self, lNode):
#         a = np.empty( ( len(lNode), 5 ) , dtype=np.float64)
#         for i, blk in enumerate(lNode): a[i, :] = [blk.x1, blk.y2, blk.x2-blk.x1, blk.y2-blk.y1, blk.fontsize]        #--- 2 3 4 5 6 
        a = np.empty( ( len(lNode), 3+3 ) , dtype=np.float64)
        for i, blk in enumerate(lNode): 
            #number of horizontal/vertical/crosspage neighbors
            a[i,0] = len(blk.lHNeighbor)
            a[i,1] = len(blk.lVNeighbor)
            a[i,2] = len(blk.lCPNeighbor)
            #number of horizontal/vertical/crosspage neighbors occuring after this block
            ax1, ay1, apnum = blk.x1, blk.y1, blk.pnum
            a[i,3] = sum(1 for _b in blk.lHNeighbor  if _b.x1 > ax1)
            a[i,4] = sum(1 for _b in blk.lVNeighbor  if _b.y1 > ay1)
            a[i,5] = sum(1 for _b in blk.lCPNeighbor if _b.pnum > apnum)
            #number of horizontal/vertical/crosspage neighbors occuring after this block
            #better to give direct info
            a[i,0:3] = a[i,0:3] - a[i,3:6]
        
        #_debug(lNode, a)
        return a

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
            if s.isalnum(): a[i, 0] = 1.0 
            if s.isalpha(): a[i, 1] = 1.0 
            if s.isdigit(): a[i, 2] = 1.0 
            if s.islower(): a[i, 3] = 1.0
            if s.istitle(): a[i, 4] = 1.0 
            if s.isupper(): a[i, 5] = 1.0
            if blk.pnum%2 == 0: a[i, 6] = 1.0 #odd/even page number
            
            #new in READ
            #are we in page 1 or 2 or next ones?
            a[i, 6 +max(1 , min(3, blk.pnum))]      = 1.0  #  a[i, 7-8-9 ]
            #are we in page -2 or -1 or previous ones?
            a[i, 12+max(-2, blk.pnum-blk.page.pagecnt)]  = 1.0  #  a[i, 10-11-12 ]
            #a[i,blk.orientation] = 1.0   
            
        return a

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
class EdgeTransformerSourceText(Transformer):
    """
    we will get a list of edges and need to send back what a textual feature extractor (TfidfVectorizer) needs.
    So we return a list of strings, of the source node of the edge
    """
    def __init__(self, n):
        Transformer.__init__(self)
        self._edgeClass = [HorizontalEdge, VerticalEdge, CrossPageEdge][n]
        
    def transform(self, lEdge):
        #return map(lambda x: x.A.text, lEdge)
#         return map(lambda x: "{%s}"%x.A.text, lEdge)
        return map(lambda edge: "{%s}"%edge.A.text if isinstance(edge, self._edgeClass) else "_", lEdge)

#------------------------------------------------------------------------------------------------------
class EdgeTransformerTargetText(Transformer):
    """
    we will get a list of edges and need to send back what a textual feature extractor (TfidfVectorizer) needs.
    So we return a list of strings, of the source node of the edge
    """
    def __init__(self, n):
        Transformer.__init__(self)
        self._edgeClass = [HorizontalEdge, VerticalEdge, CrossPageEdge][n]

    def transform(self, lEdge):
        #return map(lambda x: x.B.text, lEdge)
#         return map(lambda x: "{%s}"%x.B.text, lEdge)
        return map(lambda edge: "{%s}"%edge.B.text if isinstance(edge, self._edgeClass) else "_", lEdge)

#------------------------------------------------------------------------------------------------------
class Edge1HotFeatures(Transformer):
    """
    we will get a list of edges and return a boolean array, directly

    above/below, left/right, neither but on same page
    same or consecutive pages    
    vertical-, horizontal- centered  (at epsilon precision, epsilon typically being 5pt ?)
    left-, top-, right-, bottom- justified  (at epsilon precision)
    TODO sequentiality of content
    TODO crossing ruling-line

    """
    def __init__(self, pageNumSequenciality):
        Transformer.__init__(self)
        self.sqnc = pageNumSequenciality
    def fit(self, x, y=None):
        return self
    
    def transform(self, lEdge):
        a = np.zeros( ( len(lEdge), 3 + 17*3 ) , dtype=np.float64)
        for i, edge in enumerate(lEdge):
            #-- vertical / horizontal / virtual / cross-page / not-neighbor
            if isinstance(edge, VerticalEdge):
                a[i,0] = 1.0
                z = 0
            elif isinstance(edge, HorizontalEdge):
                a[i,1] = 1.0
                z = 17
            elif isinstance(edge, CrossPageEdge):
                a[i,2] = 1.0
                z = 34
            else:
                assert False, "INTERNAL ERROR: unknown type of edge"
            
# 14/12/2016 - useless because of A[i, 0:1]            
#             #-- same or consecutive page
#             A,B = edge.A, edge.B        
#             if A.pnum == B.pnum: 
#                 a[i,3] = 1.0
            
            A,B = edge.A, edge.B        
            
            #sequenciality, either None, or increasing or decreasing
            sA, sB = A.text, B.text
            if sA == sB: a[i, z + 3] = 1.0
            if self.sqnc.isPossibleSequence(sA, sB):
                fInSequence = 1.0
            elif self.sqnc.isPossibleSequence(sB, sA): 
                fInSequence = -1.0
            else:
                fInSequence = 0.0
                
            if A.pnum == B.pnum:
                a[i, z + 4] = fInSequence          #-1, 0, +1
                a[i, z + 5] = abs(fInSequence)     # 0 or 1
            else:
                a[i, z + 6] = fInSequence          #-1, 0, +1
                a[i, z + 7] = abs(fInSequence)     # 0 or 1
             
            if sA.isalnum(): a[i, z +  8] = 1.0
            if sA.isalpha(): a[i, z +  9] = 1.0
            if sA.isdigit(): a[i, z + 10] = 1.0
            if sA.islower(): a[i, z + 11] = 1.0
            if sA.istitle(): a[i, z + 12] = 1.0
            if sA.isupper(): a[i, z + 13] = 1.0  
                       
            if sB.isalnum(): a[i, z + 14] = 1.0
            if sB.isalpha(): a[i, z + 15] = 1.0
            if sB.isdigit(): a[i, z + 16] = 1.0
            if sB.islower(): a[i, z + 17] = 1.0
            if sB.istitle(): a[i, z + 18] = 1.0
            if sB.isupper(): a[i, z + 19] = 1.0  

        return a

#------------------------------------------------------------------------------------------------------
class EdgeBooleanFeatures(Transformer):
    """
    we will get a list of edges and return a boolean array, directly

    vertical-, horizontal- centered  (at epsilon precision, epsilon typically being 5pt ?)
    left-, top-, right-, bottom- justified  (at epsilon precision)
    """
    def transform(self, lEdge):
        #DISC a = np.zeros( ( len(lEdge), 16 ) , dtype=np.float64)
        a = - np.ones( ( len(lEdge), 3*6 ) , dtype=np.float64)
        for i, edge in enumerate(lEdge):
            if isinstance(edge, VerticalEdge):
                z=0
            elif isinstance(edge, HorizontalEdge):
                z=6
            elif isinstance(edge, CrossPageEdge):
                z=12
            
            A,B = edge.A, edge.B        
            #-- centering
            if A.x1 + A.x2 - (B.x1 + B.x2) <= 2 * fEPSILON:     #horizontal centered
                a[i, z + 0] = 1.0
            if A.y1 + A.y2 - (B.y1 + B.y2) <= 2 * fEPSILON:     #V centered
                a[i, z + 1] = 1.0
            
            #justified
            if abs(A.x1-B.x1) <= fEPSILON:
                a[i, z + 2] = 1.0
            if abs(A.y1-B.y1) <= fEPSILON:
                a[i, z + 3] = 1.0
            if abs(A.x2-B.x2) <= fEPSILON:
                a[i, z + 4] = 1.0
            if abs(A.y2-B.y2) <= fEPSILON:
                a[i, z + 5] = 1.0       
        #_debug(lEdge, a)
        return a

class EdgeBooleanFeatures_v2(Transformer):
    """
    we will get a list of edges and return a boolean array, directly

    vertical-, horizontal- centered  (at epsilon precision, epsilon typically being 5pt ?)
    left-, top-, right-, bottom- justified  (at epsilon precision)
    """
    def transform(self, lEdge):
        #DISC a = np.zeros( ( len(lEdge), 16 ) , dtype=np.float64)
        a = - np.ones( ( len(lEdge), 3*6 ) , dtype=np.float64)
        for i, edge in enumerate(lEdge):
            if isinstance(edge, VerticalEdge):
                z=0
            elif isinstance(edge, HorizontalEdge):
                z=6
            elif isinstance(edge, CrossPageEdge):
                z=12
            
            A,B = edge.A, edge.B        
            
            thH = fEPSILON_v2 * (A.page.w+B.page.w) / 2.0
            thV = fEPSILON_v2 * (A.page.h+B.page.h) / 2.0
            
            #-- centering
            if A.x1 + A.x2 - (B.x1 + B.x2) <= thH:     #horizontal centered
                a[i, z + 0] = 1.0
            if A.y1 + A.y2 - (B.y1 + B.y2) <= thV:     #V centered
                a[i, z + 1] = 1.0
            
            #justified
            if abs(A.x1-B.x1) <= thH:   a[i, z + 2] = 1.0
            if abs(A.y1-B.y1) <= thV:   a[i, z + 3] = 1.0
            if abs(A.x2-B.x2) <= thH:   a[i, z + 4] = 1.0
            if abs(A.y2-B.y2) <= thV:   a[i, z + 5] = 1.0
                   
        #_debug(lEdge, a)
        return a

#------------------------------------------------------------------------------------------------------
class EdgeNumericalSelector(Transformer):
    """
    we will get a list of block and need to send back what StandardScaler needs for in-place scaling, a numpy array!.

    overlap size (ratio of intersection to union of surfaces)
    max(overlap size, 5000)
    identical content in [0, 1] as ratio of lcs to "union"
    max( lcs, 25)
    """
    def transform(self, lEdge):
        #no font size a = np.zeros( ( len(lEdge), 5 ) , dtype=np.float64)
#         a = np.zeros( ( len(lEdge), 7 ) , dtype=np.float64)
        a = np.zeros( ( len(lEdge), 3*8 ) , dtype=np.float64)
        for i, edge in enumerate(lEdge):
            if isinstance(edge, VerticalEdge):
                z=0
            elif isinstance(edge, HorizontalEdge):
                z=8
            elif isinstance(edge, CrossPageEdge):
                z=16

            A,B = edge.A, edge.B        
            
            #overlap
            ovr = A.significantOverlap(B, 0)
            try:
                a[i, z+0] = ovr / (A.area() + B.area() - ovr)
            except ZeroDivisionError:
                pass
            a[i, z+1] = min(ovr, 5000.0)
            
            #
            na, nb = len(A.text), len(B.text)
            lcs = lcs_length(A.text,na, B.text,nb)
            try:
                a[i, z+2] =  float( lcs / (na+nb-lcs) )
            except ZeroDivisionError:
                pass
            a[i, z+3] = min(lcs, 50.0)
            a[i, z+4] = min(lcs, 100.0)
            
            #new in READ: the length of a same-page edge, along various normalisation schemes
            if isinstance(edge, SamePageEdge):
                if isinstance(edge, VerticalEdge):
                    norm_length = edge.length / float(edge.A.page.h)
                    a[i, z+5] = norm_length
                else:
                    norm_length = edge.length / float(edge.A.page.w)
                    a[i, z+6] = norm_length
                a[i, z+7] = norm_length    #normalised length whatever direction it has
                    
            #if A.txt == B.txt: a[i, z+ 4] = 1.0
            
#             #fontsize
#             a[i, z+5] = B.fontsize - A.fontsize
#             a[i, z+6] = (B.fontsize+1) / (A.fontsize+1)
            
        return a  

# -----------------------------------------------------------------------------------------------------------------------------    
def _debug(lO, a):
    for i,o in enumerate(lO):
        print o
        print a[i]
        print
                
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

