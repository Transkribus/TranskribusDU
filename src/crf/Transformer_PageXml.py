# -*- coding: utf-8 -*-

"""
    Node and edge feature transformers to extract features for PageXml
    

    Copyright Xerox(C) 2016 JL. Meunier

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
from Edge import HorizontalEdge, VerticalEdge, CrossPageEdge

fEPSILON = 10

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
class NodeTransformerCrossPageNeighborText(Transformer):
    """
    for each node, returning the space-separated text of all its neighbor on the previous or next page
    """
    def transform(self, lNode):
        return [" ".join([nd2.text for nd2 in nd1.lCPNeighbor]) for nd1 in lNode]  


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
                xb1, xb2    = w - x2    , h - x1
                xnb1, xnb2  = 1.0 - xn2 , 1.0 - xn1
            else:
                xb1, xb2    = x1    , x2
                xnb1, xnb2  = xn1   , xn2
            a[i, :] = [xb1, xb2     , x1, y2, x2-x1, y2-y1   , xnb1, xnb2   , xn1, yn2, xn2-xn1, yn2-yn1] 
        return a

#------------------------------------------------------------------------------------------------------
class Node1HotFeatures(Transformer):
    """
    we will get a list of block and return a one-hot encoding, directly
    """
    def transform(self, lNode):
        #We allocate TWO more columns to store in it the tfidf and idf computed at document level.
        #a = np.zeros( ( len(lNode), 10 ) , dtype=np.float64)  # 4 possible orientations: 0, 1, 2, 3
        a = np.zeros( ( len(lNode), 7 ) , dtype=np.float64)  # 4 possible orientations: 0, 1, 2, 3
        
        for i, blk in enumerate(lNode): 
            s = blk.text
            if s.isalnum(): a[i, 0] = 1.0 
            if s.isalpha(): a[i, 1] = 1.0 
            if s.isdigit(): a[i, 2] = 1.0 
            if s.islower(): a[i, 3] = 1.0
            if s.istitle(): a[i, 4] = 1.0 
            if s.isupper(): a[i, 5] = 1.0
            if blk.pnum%2 == 0: a[i, 5] = 1.0 #odd/even page number
            
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
    def transform(self, lEdge):
        #return map(lambda x: x.A.text, lEdge)
        return map(lambda x: "{%s}"%x.A.text, lEdge)

#------------------------------------------------------------------------------------------------------
class EdgeTransformerTargetText(Transformer):
    """
    we will get a list of edges and need to send back what a textual feature extractor (TfidfVectorizer) needs.
    So we return a list of strings, of the source node of the edge
    """
    def transform(self, lEdge):
        #return map(lambda x: x.B.text, lEdge)
        return map(lambda x: "{%s}"%x.B.text, lEdge)

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
        #DISC a = np.zeros( ( len(lEdge), 16 ) , dtype=np.float64)
        a = np.zeros( ( len(lEdge), 27 ) , dtype=np.float64)
        for i, edge in enumerate(lEdge):
            #-- vertical / horizontal / virtual / cross-page / not-neighbor
            if isinstance(edge, VerticalEdge):
                a[i,0] = 1.0
            elif isinstance(edge, HorizontalEdge):
                a[i,1] = 1.0
            elif isinstance(edge, CrossPageEdge):
                a[i,2] = 1.0
            else:
                assert False, "INTERNAL ERROR: unknown type of edge"
            
            #-- same or consecutive page
            A,B = edge.A, edge.B        
            if A.pnum == B.pnum: 
                a[i,3] = 1.0

            #-- centering
            if A.x1 + A.x2 - (B.x1 + B.x2) <= 2 * fEPSILON:     #horizontal centered
                a[i, 4] = 1.0
            if A.y1 + A.y2 - (B.y1 + B.y2) <= 2 * fEPSILON:     #V centered
                a[i, 5] = 1.0
            
            #justified
            if abs(A.x1-B.x1) <= fEPSILON:
                a[i, 6] = 1.0
            if abs(A.y1-B.y1) <= fEPSILON:
                a[i, 7] = 1.0
            if abs(A.x2-B.x2) <= fEPSILON:
                a[i, 8] = 1.0
            if abs(A.y2-B.y2) <= fEPSILON:
                a[i, 9] = 1.0       
            
            #sequenciality, either None, or increasing or decreasing
            sA, sB = A.text, B.text
            if self.sqnc.isPossibleSequence(sA, sB):
                fInSequence = 1.0
            elif self.sqnc.isPossibleSequence(sB, sA): 
                fInSequence = -1.0
            else:
                fInSequence = 0.0
                
            if A.pnum == B.pnum:
                a[i, 10] = fInSequence          #-1, 0, +1
                a[i, 11] = abs(fInSequence)     # 0 or 1
            else:
                a[i, 12] = fInSequence          #-1, 0, +1
                a[i, 13] = abs(fInSequence)     # 0 or 1
             
            if sA.isalnum(): a[i, 14] = 1.0
            if sA.isalpha(): a[i, 15] = 1.0
            if sA.isdigit(): a[i, 16] = 1.0
            if sA.islower(): a[i, 17] = 1.0
            if sA.istitle(): a[i, 18] = 1.0
            if sA.isupper(): a[i, 19] = 1.0  
                       
            if sB.isalnum(): a[i, 20] = 1.0
            if sB.isalpha(): a[i, 21] = 1.0
            if sB.isdigit(): a[i, 22] = 1.0
            if sB.islower(): a[i, 23] = 1.0
            if sB.istitle(): a[i, 24] = 1.0
            if sB.isupper(): a[i, 25] = 1.0  

            if sA == sB: a[i, 26] = 1.0
                                     
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
        a = np.zeros( ( len(lEdge), 5 ) , dtype=np.float64)
        for i, edge in enumerate(lEdge):
            A,B = edge.A, edge.B        
            
            #overlap
            ovr = A.significantOverlap(B, 0)
            try:
                a[i,0] = ovr / (A.area() + B.area() - ovr)
            except ZeroDivisionError:
                pass
            a[i,1] = min(ovr, 5000.0)
            
            #
            na, nb = len(A.text), len(B.text)
            lcs = lcs_length(A.text,na, B.text,nb)
            try:
                a[i,2] =  float( lcs / (na+nb-lcs) )
            except ZeroDivisionError:
                pass
            a[i,3] = min(lcs, 50.0)
            a[i,4] = min(lcs, 100.0)
            
            #if A.txt == B.txt: a[i, 4] = 1.0
            
#             #fontsize
#             a[i,5] = B.fontsize - A.fontsize
#             a[i,6] = (B.fontsize+1) / (A.fontsize+1)
            
        return a  

# -----------------------------------------------------------------------------------------------------------------------------    
                
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

