# -*- coding: utf-8 -*-

"""
    Standard Node and Edge feature transformer pipelines to extract features
    
    It supercedes Transformer_PageXml
    
    Copyright NAVER(C) 2019 JL. Meunier
    

    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from common.trace import traceln

from shapely.geometry.polygon import Polygon
from util.Shape import ShapeLoader 
from shapely.prepared import prep

 
from .FeatureDType      import dtype as Feat_dtype
from .Transformer       import Pipeline, FeatureUnion
from .Transformer       import EmptySafe_QuantileTransformer as QuantileTransformer
from .Transformer       import Transformer
from .Transformer       import SparseToDense

from .Edge import HorizontalEdge, VerticalEdge, SamePageEdge, CrossPageEdge, CrossMirrorPageEdge

lDEFAULT_EDGE_CLASS = [HorizontalEdge, VerticalEdge]
# lDEFAULT_EDGE_CLASS = [HorizontalEdge, VerticalEdge, CrossPageEdge, CrossMirrorPageEdge]
 
def getDefaultEdgeClassList():
    global lDEFAULT_EDGE_CLASS
    return lDEFAULT_EDGE_CLASS

def appendDefaultEdgeClassList(cls):
    global lDEFAULT_EDGE_CLASS
    lDEFAULT_EDGE_CLASS.append(cls)
    return lDEFAULT_EDGE_CLASS

def setDefaultEdgeClassList(l):
    global lDEFAULT_EDGE_CLASS
    lDEFAULT_EDGE_CLASS = l
    return lDEFAULT_EDGE_CLASS



#------------------------------------------------------------------------------------------------------
class Node_Geometry(Pipeline):
    """
    Node geometry feature quantiled
    """
    nQUANTILE = 32

    class Selector(Transformer):
        N = 10
        def transform(self, lNode):
            a = np.empty( ( len(lNode), self.N ) , dtype=Feat_dtype)
            for i, blk in enumerate(lNode): 
                x1,y1,x2,y2 = blk.x1, blk.y1, blk.x2, blk.y2
                page = blk.page
                w,h = float(page.w), float(page.h)
                #Normalize by page with and height to range (-1, +1]
                xn1, yn1, xn2, yn2 = 2*x1/w-1, 2*y1/h-1, 2*x2/w-1, 2*y2/h-1
                a[i, :] = [xn1, yn1, xn2, yn2
                           , xn1*xn1, yn1*yn1, xn2*xn2, yn2*yn2
                           , xn1*yn2         , xn2*yn1
                           ] 
            return a

    def __init__(self, nQuantile=None):
        self.nQuantile = Node_Geometry.nQUANTILE if nQuantile is None else nQuantile
        Pipeline.__init__(self, [
                               ('geometry' , Node_Geometry.Selector())
                             , ('quantiled', QuantileTransformer(n_quantiles=self.nQuantile, copy=False))  #use in-place scaling
                         ])
        
    def __str__(self):
        return "- Geometry %s (#%d, quantile=%d)" % (self.__class__, Node_Geometry.Selector.N, self.nQuantile)

# -----------------------------------------------------------------------------
class Node_Text_NGram(Pipeline):
    """
    Node text
    """
    class Selector(Transformer):
        bTextNotShownYet = True
        def transform(self, lNode):
            if self.bTextNotShownYet:
                traceln(" Example of text: ", list(map(lambda o: o.text, lNode[0:10])))
                self.bTextNotShownYet = False
                
            return map(lambda x: "¤%s¤"%x.text, lNode) #start/end characters
    
    def __init__(self
                 , analyzer   = 'char'  # analyzer : string, {'word', 'char', 'char_wb'} or callable
                 , n_tfidf    = 500     # how many features
                 , t_ngrams   = (2,3)   # tuple for min, max ngram size, e.g. (2,6)
                 , b_tfidf_lc = False   # boolean, lowercase or not?
                 ):
        
        # we keep a pointer to the TFIDF vectorizer to be able to clean it
        # we can also use it separately from the pipeline once fitted
        self.tdidfNodeTextVectorizer = TfidfVectorizer(lowercase     = b_tfidf_lc
                                                      , max_features = n_tfidf
                                                      , analyzer     = analyzer
                                                      , ngram_range  = t_ngrams #(2,6)
                                                      , dtype        = Feat_dtype)
        
        Pipeline.__init__(self, [
                               ('selector'    , Node_Text_NGram.Selector())
                             , ('tfidf'       , self.tdidfNodeTextVectorizer) 
                             , ('todense'     , SparseToDense())  #pystruct needs an array, not a sparse matrix
                         ])

    def __str__(self):
        return "- Text_NGram %s <<%s>>" % (self.__class__, self.tdidfNodeTextVectorizer)

    def cleanTransformers(self):
        """
        some transformers benefit from being cleaned before saved on disk...
        """
        # the TFIDF transformers are keeping the stop words => huge pickled file!!!
        self.tdidfNodeTextVectorizer.stop_words_ = None

# -----------------------------------------------------------------------------
class Node_Neighbour_Count(Pipeline):
    """
    Node neighbour count feature quantiled
    """
    nQUANTILE = 8

    class Selector(Transformer):
        """
        Characterising the neighborough by the number of neighbour before and after
        """
        N = 3+3
        def transform(self, lNode):
            a = np.empty( ( len(lNode), self.N ) , dtype=Feat_dtype)
            for i, blk in enumerate(lNode): 
                ax1, ay1, apnum = blk.x1, blk.y1, blk.pnum
                #number of horizontal/vertical/crosspage neighbors
                a[i,0:3] = len(blk.lHNeighbor), len(blk.lVNeighbor), len(blk.lCPNeighbor)
                #number of horizontal/vertical/crosspage neighbors occuring after this block
                a[i,3:6] = (sum(1 for _b in blk.lHNeighbor  if _b.x1 > ax1), 
                            sum(1 for _b in blk.lVNeighbor  if _b.y1 > ay1), 
                            sum(1 for _b in blk.lCPNeighbor if _b.pnum > apnum))
                #better for human reading, to give direct info
                a[i,0:3] = a[i,0:3] - a[i,3:6]
            
            return a

    def __init__(self, nQuantile=None):
        self.nQuantile = Node_Neighbour_Count.nQUANTILE if nQuantile is None else nQuantile
        Pipeline.__init__(self, [
                             ('count'    , Node_Neighbour_Count.Selector()),
                             ('quantiled', QuantileTransformer(n_quantiles=self.nQuantile, copy=False))  #use in-place scaling
                         ])

    def __str__(self):
        return "- Neighbour_Count %s (#%d, quantile=%d)" % (self.__class__, Node_Neighbour_Count.Selector.N, self.nQuantile)


# -----------------------------------------------------------------------------
class Edge_1(Transformer):
    """
    a constant 1 (for CRF)
    
    """
    def transform(self, lEdge):
        return np.ones( ( len(lEdge), 1), dtype=Feat_dtype)

    def __str__(self):
        return "- Constant 1 %s" % (self.__class__)

# -----------------------------------------------------------------------------
class Edge_Type_1Hot(Transformer):
    """
    a 1-hot encoding of the edge type
    """
    def __init__(self, lEdgeClass=None):
        global lDEFAULT_EDGE_CLASS
        Transformer.__init__(self)
        if lEdgeClass is None: 
            lEdgeClass = lDEFAULT_EDGE_CLASS
        self.nbClass = len(lEdgeClass)
        self._d        = {cls:i for i,cls in enumerate(lEdgeClass)}
            
    def transform(self, lEdge):
        nb_edge = len(lEdge)
        a = np.zeros((nb_edge, self.nbClass), dtype=Feat_dtype)
        a[np.arange(nb_edge), [self._d[e.__class__] for e in lEdge]] = 1
        return a

    def __str__(self):
        return "- Type_1Hot %s (%d edge types)" % (self.__class__, self.nbClass)


# -----------------------------------------------------------------------------
class EdgeClassShifter(Transformer):
    """
    This transformer generates one range of feature per class of edge.
    
    It depends on:
    - the use of Edge_Type_1Hot being placed as first transformer of a UeatureUnion
    - indicating the same number of classes as for the 1-HOT
    """
    def __init__(self, nbClass):
        Transformer.__init__(self)
        self.nbClass = nbClass
        
    def transform(self, o):
        n, nf = o.shape
        
        nf_1hot = self.nbClass
        nf_other = nf - self.nbClass
        
        new_nf = nf_1hot + nf_other * self.nbClass
        a = np.zeros( ( n, new_nf), dtype=o.dtype)
        
        # preserve the 1-hot part
        a[:,0:nf_1hot] = o[:,0:nf_1hot]
        
        # class indices from 1-hot encoding
        i = np.argmax(o[:,0:nf_1hot], axis=1)

        # start position per edge as a fonction of its class
        start_pos = nf_1hot + i * nf_other
        end_pos   = start_pos + nf_other
        #a[np.arange(n), start_pos:end_pos] = o
        
        for k in range(n):
            a[k, start_pos[k]:end_pos[k]] = o[k, nf_1hot:]

        return a

    def __str__(self):
        return "- Feature shifter by class %s (%d edge types: # -> %d+(#-%d)x%d)" % (self.__class__
                                                                                     , self.nbClass, self.nbClass, self.nbClass, self.nbClass
                                                                                    )

# -----------------------------------------------------------------------------
class Edge_Geometry(Pipeline):
    """
    Edge geometry feature quantiled
    """
    nQUANTILE = 32

    class Selector(Transformer):
        """
        features about the geometry of the edge, or of the area between the two blocks.
        
        a range of features is dedicated for each class of edges (depends on bMultiPage).
        (Vertical, horizontal, cross-page, ...)
        """    
        N = 21
        def transform(self, lEdge):
            a = np.zeros( ( len(lEdge), self.N ) , dtype=Feat_dtype)
            for i, edge in enumerate(lEdge):
                A,B = edge.A, edge.B        
    
                # length
                l = edge.length
                l_nv = l / float(edge.A.page.h) 
                l_nh = l / float(edge.A.page.w) # normalized horizontally
    
                # overlap due to masking
                ovrl = edge.overlap
                # overlap ignoring masking
                # ovrl_max = edge.computeOverlap()                # new 9/8/19
                ovrl_max, pA, pB = edge.computeOverlapPosition()  # new 8/8/19
                r_ovrl = ovrl / (0.001+ovrl_max) # avoid zero div.
                
                # IoU
                iou = edge.iou
                
                # "blank space" between them
                # note: it is truely blank only in graph mode 2
                space = ovrl * l 
                
                a[i, :] = (  l          , l*l
                           , l_nh       , l_nh*l_nh
                           , l_nv       , l_nv*l_nv
                           , ovrl       , ovrl*ovrl
                           , ovrl_max   , ovrl_max*ovrl_max # new 8/8/19
                           , iou        , iou*iou
                           , r_ovrl     , r_ovrl*r_ovrl     # new 8/8/19
                           , space      , space*space
                           , r_ovrl / max(l,1)              # new 8/8/19
                           , pA         , pA*pA             # new 9/8/19
                           , pB         , pB*pB             # new 9/8/19
                           )
                    
            return a  

    def __init__(self, nQuantile=None):
        self.nQuantile = Edge_Geometry.nQUANTILE if nQuantile is None else nQuantile
        Pipeline.__init__(self, [
                             ('geometry' , Edge_Geometry.Selector()),
                             ('quantiled', QuantileTransformer(n_quantiles=self.nQuantile, copy=False))  #use in-place scaling
                         ])

    def __str__(self):
        return "- Geometry %s (#%d, quantile=%d)" % (self.__class__, Edge_Geometry.Selector.N, self.nQuantile)


# -----------------------------------------------------------------------------
class Edge_BB(Pipeline):
    """
    Edge BB  feature quantiled
    intersection between BB(edge.A,edlge.B) and other nodes
    """
    nQUANTILE = 32

    class Selector(Transformer):
        """
        features about the geometry of the edge, or of the area between the two blocks.
        
        a range of features is dedicated for each class of edges (depends on bMultiPage).
        (Vertical, horizontal, cross-page, ...)
        """    
        N = 1
        def transform(self, lEdge):
            # get nodes
            lpNodes=[]
            salready=set()
            ## no time to find a better way right now!
            for e in lEdge:
                for A in [e.A, e.B]:
                    if A not in salready:  
                        # might fail...  (e.g. in 2021, on cTDaR trn data for GLD2021)
                        # lpNodes.append(ShapeLoader.node_to_Polygon(e.A.node) )
                        # Safer, simpler:
                        x1, y1, x2, y2 = A.x1, A.y1, A.x2, A.y2
                        lpNodes.append(Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)]))
                        salready.add(A)
            a = np.zeros( ( len(lEdge), self.N ) , dtype=Feat_dtype)
            for i, edge in enumerate(lEdge):
                A,B = edge.A, edge.B        
    
                x, y, w, h = edge.computeBB()
#                 print (x,y,h,w)
                bbploy = Polygon([[x,y],[x+w,y],[x+w,y+h],[x,y+h],[x,y]])
#                 print ("A",A.getBB())
#                 print ("B",B.getBB())
                #print (A.x1,A.y1,A.x2,A.y2, B.x1,B.y1,B.x2,B.y2)
#                 print ([x,y],[x+w,y],[x+w,y+h],[x,y+h])
                bbploy=prep(bbploy)
                nbCuts = 0
                for i,p in enumerate(lpNodes): 
                    if bbploy.intersects(p):
                        nbCuts+=1 
#                         print ('\t',lfoo[i].text,p,lfoo[i].x1,lfoo[i].x2,lfoo[i].y1,lfoo[i].y2)
                a[i, :] = (nbCuts-2 # A and B are included!
                           )
#                 print(f'{A.text} {B.text}   {A.node.get("id")}, {B.node.get("id")},{nbCuts}')

                              
            return a  

    def __init__(self, nQuantile=None):
        self.nQuantile = Edge_BB.nQUANTILE if nQuantile is None else nQuantile
        Pipeline.__init__(self, [
                             ('BB' , Edge_BB.Selector()),
                             ('quantiled', QuantileTransformer(n_quantiles=self.nQuantile, copy=False))  #use in-place scaling
                         ])

    def __str__(self):
        return "- Geometry %s (#%d, quantile=%d)" % (self.__class__, Edge_BB.Selector.N, self.nQuantile)


# -----------------------------------------------------------------------------
class Edge_Source_Text_NGram(Pipeline):
    """
    Edge source and target text
    """
    class Selector(Transformer):
        """
        TFIDF selection of ngrams of characters
        """
        def transform(self, lEdge):
            return map(lambda x: "¤%s¤"%x.A.text, lEdge) #start/end characters

    def __init__(self
                 , analyzer   = 'char'  # analyzer : string, {'word', 'char', 'char_wb'} or callable
                 , n_tfidf    = 250     # how many features
                 , t_ngrams   = (2,3)   # tuple for min, max ngram size, e.g. (2,6)
                 , b_tfidf_lc = False   # boolean, lowercase or not?
                 ):
        
        # we keep a pointer to the TFIDF vectorizer to be able to clean it
        # we can also use it separately from the pipeline once fitted
        self.tdidfSourceText = TfidfVectorizer(lowercase     = b_tfidf_lc
                                              , max_features = n_tfidf
                                              , analyzer     = analyzer
                                              , ngram_range  = t_ngrams #(2,6)
                                              , dtype        = Feat_dtype)

        Pipeline.__init__(self, [
                     ('src'       , Edge_Source_Text_NGram.Selector())
                   , ('src_tfidf' , self.tdidfSourceText) 
                   , ('todense'   , SparseToDense())  #pystruct needs an array, not a sparse matrix
               ])

    def __str__(self):
        return "- Source NGram %s <<%s>>" % (self.__class__, self.tdidfSourceText)

    def cleanTransformers(self):
        """
        some transformers benefit from being cleaned before saved on disk...
        """
        # the TFIDF transformers are keeping the stop words => huge pickled file!!!
        self.tdidfSourceText.stop_words_ = None


# -----------------------------------------------------------------------------
class Edge_Target_Text_NGram(Pipeline):
    """
    Edge source and target text
    """
    class Selector(Transformer):
        """
        TFIDF selection of ngrams of characters
        """
        def transform(self, lEdge):
            return map(lambda x: "¤%s¤"%x.B.text, lEdge) #start/end characters
        
    def __init__(self
                 , analyzer   = 'char'  # analyzer : string, {'word', 'char', 'char_wb'} or callable
                 , n_tfidf    = 250     # how many features
                 , t_ngrams   = (2,3)   # tuple for min, max ngram size, e.g. (2,6)
                 , b_tfidf_lc = False   # boolean, lowercase or not?
                 ):
        
        # we keep a pointer to the TFIDF vectorizer to be able to clean it
        # we can also use it separately from the pipeline once fitted
        self.tdidfTargetText = TfidfVectorizer(lowercase     = b_tfidf_lc
                                              , max_features = n_tfidf
                                              , analyzer     = analyzer
                                              , ngram_range  = t_ngrams #(2,6)
                                              , dtype        = Feat_dtype)

        Pipeline.__init__(self, [
                     ('tgt'       , Edge_Target_Text_NGram.Selector())
                   , ('tgt_tfidf' , self.tdidfTargetText) 
                   , ('todense'   , SparseToDense())  #pystruct needs an array, not a sparse matrix
               ])

    def __str__(self):
        return "- Target NGram %s <<%s>>" % (self.__class__, self.tdidfTargetText)

    def cleanTransformers(self):
        """
        some transformers benefit from being cleaned before saved on disk...
        """
        # the TFIDF transformers are keeping the stop words => huge pickled file!!!
        self.tdidfTargetText.stop_words_ = None


# ----------- AUTO-TESTS --------------------------------------------------
def test_Edge_Type_1Hot():
    
    trEdge1Hot = Edge_Type_1Hot([HorizontalEdge, VerticalEdge, CrossPageEdge, CrossMirrorPageEdge])
    trEdge1Hot.fit(None, None)
    
    e_1_2 = HorizontalEdge(10, 20, 10)
    
    a = trEdge1Hot.transform([e_1_2])
    print(a)
    assert (a == np.array([[1,0,0,0]
                           ])).all()

    e_1_11 = VerticalEdge(1, 11, 99)
    a = trEdge1Hot.transform([e_1_11, e_1_2])
    print(a)
    assert (a == np.array([ [0,1,0,0]
                           ,[1,0,0,0]
                           ])).all()
                           
    return trEdge1Hot, [e_1_11, e_1_2]

def test_Edge_Type_1Hot_reduced():
    
    trEdge1Hot = Edge_Type_1Hot([HorizontalEdge, VerticalEdge, CrossPageEdge])
    trEdge1Hot.fit(None, None)
    
    e_1_2 = HorizontalEdge(10, 20, 10)
    
    a = trEdge1Hot.transform([e_1_2])
    print(a)
    assert (a == np.array([[1,0,0]
                           ])).all()

    e_1_11 = VerticalEdge(1, 11, 99)
    a = trEdge1Hot.transform([e_1_11, e_1_2])
    print(a)
    assert (a == np.array([ [0,1,0]
                           ,[1,0,0]
                           ])).all()
                           
    return trEdge1Hot, [e_1_11, e_1_2]
    
    
def test_EdgeClassShifter_reduced():
    
    trEdge1Hot, lEdge = test_Edge_Type_1Hot_reduced()
    a = trEdge1Hot.transform(lEdge)
    
    trEdgeShifter = EdgeClassShifter(3)
    trEdgeShifter.fit(None, None)
    
    aa = trEdgeShifter.transform(a)
    print("test_EdgeClassShifter_reduced ", aa)
    assert (aa == np.array([[0,1,0]
                           ,[1,0,0]
                           ])).all()
    
def test_EdgeClassShifter():
    
    trEdge1Hot, lEdge = test_Edge_Type_1Hot()
    a = trEdge1Hot.transform(lEdge)
    
    trEdgeShifter = EdgeClassShifter(4)
    trEdgeShifter.fit(None, None)
    
    aa = trEdgeShifter.transform(a)
    print("test_EdgeClassShifter  ", aa)
    assert (aa == np.array([[0,1,0,0]
                           ,[1,0,0,0]
                           ])).all()

def test_Edge_Geometry(EdgeClass=VerticalEdge):
    import numpy.testing
    class Node: pass
    e = EdgeClass(Node(), Node(), 10)
    e.A.x1, e.A.x2, e.A.y1, e.A.y2 =  0,  10,  0,  10
    e.B.x1, e.B.x2, e.B.y1, e.B.y2 = 10, 110, 10, 110
    e.A.page = Node()
    e.A.page.h = 100
    e.A.page.w = 100
    
    trGeom = Edge_Geometry.Selector()
    trGeom.fit(None, None)
    a = trGeom.transform([e])
    #print(a)
    aref = np.array([[ 10, 100 # length
                      , 0.1, 0.1,  0.01,  0.01  #normalized length
                      ,  0,0,0,0    # overlap
                      , -1,-1,1,1 # slope
                      ]])
    numpy.testing.assert_almost_equal(a, aref) #, decimal=4)
    return [e], a

def test_FeatureUnion(EdgeClass=VerticalEdge):
    import numpy.testing
    le, a = test_Edge_Geometry(EdgeClass=EdgeClass)
    lEdgeClass = [HorizontalEdge, VerticalEdge, CrossPageEdge]
    fu =  FeatureUnion([ 
              ('1hot'   , Edge_Type_1Hot(lEdgeClass=lEdgeClass))
            , ('1'      , Edge_1())
            , ('geom'   , Edge_Geometry.Selector())
        ])
    fu.fit(None)
    
    b = fu.transform(le)
    
    feat_1hot = np.zeros((1,len(lEdgeClass)))
    feat_1hot[0, lEdgeClass.index(EdgeClass)] = 1
    
    bref = np.hstack([feat_1hot, np.ones((1,1)), a])
    numpy.testing.assert_almost_equal(b, bref)
    
    if EdgeClass == HorizontalEdge:
        brefH = np.hstack([np.array([[1,0,0]]), np.ones((1,1)), a])
        numpy.testing.assert_almost_equal(b, brefH)
    if EdgeClass == VerticalEdge:
        brefV = np.hstack([np.array([[0,1,0]]), np.ones((1,1)), a])
        numpy.testing.assert_almost_equal(b, brefV)
        
    return le, b, lEdgeClass
    
    
def test_Pipeline_of_EdgeClassShifter():
    import numpy.testing
    le, b, lEdgeClass = test_FeatureUnion()
    fu =  FeatureUnion([ 
              ('1hot'   , Edge_Type_1Hot(lEdgeClass))
            , ('1'      , Edge_1())
            , ('geom'   , Edge_Geometry.Selector())
        ])
    
    ppl = Pipeline([('fu', fu)
                    , ('shifter', EdgeClassShifter(len(lEdgeClass)))])
    ppl.fit([])
    
    c = ppl.transform(le)
    
    nbClass = len(lEdgeClass)
    
    zeros = np.zeros((1,18-nbClass))
    
    # 1 hot, zeros, other features, zeros
    cref = np.hstack([b[:,:nbClass], zeros, b[:,nbClass:], zeros])

    # print(c)
    # print(cref)
    
    numpy.testing.assert_almost_equal(c, cref)
    

    leH, bH, lEdgeClass2 = test_FeatureUnion(EdgeClass=HorizontalEdge)
    assert lEdgeClass == lEdgeClass2
    
    # apart 1hot the feature must be same 
    numpy.testing.assert_almost_equal(b[:,nbClass:], bH[:,nbClass:])
    
    
    # Two edges!!!  :-O
    le = le + leH
    d = ppl.transform(le)
    
    # the 1hot part
    numpy.testing.assert_almost_equal(d[:,:nbClass]
                                      , np.array([  [0,1,0]
                                                  , [1,0,0]]))
    # the "features" part
    d_feat_ref = np.vstack([
          np.hstack([zeros          , b[:,nbClass:] , zeros])
        , np.hstack([b[:,nbClass:]  , zeros         , zeros])
        ])
    numpy.testing.assert_almost_equal(d[:,nbClass:], d_feat_ref)
    
    return le, c
         
    
# if __name__ == "__main__":
#     test_Edge_Type_1Hot()
#     test_EdgeClassShifter()
#     test_Edge_Geometry()
#     #TODO : test a complete PipeLine...
#     
