# -*- coding: utf-8 -*-

"""
    Train, test, predict steps for a graph-based model using a binary conjugate 
    (two classes on the primal edges)

    Structured machine learning, currently using graph-CRF or Edge Convolution Network

    Copyright NAVER(C) 2019 JL. Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""
import lxml.etree as etree

from xml_formats.PageXml import PageXml 

from graph.Graph_DOM import Graph_DOM
from .GraphBinaryConjugateSegmenter import GraphBinaryConjugateSegmenter

from .I_GraphBinaryConjugateClusterer_DOM import I_GraphBinaryConjugateClusterer_DOM


class GraphBinaryConjugateSegmenter_DOM(I_GraphBinaryConjugateClusterer_DOM, GraphBinaryConjugateSegmenter, Graph_DOM):
    # --- NODE TYPES and LABELS
    _lNodeType       = []       #the list of node types for this class of graph
    _bMultitype      = False    # equivalent to len(_lNodeType) > 1
    _dLabelByCls     = None     #dictionary across node types
    _dClsByLabel     = None     #dictionary across node types
    _nbLabelTot      = 0        #total number of labels
    
    def __init__(self, lNode = [], lEdge = [], sOuputXmlAttribute=None):
        I_GraphBinaryConjugateClusterer_DOM.__init__(self, sOuputXmlAttribute=sOuputXmlAttribute)
        GraphBinaryConjugateSegmenter.__init__(self)
        Graph_DOM.__init__(self, lNode, lEdge)
        

    def addEdgeToDoc(self, ndPage=None):
        """
        To display the graph conveniently we add new Edge elements
        Exact same code as in Graph_DOM.py except that we add the GT value is possible
        """
        if self.lNode:
            ndPage = self.lNode[0].page.node if ndPage is None else ndPage    
            ndPage.append(etree.Comment("Edges added to the XML for convenience"))
            for edge in self.lEdge:
                A, B = edge.A , edge.B   #shape.centroid, edge.B.shape.centroid
                ndEdge = PageXml.createPageXmlNode("Edge")
                if bool(edge.A.domid):
                    # edge between reified edges  do not have a domid
                    ndEdge.set("src", edge.A.domid)
                    ndEdge.set("tgt", edge.B.domid)
                ndEdge.set("type", edge.__class__.__name__)

                # in case the edge has a predicted label
                if hasattr(edge, "cls"):
                    if len(edge.cls) > 1:
                        cls = edge.cls.argmax()
                        ndEdge.set("proba", "%.3f" % edge.cls[cls])
                        ndEdge.set("distr", str(edge.cls.tolist()))
                    else:
                        cls = edge.cls
                    # ndEdge.set("label", self.lEdgeLabel[cls])
                    ndEdge.set("label", self.getLabelNameList()[cls])
                    ndEdge.set("label_cls", str(cls))
                    
                    try:
                        lblA = str(A.type.parseDocNodeLabel(A))
                        lblB = str(B.type.parseDocNodeLabel(B))
                        ndEdge.set("gt", "continue" if lblA == lblB else "break")
                    except:
                        pass
                    
                ndEdge.tail = "\n"
                ndPage.append(ndEdge)
#                 PageXml.setPoints(ndEdge, [(A.x1, A.y1), (B.x1, B.y1)]) 
                o = edge.getCoords()
                if o is not None:
                    x1, y1, x2, y2 = o
                    PageXml.setPoints(ndEdge, [(x1, y1), (x2, y2)])                 
        
        return 