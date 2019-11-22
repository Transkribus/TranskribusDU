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
from util.Shape import ShapeLoader

from graph.Graph_DOM import Graph_DOM
from .GraphBinaryConjugateSegmenter import GraphBinaryConjugateSegmenter



class GraphBinaryConjugateSegmenter_DOM(GraphBinaryConjugateSegmenter, Graph_DOM):
    # --- NODE TYPES and LABELS
    _lNodeType       = []       #the list of node types for this class of graph
    _bMultitype      = False    # equivalent to len(_lNodeType) > 1
    _dLabelByCls     = None     #dictionary across node types
    _dClsByLabel     = None     #dictionary across node types
    _nbLabelTot      = 0        #total number of labels
    
    def __init__(self, lNode = [], lEdge = [], sOuputXmlAttribute=None):
        GraphBinaryConjugateSegmenter.__init__(self, sOuputXmlAttribute=sOuputXmlAttribute)
        Graph_DOM.__init__(self, lNode, lEdge)
        
    def addClusterToDoc(self, lCluster):
        """
        DOM version
        """
        for num, lNodeIdx in enumerate(lCluster):
            for ndIdx in lNodeIdx:
                node = self.lNode[ndIdx]
                node.node.set(self.sOutputAttribute, "%d"%num)

        self.addClusterToDom(lCluster, sAlgo=lCluster.sAlgo)            
        return

    def addEdgeToDoc(self, Y_proba):
        """
        To display the graph conveniently we add new Edge elements
        
        # for y_p, x_u, in zip(lY_pred, [X]):
            # edges = x_u[1][:int(len(x_u[1])/2)]
            # for i, (p,ie)  in enumerate(zip(y_p, edges)):     
                # print(p,  g.lNode[ie[0]].text,g.lNode[ie[1]].text, g.lEdge[i])
        """
        if self.lNode:
            ndPage = self.lNode[0].page.node
            ndPage.append(etree.Comment("\nEdges labeled by the conjugate graph\n"))
            Y = Y_proba.argmax(axis=1)
            for i, edge in enumerate(self.lEdge):
                A, B = edge.A ,edge.B   #shape.centroid, edge.B.shape.centroid
                ndEdge = PageXml.createPageXmlNode("Edge")
                try:
                    cls = Y[i]
                    ndEdge.set("label", self.lEdgeLabel[cls])
                    ndEdge.set("proba", "%.3f" % Y_proba[i, cls])
                except IndexError:
                    # case of a conjugate graph without edge, so the edges 
                    # of the original graph cannot be labelled
                    pass
                ndEdge.set("src", edge.A.node.get("id"))
                ndEdge.set("tgt", edge.B.node.get("id"))
                ndEdge.set("type", edge.__class__.__name__)
                ndPage.append(ndEdge)
                ndEdge.tail = "\n"
                PageXml.setPoints(ndEdge, [(A.x1, A.y1), (B.x1, B.y1)]) 
                           
        return         

    def addClusterToDom(self, lCluster, bMoveContent=False, sAlgo="", pageNode=None):
        """
        Add Cluster elements to the Page DOM node
        """
        lNdCluster = []
        for name, lnidx in enumerate(lCluster):    
            #self.analysedCluster()                             
            if pageNode is None:
                for idx in lnidx:
                    pageNode = self.lNode[idx].page.node
                    break    
                pageNode.append(etree.Comment("\nClusters created by the conjugate graph\n"))

            ndCluster = PageXml.createPageXmlNode('Cluster')  
            ndCluster.set("name", str(name))   
            ndCluster.set("algo", sAlgo)   
            # add the space separated list of node ids
            ndCluster.set("content", " ".join(self.lNode[_i].node.get("id") for _i in lnidx))   
            coords = PageXml.createPageXmlNode('Coords')        
            ndCluster.append(coords)
            spoints = ShapeLoader.minimum_rotated_rectangle([self.lNode[_i].node for _i in lnidx])
            coords.set('points',spoints)                     
            pageNode.append(ndCluster)
            ndCluster.tail = "\n"   
                 
            if bMoveContent:
                # move the DOM node of the content to the cluster
                for _i in lnidx:                               
                    ndCluster.append(self.lNode[_i].node)
            lNdCluster.append(ndCluster)
            
        return lNdCluster
