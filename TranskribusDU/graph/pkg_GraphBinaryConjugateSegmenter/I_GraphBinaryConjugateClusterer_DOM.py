# -*- coding: utf-8 -*-

"""
    Interface: Clustering function for Conjugate graph  for DOM data

    Structured machine learning, currently using graph-CRF or Edge Convolution Network

    Copyright NAVER(C) 2019 JL. Meunier

    Developed  for the EU project READ. The READ project has received funding 
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""
import lxml.etree as etree
from util.Shape import ShapeLoader

from xml_formats.PageXml import PageXml
from graph.pkg_GraphBinaryConjugateSegmenter.I_GraphBinaryConjugateClusterer import I_GraphBinaryConjugateClusterer


class I_GraphBinaryConjugateClusterer_DOM(I_GraphBinaryConjugateClusterer):
    """
    What is independent from the input format (XML or JSON, currently)
    """

    def __init__(self, sOuputXmlAttribute=None):
        """
        a CRF model, with a name and a folder where it will be stored or retrieved from
        
        the cluster index of each object with be store in an Xml attribute of given name
        """
        I_GraphBinaryConjugateClusterer.__init__(self, sOuputXmlAttribute)
        
    def addClusterToDoc(self, lCluster, sAlgo=None):
        """
        DOM version
        """
        for num, lNodeIdx in enumerate(lCluster):
            for ndIdx in lNodeIdx:
                node = self.lNode[ndIdx]
                node.node.set(self.sOutputAttribute, "%d"%num)

        self.addClusterToDom(lCluster, sAlgo=lCluster.sAlgo if sAlgo is None else sAlgo)            
        return

    def addClusterToDom(self, lCluster, bMoveContent=False, sAlgo="", pageNode=None):
        """
        Add Cluster elements to the Page DOM node
        """
        lNdCluster = []
        
        if pageNode is None and self.lNode:
            pageNode = self.lNode[0].page.node
        if lCluster:
            assert pageNode is not None, "no PAGE DOM node??"
            pageNode.append(etree.Comment("\nClusters created by the conjugate graph\n"))
        
        for name, lnidx in enumerate(lCluster):    
            #self.analysedCluster()                             
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
