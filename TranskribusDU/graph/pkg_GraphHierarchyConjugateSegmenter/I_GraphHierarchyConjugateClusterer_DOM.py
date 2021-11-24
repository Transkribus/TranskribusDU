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
from .I_GraphHierarchyConjugateClusterer import I_GraphHierarchyConjugateClusterer


class I_GraphHierarchyConjugateClusterer_DOM(I_GraphHierarchyConjugateClusterer):
    """
    Cluster by level
    """
    def __init__(self):
        I_GraphHierarchyConjugateClusterer.__init__(self)
        
    def addClusterToDoc(self, lvl, lCluster, sAlgo=None):
        """
        DOM version
        lvl starts at 0
        """
        for num, lNodeIdx in enumerate(lCluster):
            for ndIdx in lNodeIdx:
                node = self.lNode[ndIdx]
                node.node.set("DU_cluster_lvl%d" % lvl, "%d"%num)

        self.addClusterToDom(lvl, lCluster, sAlgo=lCluster.sAlgo if sAlgo is None else sAlgo)            
        return

    def addClusterToDom(self, lvl, lCluster, bMoveContent=False, sAlgo="", pageNode=None):
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
            ndCluster.set("level", str(lvl))   
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
