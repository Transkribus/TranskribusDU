# -*- coding: utf-8 -*-

"""
    Computing the graph for a MultiPageXml document

    Copyright Xerox(C) 2016 JL. Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""




from  lxml import etree

from common.trace import traceln

from ..Graph_MultiPageXml import ContinousPage
from .FactorialGraph_MultiPageXml import FactorialGraph_MultiPageXml

from ..Transformer_PageXml import  EdgeTransformerClassShifter
from ..Edge import Edge, HorizontalEdge, VerticalEdge, CrossPageEdge, CrossMirrorContinuousPageVerticalEdge


class FactorialGraph_MultiContinuousPageXml(FactorialGraph_MultiPageXml, ContinousPage):
    """
    FactorialCRF for MultiPageXml document
    
    we also compute edges between continuous pages.
    
    FOR NOW: SIMPLE APPROXIMATE METHOD: we take N "last" block of the page and M "first" blocks of next page, and look for horizontal mirror overlap.
    
    """
    def __init__(self, lNode = [], lEdge = []):
        FactorialGraph_MultiPageXml.__init__(self, lNode, lEdge)
        
        EdgeTransformerClassShifter.setDefaultEdgeClass([HorizontalEdge, VerticalEdge, CrossPageEdge, CrossMirrorContinuousPageVerticalEdge])

    def parseDocFile(self, sFilename, iVerbose=0):
        """
        Load that document as a CRF Graph.
        Also set the self.doc variable!
        
        Return a CRF Graph object
        """
    
        self.doc = etree.parse(sFilename)
        self.lNode, self.lEdge = list(), list()
        #load the block of each page, keeping the list of blocks of previous page
        lPrevPageNode = None

        nodeType0 = self.getNodeTypeList()[0]    #all nodes have same type
        
        for pnum, page, domNdPage in self._iter_Page_DocNode(self.doc):
            #now that we have the page, let's create the node for each type!
            lPageNode = list()
            #setPageNdDomId = set() #the set of DOM id
            # because the node types are supposed to have an empty intersection
                            
            lPageNode = list(nodeType0._iter_GraphNode(self.doc, domNdPage, page))
            
#             #check that each node appears once
#             setPageNdDomId = set([nd.domid for nd in lPageNode])
#             assert len(setPageNdDomId) == len(list(lPageNode)), "ERROR: some nodes fit with multiple NodeTypes"
            
        
            self.lNode.extend(lPageNode)
            
            lPageEdge = Edge.computeEdges(lPrevPageNode, lPageNode, self.iGraphMode)
            self.lEdge.extend(lPageEdge)

            # only difference with its parent method from FactorialGraph_MultiPageXml
            lContinuousPageEdge = self.computeContinuousPageEdges(lPrevPageNode, lPageNode)
            self.lEdge.extend(lContinuousPageEdge)
            
            
            if iVerbose>=2: traceln("\tPage %5d    %6d nodes    %7d edges"%(pnum, len(lPageNode), len(lPageEdge)))
            
            lPrevPageNode = lPageNode
        if iVerbose: traceln("\t\t (%d nodes,  %d edges)"%(len(self.lNode), len(self.lEdge)) )
        
        return self
        
        