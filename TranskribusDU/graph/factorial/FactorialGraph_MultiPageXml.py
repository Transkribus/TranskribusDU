# -*- coding: utf-8 -*-

"""
    Computing the graph for a MultiPageXml document

    Copyright Xerox(C) 2016 JL. Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""




import numpy as np

from  lxml import etree

from common.trace import traceln

from ..Graph_MultiPageXml import Graph_MultiPageXml
from ..Edge import Edge

from .FactorialGraph import FactorialGraph

# ------------------------------------------------------------------------------------------------------------------------------------------------
class FactorialGraph_MultiPageXml(Graph_MultiPageXml, FactorialGraph):
    """
    FactorialCRF for MultiPageXml document
    
    We build a multitype graph, which can be seen as a layered graph. 
    A node appears in each layer, under a different type, so as to get a label in each different label space.
    
    For instance with node types * and #, any edge -- is replicated among layers, and an edge | connect the multiple replicas of each node:
    
    * -- * -- ...
    |    |
    # -- # -- ...
    
    """
    
    def __init__(self, lNode = [], lEdge = []):
        self._actual_lNodeType  = self._lNodeType
        self._lNodeType = self._lNodeType[0]
        
        Graph_MultiPageXml.__init__(self, lNode, lEdge)
        
        self._lNodeType         = self._actual_lNodeType

        if len(self._lNodeType) < 2: raise ValueError("FactorialCRF requires at least two node types.")
        nt0 = self._lNodeType[0]
        for nt in self._lNodeType[1:]: 
            if nt.__class__      != nt0.__class__     : raise ValueError("FactorialCRF requires all NodeType to be of same class.")
            if nt.getXpathExpr() != nt0.getXpathExpr(): raise ValueError("FactorialCRF requires all NodeType to have same Xpath selection expressions.")

    # --- Labels ----------------------------------------------------------
    def parseDocLabels(self):
        """
        Parse the label of the graph from the dataset, and set the node label
        return the set of observed class (set of integers in N+)
        
        == FactorialCRF ==
        Each node belongs to all NodeType and therefore has a list of .cls
        """
        setSeensLabels = set()
        for nd in self.lNode:
            try:
                lcls = [self._dClsByLabel[nodeType.parseDocNodeLabel(nd)] for nodeType in self.getNodeTypeList()]
            except KeyError:
                raise ValueError("Page %d, unknown label in %s (Known labels are %s)"%(nd.pnum, str(nd.node), self._dClsByLabel))
            nd.cls = lcls
            for cls in lcls:
                setSeensLabels.add(cls)
        return setSeensLabels    

    def setDocLabels(self, Y):
        """
        Set the labels of the graph nodes from the Y matrix
        return the DOM

        == FactorialCRF ==
        Y if a flatened matrix nodes x types
        """
        nbNode = len(self.lNode)
        zeroType = 0 
        for nodeType in self.getNodeTypeList():
            for i,nd in enumerate(self.lNode):
                sLabel = self._dLabelByCls[ Y[zeroType+i] ]
                nodeType.setDocNodeLabel(nd, sLabel)
            zeroType += nbNode
        return self.doc
    
    def parseDocFile(self, sFilename, iVerbose=0):
        """
        Load that document as a CRF Graph.
        Also set the self.doc variable!
        
        Return a CRF Graph object

        == FactorialCRF ==
        We parse the XML for one of the node types
        That enough to get the list of nodes and edges. We can then have a virtually layered graph, one layer per node type
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
            
#            #check that each node appears once
#            setPageNdDomId = set([nd.domid for nd in lPageNode])
#            assert len(setPageNdDomId) == len(lPageNode), "ERROR: some nodes fit with multiple NodeTypes"
            
        
            self.lNode.extend(lPageNode)
            
            lPageEdge = Edge.computeEdges(lPrevPageNode, lPageNode, self.iGraphMode)
            
            self.lEdge.extend(lPageEdge)
            if iVerbose>=2: traceln("\tPage %5d    %6d nodes    %7d edges"%(pnum, len(lPageNode), len(lPageEdge)))
            
            lPrevPageNode = lPageNode
        if iVerbose: traceln("\t\t (%d nodes,  %d edges)"%(len(self.lNode), len(self.lEdge)) )
        
        return self
        

    def getY(self):
        """
        WARNING, in multitype graphs, the order of the Ys is bad
        """
        return self._buildLabelMatrix_S()

    #----- SINGLE TYPE -----   
    def _buildLabelMatrix_S(self):
        """
        Return the matrix of labels        BAD ORDER!!!
        """
        #better code based on fromiter is below (I think, JLM April 2017) 
        #Y = np.array( [nd.cls for nd in self.lNode] , dtype=np.uint8)
        Y = np.fromiter( (cls for nd in self.lNode for cls in nd.cls), dtype=np.int, count=len(self.lNode))
        return Y
        
    #----- MULTITYPE -----  
    def _buildNodeEdgeLabelMatrices_T(self, node_transformer, edge_transformer, bY=True):
        """
        make a list of node feature matrices
         and a list of edge definition matrices
         and a list of edge feature matrices
         for the graph
        and optionnaly the Y, if bY is True
        return  a triplet
             or a tuple (triplet, Y)
        """
        nbType = len(self._lNodeType)
        nbNode = len(self.lNode)
        #get the node features, edges, edge features as if single type graph
        (NF, E, EF) = Graph_MultiPageXml._buildNodeEdgeMatrices_S(self, node_transformer, edge_transformer)
                                                                  
        #The NF per type are all the same
        lNF = [NF for _nodeType in self.getNodeTypeList()]
        
        #The Edge definitions and edge features
        # edge within a layer are all the same (but between the node replcias for the layer)
        # we add one edge between each pair of node replica between each pairs of layer  (only in one direction)
        lE = []
        lEF = []
        E_empty  = np.empty( (0, 2) , dtype=np.int)
        EF_empty = np.empty( (0, 1) , dtype=np.float64)
        E_replicas = np.array( [ [i, i] for i in range(nbNode)] , dtype=np.int)
        EF_ones_replicas = np.array( [ [1.0] for i in range(nbNode)] , dtype=np.float64)
        for typ1 in range(nbType):
            #bottom-left part if empty
            for i in range(typ1): 
                lE.append (E_empty) 
                lEF.append(EF_empty)   
            #edge within a layer
            lE.append (E)
            lEF.append(EF)
            #top-right part give the links between node replicas accross layers
            for _typ2 in range(typ1+1, nbType): 
                lE.append (E_replicas)
                lEF.append(EF_ones_replicas)

        if bY:
            Y = np.zeros( (len(self.lNode)*nbType,), dtype=np.int)
            
            for ityp in range(nbType):
                zeroNode = ityp*nbNode
                Y[zeroNode:zeroNode+nbNode] = [nd.cls[ityp] for nd in self.lNode]
                
        if bY:
            return (lNF, lE, lEF), Y
        else:
            return (lNF, lE, lEF)

    #----- STUFF -----  
    def getNodeIndexByPage(self):
        """
        return a list of list of index
        Both lists are sorted (by page number and by index)
        empty pages are skipped (and _not_ reflected as empty list)
        """
        raise Exception("Not implemented: getNodeIndexByPage. COnstraints not (yet) supported in factorialCRF.")
        
