# -*- coding: utf-8 -*-

"""
    Computing the graph for a document
    

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
import collections

import numpy as np
import libxml2

from common.trace import traceln

import Edge

class Graph:
    """
    A graph to be used as a CRF graph with pystruct
    """
    
    # --- NODE TYPES and LABELS
    _lNodeType       = []       #the list of node types for this class of graph
    _dLabelByCls     = None     #dictionary across node types
    _dClsByLabel     = None     #dictionary across node types
    
    #--- CONSTRAINTS
    _lPageConstraintDef = None  #optionnal page-level constraints
                
    def __init__(self, lNode = [], lEdge = []):
        self.lNode = lNode
        self.lEdge = lEdge
        self.doc   = None
        
    # --- Node Types -------------------------------------------------
    @classmethod
    def getNodeTypeList(cls):
        """
        Return the list of label set
        """
        return cls._lNodeType
    
    @classmethod
    def addNodeType(cls, nodeType):
        """
        Add a new node type to this class of graph
        return the new whole list of labels
        """
        #list of labels
        lsNewLabel = nodeType.getLabelNameList()
        lsAllLabel = cls.getLabelNameList()
        
        for sNewLabel in lsNewLabel:
            if sNewLabel in lsAllLabel: 
                raise ValueError("A label must be globally unique (within and across LabelSet(s)): '%s'"%sNewLabel)
            lsAllLabel.append(sNewLabel)
            
        cls._lNodeType.append(nodeType)
        assert lsAllLabel == cls.getLabelNameList(), "Internal error"
        
        #and make convenience data structures
        cls._dLabelByCls = { i:sLabel for i,sLabel in enumerate(lsAllLabel) }         
        cls._dClsByLabel = { sLabel:i for i,sLabel in enumerate(lsAllLabel) } 
        
        return lsAllLabel
    
    # --- Labels ----------------------------------------------------------
    @classmethod
    def getLabelNameList(cls):
        """
        Return the list of label names for all label sets
        """
        return [sLabelName for lblSet in cls._lNodeType for sLabelName in lblSet.getLabelNameList()]

    def parseDomLabels(self):
        """
        Parse the label of the graph from the dataset, and set the node label
        return the set of observed class (set of integers in N+)
        """
        setSeensLabels = set()
        for nd in self.lNode:
            nodeType = nd.type 
            #a LabelSet object knows how to parse a DOM node of a Graph object!!
            sLabel = nodeType.parseDomNodeLabel(nd.node)
            try:
                cls = self._dClsByLabel[sLabel]  #Here, if a node is not labelled, and no default label is set, then KeyError!!!
            except KeyError:
                raise ValueError("Page %d, unknown label '%s' in %s"%(nd.pnum, sLabel, str(nd.node)))
            nd.cls = cls
            setSeensLabels.add(cls)
        return setSeensLabels    

    def setDomLabels(self, Y):
        """
        Set the labels of the graph nodes from the Y matrix
        return the DOM
        """
        for i,nd in enumerate(self.lNode):
            sLabel = self._dLabelByCls[ Y[i] ]
            nd.type.setDomNodeLabel(nd.node, sLabel)
        return self.doc

    # --- Constraints -----------------------------------------------------------
    def setPageConstraint(cls, lPageConstraintDef):
        """
        We get the definition of the constraint per page
        The constraints must be a list of tuples like ( <operator>, <label>, <negated> )
            where:
            - operator is one of 'XOR' 'XOROUT' 'ATMOSTONE' 'OR' 'OROUT' 'ANDOUT' 'IMPLY'
            - nodeType is the NodeType, which defines the labels
            - states is a list of unary state names, 1 per involved unary. If the states are all the same, you can pass it directly as a single string.
            - negated is a list of boolean indicated if the unary must be negated. Again, if all values are the same, pass a single boolean value instead of a list 
        """
        cls._lPageConstraintDef = [ (op, nt.getInternalLabelName(label), neg) for (op, nt, label, neg) in lPageConstraintDef]
    setPageConstraint = classmethod(setPageConstraint)
    
    def getPageConstraint(cls):
        return cls._lPageConstraintDef
    getPageConstraint = classmethod(getPageConstraint)

    def instanciatePageConstraints(self):
        """
        Instanciate for this particular graph the defined constraints
        return a list of tuples like ( <operator>, <unaries>, <states>, <negated> )
            where:
            - operator is one of 'XOR' 'XOROUT' 'ATMOSTONE' 'OR' 'OROUT' 'ANDOUT' 'IMPLY'
            - unaries is a list of the index of the unaries involved in this constraint
            - states is a list of unary states, 1 per involved unary. If the states are all the same, you can pass it directly as a scalar value.
            - negated is a list of boolean indicated if the unary must be negated. Again, if all values are the same, pass a single boolean value instead of a list 
        """
        lUnaries = self.getNodeIndexByPage()
        lRet = [ (op, unaries, self._dClsByLabel[label], neg) for unaries in lUnaries for (op, label, neg) in self._lPageConstraintDef]
#         for (op, unaries, cls, neg) in lRet:
#             print [self.lNode[i].domid for i in unaries]
        return lRet
    
    # --- Graph building --------------------------------------------------------
    @classmethod
    def loadGraphs(cls, lsFilename, bNeighbourhood=True, bDetach=False, bLabelled=False, iVerbose=0):
        """
        Load one graph per file, and detach its DOM
        return the list of loaded graphs
        """
        lGraph = []
        for sFilename in lsFilename:
            if iVerbose: traceln("\t%s"%sFilename)
            g = cls()
            g.parseXmlFile(sFilename, iVerbose)
            if bNeighbourhood: g.collectNeighbors()            
            if bLabelled: g.parseDomLabels()
            if bDetach: g.detachFromDOM()
            lGraph.append(g)
        return lGraph

    def parseXmlFile(self, sFilename, iVerbose=0):
        """
        Load that document as a CRF Graph.
        Also set the self.doc variable!
        
        Return a CRF Graph object
        """
    
        self.doc = libxml2.parseFile(sFilename)
        self.lNode, self.lEdge = list(), list()
        #load the block of each page, keeping the list of blocks of previous page
        lPrevPageNode = None

        for pnum, page, domNdPage in self._iter_Page_DomNode(self.doc):
            #now that we have the page, let's create the node for each type!
            lPageNode = list()
            setPageNdDomId = set() #the set of DOM id
            # because the node types are supposed to have an empty intersection
                            
            lPageNode = [nd for nodeType in self.getNodeTypeList() for nd in nodeType._iter_GraphNode(self.doc, domNdPage, page) ]
            
            #check that each node appears once
            setPageNdDomId = set([nd.domid for nd in lPageNode])
            assert len(setPageNdDomId) == len(lPageNode), "ERROR: some nodes fit with multiple NodeTypes"
            
        
            self.lNode.extend(lPageNode)
            
            lPageEdge = Edge.Edge.computeEdges(lPrevPageNode, lPageNode)
            
            self.lEdge.extend(lPageEdge)
            if iVerbose>=2: traceln("\tPage %5d    %6d nodes    %7d edges"%(pnum, len(lPageNode), len(lPageEdge)))
            
            lPrevPageNode = lPageNode
        if iVerbose: traceln("\t- %d nodes,  %d edges)"%(len(self.lNode), len(self.lEdge)) )
        
        return self

    def _iter_Page_DomNode(self, doc):
        """
        Parse a Xml DOM, by page

        iterator on the DOM, that returns per page:
            page-num (int), page object, page dom node
        
        """
        raise Exception("Must be specialized")
    
    def collectNeighbors(self):
        """
        record the lists of hotizontal-, vertical- and cross-page neighbours for each node
        """
        for blk in self.lNode:
            blk.lHNeighbor = list()
            blk.lVNeighbor = list()
            blk.lCPNeighbor = list()        
        for edge in self.lEdge:
            a, b = edge.A, edge.B
            if isinstance(edge, Edge.CrossPageEdge):
                a.lCPNeighbor.append(b)
                b.lCPNeighbor.append(a)
            elif isinstance(edge, Edge.HorizontalEdge):
                a.lHNeighbor.append(b)
                b.lHNeighbor.append(a)
            else:
                assert  isinstance(edge, Edge.VerticalEdge)
                a.lVNeighbor.append(b)
                b.lVNeighbor.append(a)
    
    def detachFromDOM(self):
        """
        Detach the graph from the DOM node, which can then be freed
        """
        for nd in self.lNode: nd.detachFromDOM()
        self.doc.freeDoc()
        self.doc = None


    # --- Numpy matrices --------------------------------------------------------
    def buildNodeEdgeMatrices(self, node_transformer, edge_transformer):
        """
        make 1 node-feature matrix
         and 1 edge-feature matrix
         and 1 edge matrix
         for the graph
        return 3 Numpy matrices
        """
        node_features = node_transformer.transform(self.lNode)
        edges = self._indexNodes_and_BuildEdgeMatrix()
        edge_features = edge_transformer.transform(self.lEdge)
        return (node_features, edges, edge_features)       
    
    def buildLabelMatrix(self):
        """
        Return the matrix of labels
        """
        Y = np.array( [nd.cls for nd in self.lNode] , dtype=np.uint8)
        return Y
    
    def _indexNodes_and_BuildEdgeMatrix(self):
        """
        - add an index attribute to nodes
        - build an edge matrix on this basis
        - return the edge matrix (a 2-columns matrix)
        """
        for i, nd in enumerate(self.lNode):
            nd.index = i

        edges = np.empty( (len(self.lEdge), 2) , dtype=np.int32)
        for i, edge in enumerate(self.lEdge):
            edges[i,0] = edge.A.index
            edges[i,1] = edge.B.index
        return edges

    def getNodeIndexByPage(self):
        """
        return a list of list of index
        Both lists are sorted (by page number and by index)
        empty pages are skipped (and _not_ reflected as empty list)
        """
        if not self.lNode: raise ValueError("Empty graph")
        try:
            self.lNode[0].index
        except AttributeError:
            for i, nd in enumerate(self.lNode):
                nd.index = i

        dlIndexByPage = collections.defaultdict(list)
        for nd in self.lNode:
            dlIndexByPage[nd.pnum].append(nd.index)
        
        llIndexByPage = []
        for pnum in sorted(dlIndexByPage.keys()):
            llIndexByPage.append( sorted(dlIndexByPage[pnum]) )
        return llIndexByPage
            
        
        
        

        
        
