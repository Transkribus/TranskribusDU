# -*- coding: utf-8 -*-

"""
    Computing the graph for a XML document
    
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
#import gc

import numpy as np
from lxml import etree

from common.trace import traceln
from xml_formats.PageXml import PageXmlException

from . import Edge
from xml_formats.PageXml import PageXml

class GraphException(Exception): pass


class Graph:
    """
    A graph to be used as a CRF graph with pystruct
    """
    # --- NODE TYPES and LABELS
    _lNodeType       = []       #the list of node types for this class of graph
    _bMultitype      = False    # equivalent to len(_lNodeType) > 1
    _dLabelByCls     = None     #dictionary across node types
    _dClsByLabel     = None     #dictionary across node types
    _nbLabelTot      = 0        #total number of labels

    iGraphMode       = 1        # how to compute edges 1=historical 2=dealing properly with line-of-sight
        
    #--- CONSTRAINTS
    _lPageConstraintDef = None  #optionnal page-level constraints

    # do we use the conjugate graph?
    bConjugate = False
                
    def __init__(self, lNode = [], lEdge = []):
        self.lNode = lNode
        self.lEdge = lEdge
        self.doc   = None
        
        
        self.aNeighborClassMask = None   #did we compute the neighbor class mask already?
        
        self.aLabelCount = None #count of seen labels
        
    # --- CONJUGATE -----------------------------------------------------
    @classmethod
    def setConjugateMode(cls
                         , lEdgeLabel        = None   # list of labels (list of strings, or of int)
                         , computeEdgeLabels = None   # to compute the edge labels
                         , exploitEdgeLabels = None   # to use the predicted edge labels
                         ):
        """
        learn and predict on the conjugate graph instead of the usual graph.
        1 - The usual graph is created as always
        2 - the function computeEdgeLabels is called to compute the edge labels
        3 - the conjugate is created and used for learning or predicting
        4 - the function exploitEdgeLabels is called once the edge labels are predicted
        
        The prototype of the functions are as shown in the code below.
        
        NOTE: since the graph class may already be dedicated to the Conjugate mode,
        we do not force to pass the list of edge labels and the 2 methods
        But bConjugate must be True already!
        """
        if cls.bConjugate is True:
            # then we accept to override some stuff...
            if not lEdgeLabel        is None: cls.lEdgeLabel          = lEdgeLabel
            if not computeEdgeLabels is None: cls.computeEdgeLabels   = computeEdgeLabels
            if not exploitEdgeLabels is None: cls.exploitEdgeLabels   = exploitEdgeLabels            
        else:
            if None in [lEdgeLabel, computeEdgeLabels, exploitEdgeLabels]:
                raise GraphException("You must provide lEdgeLabel, computeEdgeLabels, and exploitEdgeLabels")
            cls.bConjugate          = True
            cls.lEdgeLabel          = lEdgeLabel
            cls.computeEdgeLabels   = computeEdgeLabels
            cls.exploitEdgeLabels   = exploitEdgeLabels
            
        assert len(cls.lEdgeLabel) > 1, ("Invalid number of edge labels (graph conjugate mode)", lEdgeLabel)
        traceln("SETUP: Conjugate mode: %s" % str(cls))
        
        return cls.bConjugate

    @classmethod
    def getGraphMode(cls):
        return cls.iGraphMode
    @classmethod
    def setGraphMode(cls, iGraphMode):
        assert iGraphMode in (1,2)
        cls.iGraphMode = iGraphMode
        return cls.iGraphMode
    
    @classmethod
    def getEdgeLabelNameList(cls):
        """
        Return the list of label names for all label sets
        """
        return cls.lEdgeLabel

    def computeEdgeLabels(self):
        """
        Given the loaded graph with labeled nodes, compute the edge labels.
        
        This results in each edge having a .cls attribute.
    
        return the set of observed class (set of integers in N+)
        """
        raise GraphException("You must specialize this class method")

    def exploitEdgeLabels(self, Y_proba):
        """
        Do whatever is required on the (primal) graph, given the edge labels
            Y_proba is the predicted edge label probability array of shape (N_edges, N_labels)
        
        return None
        
        The node and edge indices corresponding to the order of the lNode
        and lEdge attribute of the graph object.
        
        Here we choose to set an XML attribute DU_cluster="<cluster_num>"
        """
        raise GraphException("You must specialize this class method")
        
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
        #NOTE: the cls integer value encompasses all types, from 0 to N-1 where N is the cumulative number of classes over node types
        return the new whole list of labels
        """
        if nodeType not in cls._lNodeType:  #make it idempotent
            #list of labels
            lsNewLabel = nodeType.getLabelNameList()
            lsAllLabel = cls.getLabelNameList()
            
            for sNewLabel in lsNewLabel:
                if sNewLabel in lsAllLabel: 
                    raise ValueError("A label must be globally unique (within and across LabelSet(s)): '%s'"%sNewLabel)
                lsAllLabel.append(sNewLabel)
                
            cls._lNodeType.append(nodeType)
            assert lsAllLabel == cls.getLabelNameList(), "Internal error"
            cls._nbLabelTot = len(lsAllLabel)
            
            #and make convenience data structures
            cls._dLabelByCls = { i:sLabel for i,sLabel in enumerate(lsAllLabel) }         
            cls._dClsByLabel = { sLabel:i for i,sLabel in enumerate(lsAllLabel) } 
            
            cls._bMultitype = (len(cls._lNodeType) >1) 
        
        return lsAllLabel

    @classmethod
    def resetNodeTypes(cls):
        """
        When consecutive different models are created, e;g. from pytest, there
        is the need to reset the node types declared at the class-level
        """
        cls._lNodeType       = []       #the list of node types for this class of graph
        cls._bMultitype      = False    # equivalent to len(_lNodeType) > 1
        cls._dLabelByCls     = None     #dictionary across node types
        cls._dClsByLabel     = None     #dictionary across node types
        cls._nbLabelTot      = 0        #total number of labels
    
    # --- Labels ----------------------------------------------------------
    @classmethod
    def getLabelNameList(cls):
        """
        Return the list of label names for all label sets
        """
#         if cls.bConjugate:
#             return cls.getEdgeLabelNameList()
#         else:
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
            try:
                sLabel = nodeType.parseDomNodeLabel(nd.node)
            except PageXmlException:
                sLabel='TR_OTHER'
            try:
                cls = self._dClsByLabel[sLabel]  #Here, if a node is not labelled, and no default label is set, then KeyError!!!
            except KeyError:
                raise ValueError("Page %d, unknown label '%s' in %s (Known labels are %s)"%(nd.pnum, sLabel, str(nd.node), self._dClsByLabel))
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
    def getNodeEdgeTotalNumber(cls, lGraph):
        nbNode = sum(len(g.lNode) for g in lGraph)
        nbEdge = sum(len(g.lEdge) for g in lGraph)
        return nbNode, nbEdge

    @classmethod
    def loadGraphs(cls
                   , cGraphClass          # graph class (must be subclass)
                   , lsFilename
                   , bNeighbourhood=True  # incident edges for each node, by type of edge
                   , bDetach=False        # keep or free the DOM
                   , bLabelled=False      # do we read node labels?
                   , iVerbose=0 
                   , attachEdge=False     # all incident edges for each node
                   , bConjugate=False     # Conjugate mode
                   ):
        """
        Load one graph per file, and detach its DOM
        return the list of loaded graphs
        """
        lGraph = []
        for sFilename in lsFilename:
            if iVerbose: traceln("\t%s"%sFilename)
            g = cGraphClass()
            g.parseXmlFile(sFilename, iVerbose)
            if not g.isEmpty():
                if attachEdge and bNeighbourhood: g.collectNeighbors(attachEdge=attachEdge)
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
    
        self.doc = etree.parse(sFilename)
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
            
            lPageEdge = Edge.Edge.computeEdges(lPrevPageNode, lPageNode, self.iGraphMode)
            
            self.lEdge.extend(lPageEdge)
            if iVerbose>=2: traceln("\tPage %5d    %6d nodes    %7d edges"%(pnum, len(lPageNode), len(lPageEdge)))
            
            lPrevPageNode = lPageNode
        if iVerbose: traceln("\t\t (%d nodes,  %d edges)"%(len(self.lNode), len(self.lEdge)) )
        
        return self

    def _iter_Page_DomNode(self, doc):
        """
        Parse a Xml DOM, by page

        iterator on the DOM, that returns per page:
            page-num (int), page object, page dom node
        
        """
        raise Exception("Must be specialized")

    def isEmpty(self): return self.lNode == []


    def collectNeighbors(self,attachEdge=False):
        """
        record the lists of hotizontal-, vertical- and cross-page neighbours for each node
        """
        for blk in self.lNode:
            blk.lHNeighbor = list()
            blk.lVNeighbor = list()
            blk.lCPNeighbor = list()
            if attachEdge:
                blk.edgeList=list()

        for edge in self.lEdge:
            a, b = edge.A, edge.B
            if isinstance(edge, Edge.SamePageEdge):
                if isinstance(edge, Edge.HorizontalEdge):
                    a.lHNeighbor.append(b)
                    b.lHNeighbor.append(a)
                else:
                    a.lVNeighbor.append(b)
                    b.lVNeighbor.append(a)
            else:
                if isinstance(edge, Edge.CrossPageEdge):
                    a.lCPNeighbor.append(b)
                    b.lCPNeighbor.append(a)
                else:
                    a.lCMPNeighbor.append(b)
                    b.lCMPNeighbor.append(a)
        if attachEdge:
            for edge in self.lEdge:
                a, b = edge.A, edge.B
                #Can I get all the correct function from a.lCPNeighbor etc ...
                a.edgeList.append(edge)
                b.edgeList.append(edge)


    def getNeighborClassMask(self):
        """
        record for each node a boolean for each label, indicating if the node is neighbor with a node having that label
        , one same page or accross page
        """    
        if self.aNeighborClassMask is None:
            self.aNeighborClassMask = np.zeros((len(self.lNode), self._nbLabelTot*2), dtype=np.int8)
            self._index()
            for edge in self.lEdge:
                a, b = edge.A, edge.B
                if isinstance(edge, Edge.SamePageEdge):
                    self.aNeighborClassMask[a._index, b.cls] = 1
                    self.aNeighborClassMask[b._index, a.cls] = 1
                else:
                    self.aNeighborClassMask[a._index, self._nbLabelTot+b.cls] = 1
                    self.aNeighborClassMask[b._index, self._nbLabelTot+a.cls] = 1
                    
        return self.aNeighborClassMask
        
    def detachFromDOM(self):
        """
        Detach the graph from the DOM node, which can then be freed
        """
        if self.doc != None:
            for nd in self.lNode: nd.detachFromDOM()
            self.doc = None
        #gc.collect()
        
    def revertEdges(self):
        """
        revert the direction of each edge of the graph
        """
        for e in self.lEdge: e.revertDirection()

    def addEdgeToDOM(self, Y=None):
        """
        To display the graph conveniently we add new Edge elements
        """
        ndPage = self.lNode[0].page.node    
        # w = int(ndPage.get("imageWidth"))
        ndPage.append(etree.Comment("Edges added to the XML for convenience"))
        for edge in self.lEdge:
            A, B = edge.A , edge.B   #shape.centroid, edge.B.shape.centroid
            ndEdge = PageXml.createPageXmlNode("Edge")
            ndEdge.set("src", edge.A.node.get("id"))
            ndEdge.set("tgt", edge.B.node.get("id"))
            ndEdge.set("type", edge.__class__.__name__)
            ndEdge.tail = "\n"
            ndPage.append(ndEdge)
            PageXml.setPoints(ndEdge, [(A.x1, A.y1), (B.x1, B.y1)]) 
                           
        return         
                
    # --- Numpy matrices --------------------------------------------------------
    def getXY(self, node_transformer, edge_transformer):
        """
        return a tuple (X,Y) for the graph  (X is a triplet)
        """
        self._index()
        
        if self._bMultitype:
            if self.bConjugate: 
                raise "Not yet implemented: conjugate of multitype graph"
            X, Y =  self._buildNodeEdgeLabelMatrices_T(node_transformer, edge_transformer, bY=True)
        else:
            X, Y = (  self.getX(node_transformer, edge_transformer)
                    , self.getY() )

        return (X, Y)

    def getX(self, node_transformer, edge_transformer):
        """
        make 1 node-feature matrix     (or list of matrices for multitype graphs)
         and 1 edge-feature matrix     (or list of matrices for multitype graphs)
         and 1 edge matrix             (or list of matrices for multitype graphs)
         for the graph
        return a triplet
        
        return X for the graph
        """
        self._index()
        if self._bMultitype:
            if self.bConjugate: raise "Not yet implemented: conjugate of multitype graph"
            X = self._buildNodeEdgeLabelMatrices_T(node_transformer, edge_transformer, bY=False)
        else:
            X = self._buildNodeEdgeMatrices_S(node_transformer, edge_transformer)
            if self.bConjugate: 
                X = self.convert_X_to_LineDual(X)
        return X

    def getY(self):
        """
        WARNING, in multitype graphs, the order of the Ys is bad
        """
        if self.bConjugate:
            Y = self._buildLabelMatrix_S_Y()
        else:
            Y = self._buildLabelMatrix_S()
        return Y

    # --- Conjugate --------------------------------------------------------
    def convert_X_to_LineDual(self, X):
        """
        Convert to a dual graph
        Animesh 2018
        Revisited by JL April 2019
        
        NOTE: isolated nodes are not reflected in the dual.
        Should we add a reflexive edge to have the node in the dual??
        """
        (nf, edge, ef) = X

        nb_edge = edge.shape[0]
        
        all_edges = []      # all edges created so far

        nf_dual     = ef    # edges become nodes
        edge_dual   = []
        ef_dual     = []            
        
        for i in range(nb_edge):
            edgei_from_idx, edgei_to_idx = edge[i]
            
            edge_from  = set([edgei_from_idx, edgei_to_idx])
            for j in range(i+1, nb_edge):    
                edge_to = set([edge[j][0], edge[j][1]])
                edge_candidate = edge_from.symmetric_difference(edge_to)
                # we should get 4, 2 or 0 primal nodes
                if len(edge_candidate) == 2 and edge_candidate not in all_edges:
                    # edge_to and edge_from share 1 primal node => create dual edge! 
                    all_edges.append(edge_candidate)
                    [shared_node_idx] = edge_from.intersection(edge_to)
                    shared_node_nf = nf[shared_node_idx]
                    ef_dual.append(shared_node_nf)
                    edge_dual.append([i, j])

        nf_dual     = np.array(nf_dual)
        edge_dual   = np.array(edge_dual)
        ef_dual     = np.array(ef_dual)

        assert (edge_dual.shape[0] == ef_dual.shape[0])

        return (nf_dual, edge_dual, ef_dual)

    #----- Indexing Graph Objects -----   
    def _index(self, bForce=False):
        """
        - index NodeType(s) if not done already (unless bForce is True)
        - add _index attribute to all nodes
        return True if indexing was required, False oterwise (on second and next calls)
        """
        try:
            bForce or self.__bNodeIndexed
            return False
        except AttributeError:
            self._indexNodeTypes()
            for i, nd in enumerate(self.lNode): nd._index = i
            self.__bNodeIndexed = True
            return True
            
    def _indexNodeTypes(self):
        """
        add _index attribute to registered NodeType
        """
        for i, nt in enumerate(self._lNodeType): nt._index = i

    #----- SINGLE TYPE -----   
    def _buildNodeEdgeMatrices_S(self, node_transformer, edge_transformer):
        """
        make 1 node-feature matrix
         and 1 edge-feature matrix
         and 1 edge matrix
         for the graph
        return a triplet
        """
        node_features = node_transformer.transform(self.lNode)
        edges = self._BuildEdgeMatrix_S()
        edge_features = edge_transformer.transform(self.lEdge)
        return (node_features, edges, edge_features)       
                
    def _BuildEdgeMatrix_S(self):
        """
        - add an index attribute to nodes
        - build an edge matrix on this basis
        - return the edge matrix (a 2-columns matrix)
        """
        #SINGLE TYPE GRAPH, WE KEEP THE OLD CODE
        edges = np.empty( (len(self.lEdge), 2) , dtype=np.int)
        for i, edge in enumerate(self.lEdge):
            edges[i,:] = edge.A._index, edge.B._index
            
#  #better code?
#             edges = np.fromiter( itertools.chain.from_iterable( (edge.A._index, edge.B._index) for edge in self.lEdge )
#                                  , dtype=np.int, count=2*len(self.lEdge))
#             edge = edges.reshape(len(self.lEdge), 2)
        return edges

    def _buildLabelMatrix_S(self):
        """
        Return the matrix of labels
        """
        #better code based on fromiter is below (I think, JLM April 2017) 
        #Y = np.array( [nd.cls for nd in self.lNode] , dtype=np.uint8)
        Y = np.fromiter( (nd.cls for nd in self.lNode), dtype=np.int, count=len(self.lNode))
        return Y

    def _buildLabelMatrix_S_Y(self):
        """
        Return the matrix of labels of edges
        """
        #better code based on fromiter is below (I think, JLM April 2017) 
        #Y = np.array( [nd.cls for nd in self.lNode] , dtype=np.uint8)
        Y = np.fromiter( (e.cls for e in self.lEdge), dtype=np.int, count=len(self.lEdge))
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
        n_type   = len(self._lNodeType)
        n_type_2 = n_type * n_type
        
        #list nodes per type
        lNodeByType =[ list() for _i in range(n_type)]
        _a_node_count_by_type = np.zeros((n_type,), dtype=np.int) 
        for nd in self.lNode:
            type_index = nd.type._index
            lNodeByType[type_index].append(nd)
            #to define the edges
            nd._index_in_type = _a_node_count_by_type[type_index]
            _a_node_count_by_type[type_index] += 1
        node_features = node_transformer.transform(lNodeByType)
        
        if bY:
            #we need to compute Y and reorder it since we have grouped the nodes by type, and ordered the types
            #so node with index i, that ends in type Ti, with index in type j has now the index cumulative_count_by_type[i-1] + j
            node_index_offset_by_typ = np.cumsum([0]+_a_node_count_by_type.tolist())
            Y = np.zeros( (len(self.lNode),), dtype=np.int)
            #TODO optimize this code to avoid going twice thru each node and accessing it attributes twice
            for nd in self.lNode:
                Ti = nd.type._index
                new_index = node_index_offset_by_typ[Ti] + nd._index_in_type
                Y[new_index] = nd.cls
        
        #definition of edges and list of edges by types
        t_edges = np.empty( (len(self.lEdge), 3) , dtype=np.int)      #edge_type_index, node_index_in_type, node_index_in_type
        lEdgeByType =[ list() for i in range(n_type_2)]
        for i, edge in enumerate(self.lEdge):
            A, B = edge.A, edge.B
            edge_type_index = A.type._index * n_type + B.type._index
            t_edges[i,:] = (edge_type_index,                            #index of edge's type
                            A._index_in_type,                           #index in A's type
                            B._index_in_type)                           #index in B's type
            lEdgeByType[edge_type_index].append(edge)
        edges = [ t_edges[ t_edges[:, 0]==type_index ][:,1:3] for type_index in range(n_type_2) ]
        
        edge_features = edge_transformer.transform(lEdgeByType)

        if bY:
            return (node_features, edges, edge_features), Y
        else:       
            return (node_features, edges, edge_features)    

    #----- STUFF -----  
    def getNodeIndexByPage(self):
        """
        return a list of list of index
        Both lists are sorted (by page number and by index)
        empty pages are skipped (and _not_ reflected as empty list)
        """
        if not self.lNode: raise ValueError("Empty graph")
        try:
            self.lNode[0]._index
        except AttributeError:
            for i, nd in enumerate(self.lNode):
                nd._index = i

        dlIndexByPage = collections.defaultdict(list)
        for nd in self.lNode:
            dlIndexByPage[nd.pnum].append(nd._index)
        
        llIndexByPage = []
        for pnum in sorted(dlIndexByPage.keys()):
            llIndexByPage.append( sorted(dlIndexByPage[pnum]) )
        return llIndexByPage
            
        
        
        

        
        
