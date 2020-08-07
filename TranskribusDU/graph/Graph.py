# -*- coding: utf-8 -*-

"""
    Computing the graph for a XML document
    
    Copyright Xerox(C) 2016 JL. Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""

import collections

import numpy as np

from common.trace import traceln

from . import Edge


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
    
    bConjugate = False
    
    sIN_FORMAT  = "undefined"   # tell here which input format is expected, subclasses must specialise this
    sDU         = "_du"
    sOUTPUT_EXT = ".out"

    #--- CONSTRAINTS
    _lPageConstraintDef = None  #optionnal page-level constraints

    def __init__(self, lNode = [], lEdge = []):
        self.lNode = lNode
        self.lEdge = lEdge
        self.doc   = None   # the document object (can be a DOM, or a JSON, or...)
        
        self.lCluster = None # list of clusters (resulting from a segmentation task)
        
        self.aNeighborClassMask = None   #did we compute the neighbor class mask already?
        
        self.aLabelCount = None #count of seen labels
    
    # --- I/O
    @classmethod
    def isOutputFilename(cls, sFilename):
        return sFilename.endswith(cls.sDU+cls.sOUTPUT_EXT)

    @classmethod
    def getOutputFilename(cls, sFilename):
        return sFilename[:sFilename.rindex(".")] + cls.sDU + cls.sOUTPUT_EXT


    def parseDocFile(self, sFilename, iVerbose=0):
        """
        Load that document as a Graph.
        Also set the self.doc variable!
        
        Return a Graph object
        """
        raise GraphException("You must specialise this class method")

    @classmethod
    def getDocInputFormat(cls):
        """
        return a human-readable string describing the expected input format
        """
        return cls.sIN_FORMAT
    
    def detachFromDoc(self):
        """
        Graph and graph nodes may have kept a reference to other data .
        Here we detach them
        """
        return
            
    @classmethod
    def saveDoc(cls, sFilename, doc, lG, sCreator, sComment):
        """
        sFile is the input filename
        doc   is the input data (DOM, or JSON for now) possibly enriched by the 
            prediction, depending on the class of graph
        lG    is the list of graphs, possibly enriched by the prediction
        """
        raise GraphException("You must specialise this class method")

    # --- Graph COnstruction Mode
    @classmethod
    def getGraphMode(cls):
        return cls.iGraphMode
    @classmethod
    def setGraphMode(cls, iGraphMode):
        assert iGraphMode in (1,2,4)
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
        raise GraphException("You must specialise this class method")

    def form_cluster(self, Y_proba):
        """
        Do whatever is required on the (primal) graph, given the edge labels
            Y_proba is the predicted edge label probability array of shape (N_edges, N_labels)
        
        return a ClusterList of Cluster object
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
        return [sLabelName for lblSet in cls._lNodeType for sLabelName in lblSet.getLabelNameList()]

    def parseDocLabels(self):
        """
        Parse the label of the graph from the dataset, and set the node label
        return the set of observed class (set of integers in N+)
        """
        setSeensLabels = set()
        for nd in self.lNode:
            nodeType = nd.type 
            #a LabelSet object knows how to parse a DOM node of a Graph object!!
            sLabel = nodeType.parseDocNodeLabel(nd)
            try:
                cls = self._dClsByLabel[sLabel]  #Here, if a node is not labelled, and no default label is set, then KeyError!!!
            except KeyError:
                raise ValueError("Page %d, unknown label '%s' in %s (Known labels are %s)"%(nd.pnum, sLabel, str(nd.node), self._dClsByLabel))
            nd.cls = cls
            setSeensLabels.add(cls)
        return setSeensLabels    

    def setDocLabels(self, Y):
        """
        Set the labels of the graph nodes from the Y matrix
        """
        for i,nd in enumerate(self.lNode):
            sLabel = self._dLabelByCls[ Y[i] ]
            nd.type.setDocNodeLabel(nd, sLabel)
        return
    
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
                   , bDetach=False        # keep or free the source data
                   , bLabelled=False      # do we read node labels?
                   , iVerbose=0 
                   , attachEdge=False     # all incident edges for each node
                   ):
        """
        Load one graph per file, and detach its DOM
        return the list of loaded graphs
        """
        lGraph = []
        for sFilename in lsFilename:
            if iVerbose: traceln("\t%s"%sFilename)
            g = cGraphClass()
            g.parseDocFile(sFilename, iVerbose)
            g._index()
            if not g.isEmpty():
                if attachEdge and bNeighbourhood: g.collectNeighbors(attachEdge=attachEdge)
                if bNeighbourhood: g.collectNeighbors()
                if bLabelled: g.parseDocLabels()
                if bDetach: g.detachFromDoc()
            lGraph.append(g)
        return lGraph

    @classmethod
    def castGraphList(cls
                   , cGraphClass          # graph class (must be subclass)
                   , lGraph
                   , iVerbose=0 
                   ):
        """
        Here we create an instance of graph that reuses the lists of nodes and edge from another graph
        """
        assert len(cGraphClass.getNodeTypeList()) == 1
        new_ndType = cGraphClass.getNodeTypeList()[0]
        lNewGraph = []
        for g in lGraph:
            new_g = cGraphClass()
            new_g.doc   = g.doc

            new_g.lNode = g.lNode
            # we need to change the node type of all nodes...
            # I always knew it was bad to have one type attribute on each node in signle-type graphs...
            for _nd in g.lNode: _nd.type = new_ndType

            new_g.lEdge = g.lEdge
            lNewGraph.append(new_g)
        return lNewGraph

    def _iter_Page_DocNode(self, doc):
        """
        Parse a Xml DOM, by page

        iterator on the DOM, that returns per page:
            page-num (int), page object, page dom node
        
        """
        raise Exception("Must be specialized")

    def isEmpty(self): 
        return self.lNode == []

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
        
    def revertEdges(self):
        """
        revert the direction of each edge of the graph
        """
        for e in self.lEdge: e.revertDirection()

    # --- Numpy matrices --------------------------------------------------------
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
            X = self._buildNodeEdgeLabelMatrices_T(node_transformer, edge_transformer, bY=False)
        else:
            X = self._buildNodeEdgeMatrices_S(node_transformer, edge_transformer)
        return X

    def getY(self):
        """
        WARNING, in multitype graphs, the order of the Ys is bad
        """
        Y = np.fromiter( (nd.cls for nd in self.lNode), dtype=np.int, count=len(self.lNode))
        return Y

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
            for i, nt in enumerate(self._lNodeType): nt._index = i
            for i, nd in enumerate(self.lNode)     : nd._index = i
            self.__bNodeIndexed = True
            return True
            
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

