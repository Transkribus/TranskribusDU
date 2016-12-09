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
import numpy as np

from common.trace import traceln

class Graph:
    """
    A graph to be used as a CRF graph with pystruct
    """
    
    #The labels for those graphs
    _sOTHER_LABEL   = "OTHER"
    lsLabel         = None #list of labels
    sDefaultLabel   = None #when no annotation, do we set automatically to this label? (e.g."OTHER")
    dLabelByCls     = None    
    dClsByLabel     = None
    nCls            = None
            
    def __init__(self, lNode = [], lEdge = []):
        self.lNode = lNode
        self.lEdge = lEdge
        self.doc   = None
        
    # --- Graph building --------------------------------------------------------
    def parseXmlFile(self, sFilename, iVerbose=0):
        """
        Load that document as a CRF Graph.
        Also set the self.doc variable!
        """
        raise Exception("Method must be overridden")
    
    def detachFromDOM(self):
        """
        Detach the graph from the DOM node, which can then be freed
        """
        for nd in self.lNode: nd.detachFromDOM()
        self.doc.freeDoc()
        self.doc = None

    # --- Labels ----------------------------------------------------------
    def getLabelList(cls):
        return cls.lsLabel
    getLabelList = classmethod(getLabelList)
    
    def setLabelList(cls, lsLabel, bOther=True):
        """set those properties:
            self.lsLabel    - list of label names
            dLabelByCls     - dictionary name -> id
            dClsByLabel     - dictionary id -> name
            self.nCls       - number of different labels
        """
        if bOther: 
            assert cls._sOTHER_LABEL not in lsLabel, "the label for class 'OTHER' conflicts with a task-specific label"
            cls.lsLabel        = [cls._sOTHER_LABEL] + lsLabel
            cls.sDefaultLabel  = cls._sOTHER_LABEL
        else:
            cls.lsLabel        = lsLabel
            cls.sDefaultLabel  = None
         
        cls.dLabelByCls = { i:sLabel for i,sLabel in enumerate(cls.lsLabel) }         
        cls.dClsByLabel = { sLabel:i for i,sLabel in enumerate(cls.lsLabel) } 
        cls.nCls = len(cls.lsLabel)        
        return cls.lsLabel
    setLabelList = classmethod(setLabelList)
    
    def parseDomLabels(self):
        """
        Parse the label of the graph from the dataset, and set the node label
        return the set of observed class (set of integers in N+)
        """
        setSeensLabels = set()
        for nd in self.lNode:
            sLabel = self.parseDomNodeLabel(nd.node, self.sDefaultLabel)
            cls = self.dClsByLabel[sLabel]  #Here, if a node is not labelled, and no default label is set, then KeyError!!!
            nd.cls = cls
            setSeensLabels.add(cls)
        return setSeensLabels    

    def setDomLabels(self, Y):
        """
        Set the labels of the graph nodes from the Y matrix
        return the DOM
        """
        for i,nd in enumerate(self.lNode):
            sLabel = self.lsLabel[ Y[i] ]
            if sLabel != self.sDefaultLabel:
                self.setDomNodeLabel(nd.node, sLabel)
        return self.doc

    def setDomNodeLabel(self, node, sLabel):
        """
        Set the DOM node associated to this graph node to a certain label
        """        
        raise Exception("Method must be overridden")
    
    # --- Utilities ---------------------------------------------------------
    def loadGraphs(cls, lsFilename, bDetach=False, bLabelled=False, iVerbose=0):
        """
        Load one graph per file, and detach its DOM
        return the list of loaded graphs
        """
        lGraph = []
        for sFilename in lsFilename:
            if iVerbose: traceln("\t%s"%sFilename)
            g = cls()
            g.parseXmlFile(sFilename, iVerbose)
            if bLabelled: g.parseDomLabels()
            if bDetach: g.detachFromDOM()
            lGraph.append(g)
        return lGraph
    loadGraphs = classmethod(loadGraphs)

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

        
        
        
