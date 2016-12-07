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
    _sOTHER_LABEL = "OTHER"
    
    def __init__(self, lNode = [], lEdge = []):
        self.lNode = lNode
        self.lEdge = lEdge
        self.doc   = None
        self.lsLabel        = None #list of labels
        self.sDefaultLabel  = None #when no annotation, do we set automatically to this label? (e.g."OTHER")
        
    # --- Graph building --------------------------------------------------------
    def parseFile(self, sFilename, iVerbose=0):
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
    def getLabelList(self):
        return self.lsLabel
    
    def setLabelList(self, lsLabel, bOther=True):
        """set those properties:
            self.lsLabel    - list of label names
            dLabelByCls     - dictionary name -> id
            dClsByLabel     - dictionary id -> name
            self.nCls       - number of different labels
        """
        if bOther: 
            assert self._sOTHER_LABEL not in lsLabel, "the label for class 'OTHER' conflicts with a task-specific label"
            self.lsLabel        = [self._sOTHER_LABEL] + lsLabel
            self.sDefaultLabel  = self._sOTHER_LABEL
        else:
            self.lsLabel        = lsLabel
            self.sDefaultLabel  = None
         
        self.dLabelByCls = { i:sLabel for i,sLabel in enumerate(self.lsLabel) }         
        self.dClsByLabel = { sLabel:i for i,sLabel in enumerate(self.lsLabel) } 
        self.nCls = len(self.lsLabel)        
        return self.lsLabel

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

#     def parseDomNodeLabel(self, node, defaultLabel=""):
#         """
#         Parse the graph Dom node label and return it
#         if a default label is given, absence of label becomes that one
#         """
#         raise Exception("Method must be overridden")

    def setDomNodeLabel(self, node, sLabel):
        """
        Set the DOM node associated to this graph node to a certain label
        """        
        raise Exception("Method must be overridden")
    
    # --- Utilities ---------------------------------------------------------
    def loadDetachedGraphs(cls, lsFilename, bLabelled=False, iVerbose=0):
        """
        Load one graph per file, and detach its DOM
        return the list of loaded graphs
        """
        lGraph = []
        for sFilename in lsFilename:
            if iVerbose: traceln("\t%s"%sFilename)
            g = cls()
            g.parseFile(sFilename, iVerbose)
            if bLabelled: g.parseDomLabels()
            g.detachFromDOM()
            lGraph.append(g)
        return lGraph
    loadDetachedGraphs = classmethod(loadDetachedGraphs)

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

        
        
        
