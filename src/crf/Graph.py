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

class Graph:
    """
    A graph to be used as a CRF graph with pystruct
    """
    
    def __init__(self, lNode = [], lEdge = []):
        self.lNode = lNode
        self.lEdge = lEdge
        
    # --- Graph building --------------------------------------------------------
    def parseFile(self, sFilename):
        """
        Load that document as a CRF Graph.
        """
        raise Exception("Method must be overridden")
    
    def detachFromDOM(self):
        """
        Detach the graph from the DOM node, which can then be freed
        """
        for nd in self.lNode: nd.detachFromDOM()

    # --- transformers --------------------------------------------------------
    def getNodeTransformer(self):
        """
        Obtain the transformer that prodcues features for each
        """
        raise Exception("Method must be overridden")

    def getEdgeTransformer(self):
        """
        Obtain the transformer that prodcues features for each
        """
        raise Exception("Method must be overridden")

    # --- Numpy matrices --------------------------------------------------------
    def indexNodes_and_BuildEdgeMatrix(self):
        """
        - add an index attribute to nodes
        - build an edge matrix on this basis
        - return the edge matrix (a 2-columns matrix)
        """
        for i, nd in enumerate(self.lNode):
            nd.index = i

        edges = np.empty( (self.len(self.lEdge), 2) , dtype=np.int32)
        for i, edge in enumerate(self.lEdge):
            edges[i,0] = edge.A.index
            edges[i,1] = edge.B.index
        return edges
    indexNodes_and_BuildEdgeMatrix = classmethod(indexNodes_and_BuildEdgeMatrix)

        