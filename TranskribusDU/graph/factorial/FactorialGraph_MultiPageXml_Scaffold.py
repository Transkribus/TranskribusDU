# -*- coding: utf-8 -*-

"""
    Computing the graph for a MultiPageXml document

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

from .FactorialGraph_MultiPageXml import FactorialGraph_MultiPageXml


# ------------------------------------------------------------------------------------------------------------------------------------------------
class FactorialGraph_MultiPageXml_Scaffold(FactorialGraph_MultiPageXml):
    """
    FactorialCRF for MultiPageXml document
    
    We build a multitype graph, which can be seen as a layered graph. 
    A node appears in each layer, under a different type, so as to get a label in each different label space.
    
    For instance with node types * and #, any edge -- is replicated among layers, and an edge | connect the multiple replicas of each node:
    
    * -- * -- ...
    | \/ |
    # -- # -- ...
    
    But in this SCAFFOLD variant, if node A connects to B, we have an edge from Ai to Bj for all i,j being layer index.
    
    """
    
    def __init__(self, lNode = [], lEdge = []):
        FactorialGraph_MultiPageXml.__init__(self, lNode, lEdge)

        
    #----- MULTITYPE -----  
    def _buildNodeEdgeLabelMatrices_T(self, node_transformer, edge_transformer, bY=True):
        """
        with n as number of nodes
        with e as number of edges
        with t as number of types
        we end with e.t^2 + n.t!/(t-2)!/2!) edges in the scaffold graph
        
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
        (NF, E, EF) = FactorialGraph_MultiPageXml._buildNodeEdgeMatrices_S(self, node_transformer, edge_transformer)
                                                                  
        #The NF per type are all the same
        lNF = [NF for _nodeType in self.getNodeTypeList()]
        
        
        # for cross-type, we add 1 feature that encode the artificial croos-type
        # edge from a node to 'itself'
        
        #The Edge definitions and edge features
        nbEdge = E.shape[0]

        # edge within a layer are all the same (but between the node replcias for the layer)
        # we add one edge between each pair of node replica between each pairs of layer  (only in one direction)
        lE = []
        lEF = []


        # scaffold part: just copies of the intra-type stuff 
        E_scaffold = E
        # to have same number of features for all pair of types, we need a
        # column of zeros (see EF_ladder_and_scaffold)
        EF_scaffold = np.hstack( [np.zeros((nbEdge,1), dtype=EF.dtype), EF] )
        
        # ladder-part union scaffold
        # node-to-same-node edge  union scaffold stuff
        E_ladder = np.array([[i, i] for i in range(nbNode)], dtype=E.dtype)
        E_ladder_and_scaffold = np.vstack([E_ladder, E])
        
        # we increment the observed number of edge features by 1, to have
        # one model weight for the ladder edge
        EF_ladder_and_scaffold = np.zeros((nbNode+nbEdge, 1+EF.shape[1]),
                                          dtype=EF.dtype)
        EF_ladder_and_scaffold[0:nbNode,0]  = 1
        EF_ladder_and_scaffold[nbNode:, 1:] = EF

        for typ1 in range(nbType):
            
            # scaffold: cross-type edges that are same as original intra-type
            for i in range(typ1): 
                lE.append (E_scaffold) 
                lEF.append(EF_scaffold)
                
            #edge within a layer
            lE.append (E)
            lEF.append(EF)
            
            for _typ2 in range(typ1+1, nbType): 
                # ladder-part  +
                # scaffold: cross-type edges that are same as original intra-type
                lE.append (E_ladder_and_scaffold)
                lEF.append(EF_ladder_and_scaffold)

        #        traceln( (nbNode, nbEdge, sum( e.shape[0] for e in lE)) )
        if bY:
            Y = np.zeros( (len(self.lNode)*nbType,), dtype=np.int)
            
            for ityp in range(nbType):
                zeroNode = ityp*nbNode
                Y[zeroNode:zeroNode+nbNode] = [nd.cls[ityp] for nd in self.lNode]
                
            return (lNF, lE, lEF), Y
        else:
            return (lNF, lE, lEF)

