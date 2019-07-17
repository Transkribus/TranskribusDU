# -*- coding: utf-8 -*-

"""
    Common feature definitions for DU task.
    
    Copyright Xerox(C) 2019 JL. Meunier

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
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""

from graph.Edge                       import HorizontalEdge, VerticalEdge

from graph.FeatureDefinition          import FeatureDefinition
from graph.FeatureDefinition_Standard import Node_Geometry
from graph.FeatureDefinition_Standard import Node_Neighbour_Count
from graph.FeatureDefinition_Standard import Node_Text_NGram
from graph.FeatureDefinition_Standard import Edge_Type_1Hot
from graph.FeatureDefinition_Standard import Edge_1
from graph.FeatureDefinition_Standard import Edge_Geometry
from graph.FeatureDefinition_Standard import Edge_Source_Text_NGram
from graph.FeatureDefinition_Standard import Edge_Target_Text_NGram
from graph.FeatureDefinition_Standard import EdgeClassShifter
from graph.Transformer                import Pipeline, FeatureUnion


class Features_June19_Simple(FeatureDefinition):
    """
    Simple features:
        NODE: geometry
        EDGE: type and geometry
    """
    
    n_QUANTILES = 16
    
    bShiftEdgeByClass = False

    def __init__(self): 
        FeatureDefinition.__init__(self)
        
        # NODES
        node_transformer = FeatureUnion([                               \
              ("geometry"           , Node_Geometry())         # one can set nQuantile=...
                                        ])
    
        # EDGES
        # which types of edge can we get??
        # It depends on the type of graph!!
        lEdgeClass = [HorizontalEdge, VerticalEdge]
        # standard set of features, including a constant 1 for CRF
        edge_transformer =  FeatureUnion([                                            \
                  ('1hot'   , Edge_Type_1Hot(lEdgeClass=lEdgeClass)) # Edge class 1 hot encoded (PUT IT FIRST)
                , ('geom'   , Edge_Geometry())                       # one can set nQuantile=...
                            ])        
        
        # OPTIONNALLY, you can have one range of features per type of edge.
        # the 1-hot encoding must be the first part of the union and it will determine
        #   by how much the rest of the feature are shifted.
        #
        # IMPORTANT: 1hot is first of union   AND   the correct number of edge classes
        if self.bShiftEdgeByClass:
            edge_transformer = Pipeline([
                      ('edge_transformer', edge_transformer)
                    , ('shifter', EdgeClassShifter(len(lEdgeClass)))
                    ])
        
        self.setTransformers(node_transformer, edge_transformer)


class Features_June19_Simple_Shift(Features_June19_Simple):
    """
    Same as Features_June19_Simple, but edge features are shifted by edge class
    """
    bShiftEdgeByClass = True


class Features_June19_Full(FeatureDefinition):
    """
    All features we had historically (some specific to CRF):
        NODE: geometry, neighbor count, text
        EDGE: type, constant 1, geometry, text of source and target nodes
        The features of the edges are shifted by class, apart the 1-hot ones.
    """
    
    n_QUANTILES = 16
    
    bShiftEdgeByClass = False
    
    def __init__(self): 
        FeatureDefinition.__init__(self)
        
        # NODES
        node_transformer = FeatureUnion([                               \
              ("geometry"           , Node_Geometry())         # one can set nQuantile=...
            , ("neighbor_count"     , Node_Neighbour_Count())  # one can set nQuantile=...
            , ("text"               , Node_Text_NGram( 'char'    # character n-grams
                                                       , 500     # number of N-grams
                                                       , (2,3)    # N
                                                       , False    # lowercase?))
                                                       ))
                                        ])
    
        # EDGES
        # which types of edge can we get??
        # It depends on the type of graph!!
        lEdgeClass = [HorizontalEdge, VerticalEdge]
        # standard set of features, including a constant 1 for CRF
        fu =  FeatureUnion([                                            \
                  ('1hot'   , Edge_Type_1Hot(lEdgeClass=lEdgeClass)) # Edge class 1 hot encoded (PUT IT FIRST)
                , ('1'      , Edge_1())                              # optional constant 1 for CRF
                , ('geom'   , Edge_Geometry())                       # one can set nQuantile=...
                , ('src_txt', Edge_Source_Text_NGram( 'char'    # character n-grams
                                               , 250     # number of N-grams
                                               , (2,3)    # N
                                               , False    # lowercase?))
                                               ))
                , ('tgt_txt', Edge_Target_Text_NGram( 'char'    # character n-grams
                                               , 250     # number of N-grams
                                               , (2,3)    # N
                                               , False    # lowercase?))
                                               ))
                            ])        
        
        # you can use directly this union of features!
        edge_transformer = fu
        
        # OPTIONNALLY, you can have one range of features per type of edge.
        # the 1-hot encoding must be the first part of the union and it will determine
        #   by how much the rest of the feature are shifted.
        #
        # IMPORTANT: 1hot is first of union   AND   the correct number of edge classes
        if self.bShiftEdgeByClass:
            ppl = Pipeline([
                      ('fu', fu)
                    , ('shifter', EdgeClassShifter(len(lEdgeClass)))
                    ])
            edge_transformer = ppl
        
        self.setTransformers(node_transformer, edge_transformer)


class Features_June19_Full_Shift(Features_June19_Full):
    """
    Same as Features_June19_Full, but edge features are shifted by edge class
    """
    bShiftEdgeByClass = True

