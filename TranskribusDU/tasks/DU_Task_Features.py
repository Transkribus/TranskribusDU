# -*- coding: utf-8 -*-

"""
    Common feature definitions for DU task.
    
    Copyright NAVER(C) 2019 JL. Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""
from common.trace import traceln

from graph.Edge                       import CelticEdge, EdgeInterCentroid

from graph.FeatureDefinition          import FeatureDefinition
from graph.FeatureDefinition_Standard import Node_Geometry
from graph.FeatureDefinition_Standard import Node_Neighbour_Count
from graph.FeatureDefinition_Standard import Node_Text_NGram
from graph.FeatureDefinition_Standard import Edge_Type_1Hot
from graph.FeatureDefinition_Standard import Edge_1
from graph.FeatureDefinition_Standard import Edge_Geometry
from graph.FeatureDefinition_Standard import Edge_BB
from graph.FeatureDefinition_Standard import EdgeClassShifter
from graph.FeatureDefinition_Standard import getDefaultEdgeClassList, appendDefaultEdgeClassList

from graph.Transformer_Generic        import NodeTransformerTextSentencePiece
from graph.Transformer                import Pipeline, FeatureUnion

from graph.PageXmlSeparatorRegion import Separator_boolean, Separator_num


# EDGES
# which types of edge can we get??
# It depends on the type of graph!!
def addCelticEdgeType():
    appendDefaultEdgeClassList(CelticEdge)
    traceln("lDEFAULT_EDGE_CLASS=", getDefaultEdgeClassList())
    
def addInterCentroidEdgeType():
    appendDefaultEdgeClassList(EdgeInterCentroid)
    traceln("lDEFAULT_EDGE_CLASS=", getDefaultEdgeClassList())

    
# --- SIMPLE ------------------------------------------------------
class Features_June19_Simple(FeatureDefinition):
    """
    Simple features:
        NODE: geometry
        EDGE: type and geometry
    """
    
    n_QUANTILES = 16
    
    bShiftEdgeByClass = False
    bSeparator        = False
    
    def __init__(self): 
        FeatureDefinition.__init__(self)
        
        # NODES
        self.lNodeFeature = [
              ("geometry"           , Node_Geometry())         # one can set nQuantile=...
                                        ]
        node_transformer = FeatureUnion(self.lNodeFeature)
    
        # EDGES
        # standard set of features, including a constant 1 for CRF
        self.lEdgeFeature = [
                  ('1hot'   , Edge_Type_1Hot(lEdgeClass=getDefaultEdgeClassList())) # Edge class 1 hot encoded (PUT IT FIRST)
                , ('geom'   , Edge_Geometry())                       # one can set nQuantile=...
                # TOO SLOOOOW on cTDar , ('BB'     , Edge_BB())
                , ('1'      , Edge_1())                              # optional constant 1 for CRF
                            ]
        if self.bSeparator:
            self.lEdgeFeature = self.lEdgeFeature + [ 
                  ('sprtr_bool', Separator_boolean())
                , ('sprtr_num' , Separator_num())
                            ]
        edge_transformer =  FeatureUnion(self.lEdgeFeature)        
        
        # OPTIONNALLY, you can have one range of features per type of edge.
        # the 1-hot encoding must be the first part of the union and it will determine
        #   by how much the rest of the feature are shifted.
        #
        # IMPORTANT: 1hot is first of union   AND   the correct number of edge classes
        if self.bShiftEdgeByClass:
            edge_transformer = Pipeline([
                      ('edge_transformer', edge_transformer)
                    , ('shifter', EdgeClassShifter(len(getDefaultEdgeClassList())))
                    ])
        
        self.setTransformers(node_transformer, edge_transformer)


class Features_June19_Simple_Shift(Features_June19_Simple):
    """
    Same as Features_June19_Simple, but edge features are shifted by edge class
    """
    bShiftEdgeByClass = True


# --- FULL ------------------------------------------------------
class Features_June19_Full(FeatureDefinition):
    """
    All features we had historically (some specific to CRF):
        NODE: geometry, neighbor count, text
        EDGE: type, constant 1, geometry, text of source and target nodes
        The features of the edges are shifted by class, apart the 1-hot ones.
    """
    
    n_QUANTILES = 16
    
    bShiftEdgeByClass = False
    bSeparator        = False
    
    def __init__(self): 
        FeatureDefinition.__init__(self)
        
        # NODES
        self.lNodeFeature = [                               \
              ("geometry"           , Node_Geometry())         # one can set nQuantile=...
            , ("neighbor_count"     , Node_Neighbour_Count())  # one can set nQuantile=...
            , ("text"               , Node_Text_NGram( 'char'    # character n-grams
                                                       , 500     # number of N-grams
                                                       , (2,3)    # N
                                                       , False    # lowercase?))
                                                       ))
                            ]
        node_transformer = FeatureUnion(self.lNodeFeature)
        
        # EDGES
        # which types of edge can we get??
        # It depends on the type of graph!!
        # standard set of features, including a constant 1 for CRF
        self.lEdgeFeature = [                                            \
                  ('1hot'   , Edge_Type_1Hot(lEdgeClass=getDefaultEdgeClassList())) # Edge class 1 hot encoded (PUT IT FIRST)
                , ('1'      , Edge_1())                              # optional constant 1 for CRF
                , ('geom'   , Edge_Geometry())                       # one can set nQuantile=...
                            ]
        if self.bSeparator:
            self.lEdgeFeature = self.lEdgeFeature + [ 
                  ('sprtr_bool', Separator_boolean())
                , ('sprtr_num' , Separator_num())
                            ]
        fu =  FeatureUnion(self.lEdgeFeature)        
        
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
                    , ('shifter', EdgeClassShifter(len(getDefaultEdgeClassList())))
                    ])
            edge_transformer = ppl
        
        self.setTransformers(node_transformer, edge_transformer)


class Features_June19_Full_Shift(Features_June19_Full):
    """
    Same as Features_June19_Full, but edge features are shifted by edge class
    """
    bShiftEdgeByClass = True


# --- FULL with SentencePiece model -----------------------------------------
class Features_June19_Full_SPM(FeatureDefinition):
    """
    All features we had historically (some specific to CRF):
        NODE: geometry, neighbor count, text
        EDGE: type, constant 1, geometry, text of source and target nodes
        The features of the edges are shifted by class, apart the 1-hot ones.
        
        We use a SPM model to compute textual features as a set of words
    """
    
    n_QUANTILES = 16
    
    bShiftEdgeByClass = False
    bSeparator        = False
    
    def __init__(self, sSPModel): 
        """
        sp is a SentencePiece model name
        """
        FeatureDefinition.__init__(self)

        # NODES
        self.lNodeFeature = [                               \
              ("geometry"           , Node_Geometry())         # one can set nQuantile=...
            , ("neighbor_count"     , Node_Neighbour_Count())  # one can set nQuantile=...
            # , ("text"               , NodeTransformerTextSentencePiece(sp))    # TypeError: can't pickle SwigPyObject objects
            , ("text"               , NodeTransformerTextSentencePiece(sSPModel))    # TypeError: can't pickle SwigPyObject objects
                            ]
        node_transformer = FeatureUnion(self.lNodeFeature)
        
        # EDGES
        # which types of edge can we get??
        # It depends on the type of graph!!
        # standard set of features, including a constant 1 for CRF
        self.lEdgeFeature = [                                            \
                  ('1hot'   , Edge_Type_1Hot(lEdgeClass=getDefaultEdgeClassList())) # Edge class 1 hot encoded (PUT IT FIRST)
                , ('1'      , Edge_1())                              # optional constant 1 for CRF
                , ('geom'   , Edge_Geometry())                       # one can set nQuantile=...
                            ]
        if self.bSeparator:
            self.lEdgeFeature = self.lEdgeFeature + [ 
                  ('sprtr_bool', Separator_boolean())
                , ('sprtr_num' , Separator_num())
                            ]
        fu =  FeatureUnion(self.lEdgeFeature)        
        
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
                    , ('shifter', EdgeClassShifter(len(getDefaultEdgeClassList())))
                    ])
            edge_transformer = ppl
        
        self.setTransformers(node_transformer, edge_transformer)




class Features_June19_Full_SPM_Shift(Features_June19_Full_SPM):
    """
    Same as Features_June19_Full, but edge features are shifted by edge class
    """
    bShiftEdgeByClass = True


# --- Separator ------------------------------------------------------
class Features_June19_Simple_Separator(Features_June19_Simple):
    """
    Same as Features_June19_Simple, with additional features on edges
    """
    bSeparator = True


class Features_June19_Full_Separator(Features_June19_Full):
    """
    Same as Features_June19_Full, with additional features on edges
    """
    bSeparator = True

class Features_June19_Full_SPM_Separator(Features_June19_Full_SPM):
    """
    Same as Features_June19_Full, with additional features on edges
    """
    bSeparator = True


# --- Separator Shifted ------------------------------------------------------
class Features_June19_Simple_Separator_Shift(Features_June19_Simple_Separator
                                           , Features_June19_Simple_Shift):
    pass


class Features_June19_Full_Separator_Shift(Features_June19_Full_Separator
                                         , Features_June19_Full_Shift):
    pass

class Features_June19_Full_SPM_Separator_Shift(Features_June19_Full_SPM_Separator
                                         , Features_June19_Full_Shift):
    pass


