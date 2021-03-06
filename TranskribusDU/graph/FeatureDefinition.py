# -*- coding: utf-8 -*-
"""
    Feature Definition
    
    Sub-class it and specialize getTransformer and clean_tranformers
    
    Copyright Xerox(C) 2016 JL. Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""

import math

from common.trace import traceln


class FeatureDefinition:
    """
    A class to sub-class to define which features from a Tranformer class, you want for node and edges
    """
    def __init__(self, nbClass=None, node_transformer=None, edge_transformer=None):
        self.nbClass = nbClass  #number of node classes (also called 'labels', and 'states' in pystruct)
        self._node_transformer = node_transformer
        self._edge_transformer = edge_transformer
        
    def setTransformers(self, node_transformer, edge_transformer):
        self._node_transformer = node_transformer
        self._edge_transformer = edge_transformer
        
    def getTransformers(self):
        """
        return (node transformer, edge transformer)
        """
        return self._node_transformer, self._edge_transformer
            
    def fitTranformers(self, lGraph,lY=None):
        """
        Fit the transformers using the graphs
        return True
        """
        lAllNode = [nd for g in lGraph for nd in g.lNode]
        self._node_transformer.fit(lAllNode,lY)
        del lAllNode #trying to free the memory!
        
        lAllEdge = [edge for g in lGraph for edge in g.lEdge]
        self._edge_transformer.fit(lAllEdge,lY)
        del lAllEdge
        
        return True

    def cleanTransformers(self):
        """
        Some extractors/transfomers keep a large state in memory , which is not required in "production".
        This method must clean this useless large data
        
        For instance: the TFIDF transformers are keeping the stop words => huge pickled file!!!
        """
        for _trnsf in self.getTransformers():
            try: 
                _trnsf.cleanTransformers()
            except Exception as e:
                traceln("Cleaning warning: ", e) 
        return None

    def _getTypeNumber(self, kwargs):
        """
        Utility function. In some case the __init__ method gets a dictionary of length N + N^2
            (N config for unary extractor, N^2 config for pairwise)
        Here we compute N from the dictionary length. ^^
        """
        return int(round(math.sqrt( len(kwargs) + 1/4.0)-0.5, 0))
    
