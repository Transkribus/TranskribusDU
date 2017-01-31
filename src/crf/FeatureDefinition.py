# -*- coding: utf-8 -*-

"""
    Feature Definition
    
    Sub-class it and specialize getTransformer and clean_tranformers
    
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

class FeatureDefinition:
    """
    A class to sub-class to define which features from a Tranformer class, you want for node and edges
    """
    def __init__(self): pass

    def getTransformers(self):
        """
        return (node transformer, edge transformer)
        """
        raise Exception("Method must be overridden")

        #TODO Maybe rename to -- fitTransformer anod tranf for
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
        return None
    
