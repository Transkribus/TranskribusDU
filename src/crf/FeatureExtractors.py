# -*- coding: utf-8 -*-

"""
    Feature extractors
    

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

class FeatureExtractors:
    def __init__(self): pass

    def getTransformers(self):
        """
        return the node and edge feature extractors
        """
        raise Exception("Method must be overridden")
        
    def fitTranformers(self, lGraph):
        """
        Fit the transformers using the graphs
        return True 
        """
        lAllNode = [nd for g in lGraph for nd in g.lNode]
        self.node_transformer.fit(lAllNode)
        del lAllNode #trying to free the memory!
        
        lAllEdge = [edge for g in lGraph for edge in g.lEdge]
        self.edge_transformer.fit(lAllEdge)
        del lAllEdge
        
        return True

    def clean_transformers(self):
        """
        Some extractors/transfomers keep a large state in memory , taht is not required in "production".
        This method must clean this useless large data
        
        For instance: the TFIDF transformers are keeping the stop words => huge pickled file!!!
        """
        return None
    
