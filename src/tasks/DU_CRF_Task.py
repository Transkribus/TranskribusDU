# -*- coding: utf-8 -*-

"""
    CRF DU task core
    
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

class DU_CRF_Task:

    def getTransformers(self):
        """
        Return the node and edge Tra,nsformers
        """
        raise Exception("Method must be overridden")

    def clean_transformers(self, node_transformer, edge_transformer):
        """
        the TFIDF transformer are keeping the stop words => huge pickled file!!!
        
        Here the fix is a bit rough. There are better ways....
        JL
        """
        node_transformer.transformer_list[0][1].steps[1][1].stop_words_ = None   #is 1st in the union...
        edge_transformer.transformer_list[2][1].steps[1][1].stop_words_ = None   #is 2nd and 3rd in the union....
        edge_transformer.transformer_list[3][1].steps[1][1].stop_words_ = None        
        return node_transformer, edge_transformer
