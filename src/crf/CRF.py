# -*- coding: utf-8 -*-

"""
    Computing the graph for a document
    

    Copyright Xerox(C) 2016 H. Déjean, JL. Meunier

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


class CRF_Graph:
    '''
    Computing the graph for a document
    '''
    version = "1"

    def __init__(self, params):
        '''
        Constructor
        '''
        pass

    def load(self, sDir, sFilePattern):
        """
        Process all these files and for each:
        - generate 3 numpy matrices: node-features, edges, edge-features
        - generate 1 pickle file containing metadata
        
        return the list of process filenames
        """
        raise Exception("Please specialise this method.")        