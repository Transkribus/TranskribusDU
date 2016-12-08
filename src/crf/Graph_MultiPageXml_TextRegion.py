# -*- coding: utf-8 -*-

"""
    Computing the graph for a MultiPageXml document, at TextRegion level

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

from Graph_MultiPageXml import Graph_MultiPageXml

class Graph_MultiPageXml_TextRegion(Graph_MultiPageXml):
    '''
    Computing the graph for a MultiPageXml document at TextRegion level
    '''

    #TASK SPECIFIC
    sxpNode     = ".//pc:TextRegion"
    sxpTextual  = "./pc:TextEquiv"             #CAUTION redundant TextEquiv nodes! 

    def __init__(self, lNode = [], lEdge = []):
        Graph_MultiPageXml.__init__(self, lNode, lEdge)

if __name__ == "__main__":
    import sys
    grph = Graph_MultiPageXml_TextRegion()
    grph.parseXmlFile(sys.argv[1], 1)
    print grph #just to show some result from the __main__