# -*- coding: utf-8 -*-

"""
    Computing the graph for a Factorial MultiPageXml document

    Copyright Naver(C) 2018 JL. Meunier

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




from ..Graph import Graph


# ------------------------------------------------------------------------------------------------------------------------------------------------
class FactorialGraph(Graph):
    """
    FactorialCRF 

    """
    
    def __init__(self, lNode = [], lEdge = []):
        Graph.__init__(self, lNode, lEdge)

