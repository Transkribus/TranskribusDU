# -*- coding: utf-8 -*-

"""
    Computing the graph for a Factorial MultiPageXml document

    Copyright Naver(C) 2018 JL. Meunier


    
    
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

