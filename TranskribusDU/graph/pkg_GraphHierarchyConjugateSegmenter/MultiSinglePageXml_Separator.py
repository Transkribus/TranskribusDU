# -*- coding: utf-8 -*-

"""
    Multi single PageXml graph in conjugate mode, exploting SeparatorRegion
    as additional edge features

    Copyright NAVER(C) 2019 
    
    2019-08-20    JL. Meunier
"""

from graph.PageXmlSeparatorRegion           import PageXmlSeparatorRegion
from .MultiSinglePageXml                    import MultiSinglePageXml 


class MultiSinglePageXml_Separator(PageXmlSeparatorRegion, MultiSinglePageXml):
    """
    Multi single PageXml graph in conjugate mode, exploting SeparatorRegion
    as additional edge features
    """
    def __init__(self):
        super(MultiSinglePageXml_Separator, self).__init__()

    @classmethod
    def loadGraphs(cls
                   , cGraphClass          # graph class (must be subclass)
                   , lsFilename
                   , bNeighbourhood=True
                   , bDetach=False
                   , bLabelled=False
                   , iVerbose=0):
        o = MultiSinglePageXml.loadGraphs(cGraphClass          # graph class (must be subclass)
                                       , lsFilename
                                       , bNeighbourhood=bNeighbourhood
                                       , bDetach=bDetach
                                       , bLabelled=bLabelled
                                       , iVerbose=iVerbose)
        PageXmlSeparatorRegion.clean_cache()
        return o