# -*- coding: utf-8 -*-

"""
    Computing the graph for a MultiPageXml document

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
from __future__ import absolute_import
from __future__ import  print_function
from __future__ import unicode_literals

from .Graph import Graph
from .Page import Page

class Graph_DSXml(Graph):
    '''
    Computing the graph for a MultiPageXml document

        USAGE:
        - call parseFile to load the DOM and create the nodes and edges
        - call detachFromDOM before freeing the DOM
    '''
    #Namespace, of PageXml, at least
    dNS = {}
    
    #How to list the pages of a (Multi)PageXml doc
    sxpPage     = "//PAGE"
    
    sXmlFilenamePattern = "*_ds.xml"    #how to find the DS XML files

    def __init__(self, lNode = [], lEdge = []):
        Graph.__init__(self, lNode, lEdge)

    # ---------------------------------------------------------------------------------------------------------        
    def _iter_Page_DomNode(self, doc):
        """
        Parse a Multi-pageXml DOM, by page

        iterator on the DOM, that returns per page:
            page-num (int), page object, page dom node
        
        """
        assert self.sxpPage, "CONFIG ERROR: need an xpath expression to enumerate PAGE elements"
        lNdPage = doc.getroot().xpath(self.sxpPage,
                                      namespaces=self.dNS)   #all pages
        
        pnum = 0
        pagecnt = len(lNdPage)
        for ndPage in lNdPage:
            pnum += 1
            iPageWidth  = int( float(ndPage.get("width")) )   #note: float in XML!!
            iPageHeight = int( float(ndPage.get("height")) )  #note: float in XML!!
            page = Page(pnum, pagecnt, iPageWidth, iPageHeight, cls=None, domnode=ndPage, domid=ndPage.fet("id"))
            yield (pnum, page, ndPage)
            
        raise StopIteration()        
        
