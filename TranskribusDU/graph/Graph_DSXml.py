# -*- coding: utf-8 -*-

"""
    Computing the graph for a MultiPageXml document

    Copyright Xerox(C) 2016 JL. Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""




from .Graph_DOM import Graph_DOM
from .Page import Page


class Graph_DSXml(Graph_DOM):
    '''
    Computing the graph for a MultiPageXml document
    '''
    # --- NODE TYPES and LABELS
    _lNodeType       = []       #the list of node types for this class of graph
    _bMultitype      = False    # equivalent to len(_lNodeType) > 1
    _dLabelByCls     = None     #dictionary across node types
    _dClsByLabel     = None     #dictionary across node types
    _nbLabelTot      = 0        #total number of labels

    #Namespace, of PageXml, at least
    dNS = {}
    
    #How to list the pages of a (Multi)PageXml doc
    sxpPage     = "//PAGE"

    sIN_FORMAT  = "DS_XML"   # tell here which input format is expected
    sOUTPUT_EXT = ".ds.xml"
    
    sXmlFilenamePattern = "*_ds.xml"    #how to find the DS XML files

    def __init__(self, lNode = [], lEdge = []):
        Graph_DOM.__init__(self, lNode, lEdge)

    # ---------------------------------------------------------------------------------------------------------        
    def _iter_Page_DocNode(self, doc):
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
            
        return      
        
