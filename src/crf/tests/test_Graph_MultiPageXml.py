# -*- coding: utf-8 -*-


'''
Some ad-hoc test - not very easy to reproduce, I'm afraid

Created on 1 Dec 2016

@author: meunier
'''
import os

import libxml2

import crf.Graph_MultiPageXml_TextRegion as Graph_MultiPageXml_TextRegion
import crf.Edge as Edge
import xml_formats.PageXml as PageXml


def test_RectangleFitting():
    #--- THE TRICK --------------
    Graph_MultiPageXml_TextRegion.TEST_getPageXmlBlock = True
    
    filename = os.path.join(os.path.dirname(__file__), "7749.mpxml")
    
    doc = libxml2.parseFile(filename)
    
    dNS = {"pc":PageXml.PageXml.NS_PAGE_XML}
    sxpPage     = "//pc:Page"
    sxpBlock    = ".//pc:TextRegion"
    sxpTextual  = "./pc:TextEquiv"             #CAUTION redundant TextEquiv nodes! 
        
    obj = Graph_MultiPageXml_TextRegion.Graph_MultiPageXml()

    
    #load the block of each page, keeping the list of blocks of previous page
    lPrevPageNode = None

    for (pnum, lPageNode) in obj._iter_PageXml_Nodes(doc, dNS, sxpPage, sxpBlock, sxpTextual):
    
        obj.lNode.extend(lPageNode)
        
        lPageEdge = Edge.Edge.computeEdges(lPrevPageNode, lPageNode)
        
        obj.lEdge.extend(lPageEdge)
        print "\tPage %5d    %6d nodes    %7d edges" %(pnum, len(lPageNode), len(lPageEdge))
        
        nc = 20
        fmt = "%%s p%%s %%%ss  --->  p%%s %%%ss"%(nc, nc)
        for edge in lPageEdge:
            sa, sb = edge.A.text, edge.B.text
            sa, sb = sa[:min(len(sa), nc)], sb[:min(len(sb), nc)]
            assert edge.A.node.prop("id") != edge.B.node.prop("id")
            print fmt%(edge.__class__.__name__[0], edge.A.pnum, sa, edge.B.pnum, sb)
        lPrevPageNode = lPageNode
    
    sOut = "TEST_getPageXmlBlock.mpxml"
    
    assert os.path.exists(sOut)
    
    
if __name__ == "__main__":
    test_RectangleFitting()
    
    