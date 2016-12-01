# -*- coding: utf-8 -*-


'''
Some ad-hoc test - not very easy to reproduce, I'm afraid

Created on 1 Dec 2016

@author: meunier
'''
import os

import libxml2

import crf.Graph_MultiPageXml as Graph_MultiPageXml
import xml_formats.PageXml as PageXml


def test_RectangleFitting():
    Graph_MultiPageXml.TEST_getPageXmlBlock = True
    filename = os.path.join(os.path.dirname(__file__), "7749.mpxml")
    
    doc = libxml2.parseFile(filename)
    dNS = {"pc":PageXml.PageXml.NS_PAGE_XML}
    sxpPage     = "//pc:Page"
    sxpBlock    = ".//pc:TextRegion"
    sxpTextual  = "./pc:TextEquiv"             #CAUTION redundant TextEquiv nodes! 
        
    obj = Graph_MultiPageXml.Graph_MultiPageXml()
    lNode = obj.getPageXmlBlocks(doc, dNS, sxpPage, sxpBlock, sxpTextual)

    sOut = "TEST_getPageXmlBlock.mpxml"
    
    assert os.path.exists(sOut)
    
    
if __name__ == "__main__":
    test_RectangleFitting()
    
    