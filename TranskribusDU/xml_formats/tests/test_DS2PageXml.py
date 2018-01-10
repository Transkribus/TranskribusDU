# -*- coding: utf-8 -*-
"""
    test DS2PageXml convertor
    @author:déjean
"""
import pytest
import os
import libxml2
from xml_formats.DS2PageXml import DS2PageXMLConvertor
from xml_formats.PageXml import MultiPageXml

def test_DS2PageXmlConversion():
    filename = 'testDS2PageXml/RRB_MM_01_033_Jahr_1810.ds.xml'
    outputDir = '.'
    conv= DS2PageXMLConvertor()
    conv.inputFileName = filename
    doc = conv.loadDom(filename)
    lPageXmlDocs = conv.run(doc)
    mp = MultiPageXml()
    newDoc = mp.makeMultiPageXmlMemory(map(lambda (x,y):x,lPageXmlDocs))
    newDoc.saveFormatFileEnc("testDS2PageXml/RRB_MM_01_033_Jahr_1810.mpxml", "UTF-8",True)

#     res= conv.storePageXmlSetofFiles(lPageXmlDocs)
#     print 'test:', True if res == 0  else False
    
if __name__ == "__main__":
#     test_setMetadata()
    test_DS2PageXmlConversion()