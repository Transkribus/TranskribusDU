# -*- coding: utf-8 -*-
"""
    test DS2PageXml convertor
    @author:déjean
"""
import pytest
import os

from xml_formats.DS2PageXml import DS2PageXMLConvertor


def test_DS2PageXmlConversion():
    filename = 'testDS2PageXml/RRB_MM_01_033_Jahr_1810.ds.xml'
    outputDir = '.'
    conv= DS2PageXMLConvertor()
    conv.inputFileName = filename
    doc = conv.loadDom(filename)
    pageXml = conv.run(doc)
#     conv.outputFileName = outputDir+os.sep+ filename[:len(".ds.xml")].xml
#     print "output files"
#     conv.writeDom(pageXml,bIndent=True)
    
    
if __name__ == "__main__":
#     test_setMetadata()
    test_DS2PageXmlConversion()