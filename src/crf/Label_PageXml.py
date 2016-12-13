# -*- coding: utf-8 -*-

"""
    Reading and setting the labels of a PageXml document

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

from xml_formats.PageXml import PageXml
 
class Label_PageXml:

    #where the labels can be found in the data
    sCustAttr_STRUCTURE     = "structure"
    sCustAttr2_TYPE         = "type"
    
    def __init__(self): 
        pass
        
    def parseDomNodeLabel(self, domnode, defaultCls=None):
        """
        Parse and set the graph node label and return its class index
        """
        try:
            sLabel = PageXml.getCustomAttr(domnode, self.sCustAttr_STRUCTURE, self.sCustAttr2_TYPE)
        except KeyError:
            sLabel = defaultCls
        return sLabel

    def setDomNodeLabel(self, domnode, sLabel):
        PageXml.setCustomAttr(domnode, self.sCustAttr_STRUCTURE, self.sCustAttr2_TYPE, sLabel)
        return sLabel

# --- AUTO-TESTS --------------------------------------------------------------------------------------------
def test_getset():
    import libxml2
    sXml = """
            <TextRegion type="page-number" id="p1_region_1471502505726_2" custom="readingOrder {index:9;} structure {type:page-number;}">
                <Coords points="972,43 1039,43 1039,104 972,104"/>
            </TextRegion>
            """
    doc = libxml2.parseMemory(sXml, len(sXml))
    nd = doc.getRootElement()
    obj = Label_PageXml()
    assert obj.parseDomNodeLabel(nd) == 'page-number'
    assert obj.parseDomNodeLabel(nd, "toto") == 'page-number'
    assert obj.setDomNodeLabel(nd, "index") == 'index'
    assert obj.parseDomNodeLabel(nd, "toto") == 'index'
    doc.freeDoc()

def test_getset_default():
    import libxml2
    sXml = """
            <TextRegion type="page-number" id="p1_region_1471502505726_2" custom="readingOrder {index:9;} ">
                <Coords points="972,43 1039,43 1039,104 972,104"/>
            </TextRegion>
            """
    doc = libxml2.parseMemory(sXml, len(sXml))
    nd = doc.getRootElement()
    obj = Label_PageXml()
    assert obj.parseDomNodeLabel(nd) == None
    assert obj.parseDomNodeLabel(nd, "toto") == 'toto'
    assert obj.setDomNodeLabel(nd, "index") == 'index'
    assert obj.parseDomNodeLabel(nd, "toto") == 'index'
    doc.freeDoc()

