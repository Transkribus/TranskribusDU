# -*- coding: utf-8 -*-

'''
Created on 23 Nov 2016

@author: meunier
'''
import pytest

from xml_formats.PageXml import PageXml


def test_custom():
    assert PageXml.parseCustomAttr("")    == {}
    assert PageXml.parseCustomAttr(" ")   == {}
    assert PageXml.parseCustomAttr("   ") == {}

    assert PageXml.parseCustomAttr("a {x:1;}")    == { 'a': { 'x':'1' } }
    assert PageXml.parseCustomAttr(" a {x:1;}")   == { 'a': { 'x':'1' } }
    assert PageXml.parseCustomAttr("a {x:1;} ")   == { 'a': { 'x':'1' } }
    assert PageXml.parseCustomAttr(" a {x:1;} ")  == { 'a': { 'x':'1' } }
    assert PageXml.parseCustomAttr("a {x:1 ;}")   == { 'a': { 'x':'1' } }
    assert PageXml.parseCustomAttr("a {x:1 ; }")  == { 'a': { 'x':'1' } }
    assert PageXml.parseCustomAttr("a { x:1 ; }") == { 'a': { 'x':'1' } }
    
    assert PageXml.parseCustomAttr("a{x:1;}")     == { 'a': { 'x':'1' } }
    assert PageXml.parseCustomAttr("a{x:1 ;}")    == { 'a': { 'x':'1' } }
    assert PageXml.parseCustomAttr("a{x:1 ; }")   == { 'a': { 'x':'1' } }
    assert PageXml.parseCustomAttr("a{ x:1 ; }")  == { 'a': { 'x':'1' } }
    
    assert PageXml.parseCustomAttr("a,b{x:1;}")       == { 'a': { 'x':'1' }, 'b': { 'x':'1' } }
    assert PageXml.parseCustomAttr("a, b{x:1 ;}")     == { 'a': { 'x':'1' }, 'b': { 'x':'1' } }
    assert PageXml.parseCustomAttr("a , b{x:1 ; }")   == { 'a': { 'x':'1' }, 'b': { 'x':'1' } }
    assert PageXml.parseCustomAttr("a ,b{ x:1 ; }")   == { 'a': { 'x':'1' }, 'b': { 'x':'1' } }
    assert PageXml.parseCustomAttr("a ,b { x:1 ; }")   == { 'a': { 'x':'1' }, 'b': { 'x':'1' } }
    
    assert PageXml.parseCustomAttr("a { x:1 ; y:2 }")   == { 'a': { 'x':'1', 'y':'2'} }
    assert PageXml.parseCustomAttr("a,b { x:1 ; y:2 }")   == { 'a': { 'x':'1', 'y':'2'}, 'b': { 'x':'1', 'y':'2'} }

    assert PageXml.parseCustomAttr("a {}")    == { 'a': { } }

    assert PageXml.parseCustomAttr("readingOrder {index:4;} structure {type:catch-word;}") == { 'readingOrder': { 'index':'4' }, 'structure':{'type':'catch-word'} }

def test_malformed_custom():
    with pytest.raises(ValueError): PageXml.parseCustomAttr("a {x1;}")
    with pytest.raises(ValueError): PageXml.parseCustomAttr("a x1;}")
    with pytest.raises(ValueError): PageXml.parseCustomAttr("a { x1;")
    with pytest.raises(ValueError): PageXml.parseCustomAttr("a { x1 }")
    
    #with pytest.raises(ValueError): PageXml.parseCustomAttr("a { x:1 }")  #should it fail?
    assert PageXml.parseCustomAttr("a { x:1  2}") == {'a': {'x': '1  2'}}

    #with pytest.raises(ValueError): PageXml.parseCustomAttr("a { x:1  2}")#should it fail? (or do we allow spaces in names or values?)
    assert PageXml.parseCustomAttr("  a b   {   x y : 1  2  }") == {'a b': {'x y': '1  2'}}
    
def test_getsetCustomAttr():
    import libxml2
    sXml = """
            <TextRegion type="page-number" id="p1_region_1471502505726_2" custom="readingOrder {index:9;} structure {type:page-number;}">
                <Coords points="972,43 1039,43 1039,104 972,104"/>
            </TextRegion>
            """
    doc = libxml2.parseMemory(sXml, len(sXml))
    nd = doc.getRootElement()
    assert PageXml.getCustomAttr(nd, "readingOrder", "index") == '9'
    assert PageXml.setCustomAttr(nd, "readingOrder", "index", 99) == 99
    assert PageXml.getCustomAttr(nd, "readingOrder", "index") == '99'

    assert PageXml.getCustomAttr(nd, "readingOrder") == {'index':'99'}
    
    assert PageXml.setCustomAttr(nd, "readingOrder", "toto", "zou") == "zou"
    assert PageXml.getCustomAttr(nd, "readingOrder", "toto") == 'zou'

    with pytest.raises(KeyError): PageXml.getCustomAttr(nd, "readingOrder", "axiste_pas")
    with pytest.raises(KeyError): PageXml.getCustomAttr(nd, "axiste_pas_non_plus", "axiste_pas")
    
