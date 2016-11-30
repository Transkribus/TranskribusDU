# -*- coding: utf-8 -*-

'''
Created on 23 Nov 2016

@author: meunier
'''

from xml_formats.PageXml import PageXml


def test_custom():
    assert PageXml.parse_custom_attr("")    == {}
    assert PageXml.parse_custom_attr(" ")   == {}
    assert PageXml.parse_custom_attr("   ") == {}

    assert PageXml.parse_custom_attr("a {x:1;}")    == { 'a': { 'x':'1' } }
    assert PageXml.parse_custom_attr(" a {x:1;}")   == { 'a': { 'x':'1' } }
    assert PageXml.parse_custom_attr("a {x:1;} ")   == { 'a': { 'x':'1' } }
    assert PageXml.parse_custom_attr(" a {x:1;} ")  == { 'a': { 'x':'1' } }
    assert PageXml.parse_custom_attr("a {x:1 ;}")   == { 'a': { 'x':'1' } }
    assert PageXml.parse_custom_attr("a {x:1 ; }")  == { 'a': { 'x':'1' } }
    assert PageXml.parse_custom_attr("a { x:1 ; }") == { 'a': { 'x':'1' } }
    
    assert PageXml.parse_custom_attr("a{x:1;}")     == { 'a': { 'x':'1' } }
    assert PageXml.parse_custom_attr("a{x:1 ;}")    == { 'a': { 'x':'1' } }
    assert PageXml.parse_custom_attr("a{x:1 ; }")   == { 'a': { 'x':'1' } }
    assert PageXml.parse_custom_attr("a{ x:1 ; }")  == { 'a': { 'x':'1' } }
    
    assert PageXml.parse_custom_attr("a,b{x:1;}")       == { 'a': { 'x':'1' }, 'b': { 'x':'1' } }
    assert PageXml.parse_custom_attr("a, b{x:1 ;}")     == { 'a': { 'x':'1' }, 'b': { 'x':'1' } }
    assert PageXml.parse_custom_attr("a , b{x:1 ; }")   == { 'a': { 'x':'1' }, 'b': { 'x':'1' } }
    assert PageXml.parse_custom_attr("a ,b{ x:1 ; }")   == { 'a': { 'x':'1' }, 'b': { 'x':'1' } }
    assert PageXml.parse_custom_attr("a ,b { x:1 ; }")   == { 'a': { 'x':'1' }, 'b': { 'x':'1' } }
    
    assert PageXml.parse_custom_attr("a { x:1 ; y:2 }")   == { 'a': { 'x':'1', 'y':'2'} }
    assert PageXml.parse_custom_attr("a,b { x:1 ; y:2 }")   == { 'a': { 'x':'1', 'y':'2'}, 'b': { 'x':'1', 'y':'2'} }

    assert PageXml.parse_custom_attr("a {}")    == { 'a': { } }

    assert PageXml.parse_custom_attr("readingOrder {index:4;} structure {type:catch-word;}") == { 'readingOrder': { 'index':'4' }, 'structure':{'type':'catch-word'} }

def test_malformed_custom():
    import pytest
    with pytest.raises(ValueError): PageXml.parse_custom_attr("a {x1;}")
    with pytest.raises(ValueError): PageXml.parse_custom_attr("a x1;}")
    with pytest.raises(ValueError): PageXml.parse_custom_attr("a { x1;")
    with pytest.raises(ValueError): PageXml.parse_custom_attr("a { x1 }")
    
    #with pytest.raises(ValueError): PageXml.parse_custom_attr("a { x:1 }")  #should it fail?
    assert PageXml.parse_custom_attr("a { x:1  2}") == {'a': {'x': '1  2'}}

    #with pytest.raises(ValueError): PageXml.parse_custom_attr("a { x:1  2}")#should it fail? (or do we allow spaces in names or values?)
    assert PageXml.parse_custom_attr("  a b   {   x y : 1  2  }") == {'a b': {'x y': '1  2'}}
    