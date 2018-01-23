# coding: utf8

'''
This code is yet again about sequentiality between two presumed page numbers


JL Meunier
March 21st, 2016

Copyright Xerox 2016
'''
from __future__ import absolute_import
from __future__ import  print_function
from __future__ import unicode_literals


class PageNumberSimpleSequenciality:
    """
    whether or not two strings could be considered as part of a page numbering sequence?
    """
    def __init__(self):
        pass

    def isPossibleSequence(self, s1, s2):
        try:
            n1 = int(s1)
            n2 = int(s2)
            return (n1 + 1) == n2
        except:
            return False
    

def test_basic():
    pns = PageNumberSimpleSequenciality()
    
    assert pns.isPossibleSequence("1", "2")
    assert not pns.isPossibleSequence("1", "3")
    assert not pns.isPossibleSequence("1", "")
    assert not pns.isPossibleSequence("1", "12")
    assert not pns.isPossibleSequence("1", "A1")
    assert not pns.isPossibleSequence("1", "1A")
    assert not pns.isPossibleSequence("1", "A")
    
#     assert pns.isPossibleSequence("A1", "A2")
#     assert not pns.isPossibleSequence("A1", "A3")
# 
#     assert pns.isPossibleSequence("A1", "B1")
#     assert not pns.isPossibleSequence("B1", "B3")
#     assert not pns.isPossibleSequence("B1", "B1B")
#     
#     assert pns.isPossibleSequence("A-1", "B-1")
#     assert pns.isPossibleSequence("B-3", "C-1")    
#     assert pns.isPossibleSequence("B-3", "C-3")    
#     assert not pns.isPossibleSequence("B-1", "B-3")    
#     assert not pns.isPossibleSequence("B-3", "C-2")    
#     assert not pns.isPossibleSequence("B-3", "C-4")    
#     
#     assert not pns.isPossibleSequence("31", "30 Sign Detail_")
    
    assert not pns.isPossibleSequence("114739", ".41147391")
    assert not pns.isPossibleSequence("", "")
    
        
