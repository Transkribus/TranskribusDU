# -*- coding: utf-8 -*-

"""
    Segment masking

    Copyright Naver Labs Europe(C) 2019 JL. Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""

def applyMask(lView, lViewMask):
    """
    a view is a segment (a,b)
    a mask is a segment (c,d)
    
    when you apply a mask to a segment, you get either:
    - nothing
    - a subview (which maybe the original view if the mask does not match)
    
    when you apply several mask, you get nothing or a set of subviews
    
    so applyMasks, get a list of views, and a list of mask
    and return a list of views
    
    Assumes the input views do not overlap each other
    Garanties that the output view do not overlap each other
    """
    for a,b in lView: assert a < b, "invalid view: %s, %s" %(a,b)
    # apply each mask in turn
    for (c,d) in lViewMask:
        # assert c < d, "invalid mask: %s, %s" %(c,d)
        if c >= d: continue
        lNewView = list()
        for (a,b) in lView:
            _left, _right = max(a, c), min(b, d)
            if _left < _right:
                # overlap!
                if (_right - _left) < (b - a):
                    # partially masked
                    if a      < _left: lNewView.append( (a, _left) )
                    if _right <     b: lNewView.append( (_right, b) )
                # else , since entirely masked, forget it!
            else:
                # keep it as it is
                lNewView.append( (a,b) )
        lView = lNewView
        if not lView: break  # stop if the view is empty!
    return lView


def applyMask2(lView, lViewMask):
    """
    Same as applyMask, aprt its return value
    
    Return: the remaining view width, a list of views
    
    Assumes the input views do not overlap each other
    Garanties that the output view do not overlap each other
    """
    # ok , we know it works!
    # for a,b in lView: assert a <= b, "invalid view: %s, %s" %(a,b)

    ovrl = 0  # total overlap with the masks    
    # apply each mask in turn
    for (c,d) in lViewMask:
        # assert c < d, "invalid mask: %s, %s" %(c,d)
        if c >= d: continue
        lNewView = list()
        for (a,b) in lView:
            _left, _right = max(a, c), min(b, d)
            if _left < _right:
                # overlap!
                if (_right - _left) < (b - a):
                    # partially masked
                    if a      < _left: lNewView.append( (a, _left) )
                    if _right <     b: lNewView.append( (_right, b) )
                # else , since entirely masked, forget it!
                ovrl += (_right - _left)
            else:
                # keep it as it is
                # filter our when a == b
                #if a != b: 
                lNewView.append( (a,b) )
        lView = lNewView
        if not lView: break  # stop if the view is empty!

    return lView, ovrl


def test_applyMask():
    
    def test(lV, lM, lR):
        lNV = applyMask(lV, lM)
        assert lNV == lR, " ".join(("View=", str(lV), "  mask=", str(lM), " -> ", str(lNV), "<>", str(lR)))
        return lNV
    
    lView = [(1, 10)]
    
    assert applyMask(lView, []) == lView
    test(lView, [(2,6)], [(1,2), (6, 10)])
    test(lView, [(2,10)], [(1,2)])
    test(lView, [(1,6)], [(6, 10)])
    
    test(lView, [(2,4), (4,5)], [(1,2), (5, 10)])
    test(lView, [(2,4), (3,5)], [(1,2), (5, 10)])
    test(lView, [(2,4), (1,5)], [(5, 10)])
    
    test(lView, [(2,4), (5,7)], [(1,2), (4,5), (7,10)])
    test(lView, [(2,4), (5,7), (3,5)], [(1,2), (7,10)])
    
    test( [], [], [])
    test( [], [(1,2)], [])
    test( [], [(2,4), (4,5)], [])
    test( [(1,2)], [], [(1,2)])
    
def test_applyMask2():
    
    def test(lV, lM, lR, ovrR):
        lNV, ovr = applyMask2(lV, lM)
        assert lNV == lR, " ".join(("View=", str(lV), "  mask=", str(lM), " -> ", str(lNV), "<>", str(lR)))
        assert ovr == ovrR, (ovr, " while expected ", ovrR)
        return lNV, ovr
    
    lView = [(1, 10)]
    
    assert applyMask2(lView, []) == (lView, 0)
    test(lView, [(2,6)], [(1,2), (6, 10)], 4)
    test(lView, [(2,10)], [(1,2)], 8)
    test(lView, [(1,6)], [(6, 10)], 5)
    
    test(lView, [(2,4), (4,5)], [(1,2), (5, 10)], 3)
    test(lView, [(2,4), (3,5)], [(1,2), (5, 10)], 3)
    test(lView, [(2,4), (1,5)], [(5, 10)], 4)
    
    test(lView, [(2,4), (5,7)], [(1,2), (4,5), (7,10)], 4)
    test(lView, [(2,4), (5,7), (3,5)], [(1,2), (7,10)], 5)
    
    test( [], [], [], 0)
    test( [], [(1,2)], [], 0)
    test( [], [(2,4), (4,5)], [], 0)
    test( [(1,2)], [], [(1,2)], 0)

    
#test_applyMask()
#test_applyMask2()

    