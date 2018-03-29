# -*- coding: utf-8 -*-
"""


    XYcut.py
    
    vertical/ horizontal cuts for page elements:  

    copyright Naver Labs Europe 2018
    READ project 

"""

def mergeSegments(lSegment, iMin):
    """Take as input a list of interval on some axis,
    together with the object that contributed to this interval.
    In this module it's a textbox or an image
    Merge overlapping segments
    Return a sorted list of disjoints segments together
    with the associated objects (that is the union of the objects
    associated to the segments being merged)
    Contiguous segments closer than iMin are merged as well.
    INPUT:    [ (a,b,o)       , ...]
    or INPUT: [ (a,b, [o,...])       , ...]
    OUPUT: [ (c,d,[o,...]) , ...], min, max

    bProjOn may contain the name of the axis on which the projection has
    been done ("X" for an x-cut, "Y" for an y-cut)
    then in frontier mode , we keep smal intervals if they are coinciding
    with a frontier (e.g. a very narrow horizontal split coinciding with
    a line is kept despite it's narower than iMin
    p and q are the boundaries along the other axis of the block to cut
    """
    lMergedSegment = []
    for seg in lSegment:
        (aaux,baux,o) = seg
        lo = (o,)
        a = min(aaux,baux) #just in case...
        b = max(aaux,baux) #just in case...

        #find all overlapping or close-enough segments and merge them
        lOverlap = []
        for mseg in lMergedSegment:
            [aa,bb,loaux] = mseg
            iOver = max(a,aa) - min(b, bb) #negative means overlap
            if iOver <= iMin: #overlap or spaced by less than iMin pixel
                lOverlap.append(mseg)
            else:
                pass #nothing to merge with
                    
        if lOverlap:
            #merge the current segment with all overlapping msegments
            for aa, bb, lolo in lOverlap:
                if aa<a: a=aa
                if bb>b: b=bb
                lo = lo + tuple(lolo)
            for mseg in lOverlap:
                lMergedSegment.remove(mseg)
        #mseg = [a, b, lo]
        mseg = (a, b, tuple(lo))
        lMergedSegment.append(mseg)

    #sorted list
    lMergedSegment.sort()
    amin = lMergedSegment[0][0]
    amax = lMergedSegment[-1][1]
    return tuple(lMergedSegment), amin, amax
