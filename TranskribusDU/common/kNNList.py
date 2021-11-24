# -*- coding: utf-8 -*-
"""
a list of at most k nearest neightbors, subclassing list

JL Meunier
Copyright Naver Labs Europe
March 2021
"""


class kNNList(list):
    """
    a list with at most k elements, the nearest ones, sorted by distance
    in case of equal distance, keep first inserted object
    
    to add to the kNNList, you must use 'append'
    to remove from it    , you must use 'remove'
    
    NOTE: list is difficult to subclass especially beause of statements like del l[1:3]
    I did not find the specialized method for this.
    """
    def __init__(self, k, *args, **kwargs):
        """
        initialize an empty list, of at most k nearest neighbors
        """
        assert k > 0
        super(list, self).__init__(*args, **kwargs)
        self.k    = k
        self.lDst = [] # sorted list of object's distance
        self.maxD = -1 

    def append(self, o, dO):
        """
        o is some object
        dO is the object distance
        append the object to the list, 
            if the list contains less than k objects 
            or 
            if the object is closer than one of the k objects in the list
        """
        assert len(self) <= self.k

        if len(self) == self.k and dO >= self.maxD: 
            return  # fast track, and make sure we must insert it somewhere
        
        # where to insert?
        i, imax = 0, len(self)
        lDst = self.lDst
        while i < imax:
            if lDst[i] > dO: break
            i += 1
        # insert
        self.insert(i, o)
        lDst.insert(i, dO)
        # shorten
        del self[self.k:]
        del lDst[self.k:]
        # new max distance
        self.maxD = lDst[-1]
    
    def remove(self, i, j):
        """
        equivalent of del l[i:j]
        """
        del self     [i:j]
        del self.lDst[i:j] 
        if len(self) > 0:
            self.maxD = self.lDst[-1]
        else:
            self.maxD = -1

    def zipOD(self):
        """
        return an iterator of the list of tuples [(object, distance), ...] sorted by increasing distance
        """
        return zip(self, self.lDst)

    def getDistance(self, i):
        """
        return the distance associated to object at index i
        """
        return self.lDst[i]
    
    def getMinMaxDistance(self):
        """
        get minimum and maximum distances in the list, as a tuple
        raises ValueError if list is empty
        """
        if len(self) == 0: 
            raise ValueError("Empty kNNList")
        return (self.lDst[0], self.maxD)
    
    def sDebug(self):
        return "k=%d l=%s lDst=%s maxDst=%s" %(self.k, self, self.lDst, self.maxD)


def test_kNNList():
    k = 2
    l = kNNList(k)
    assert len(l) == 0
    
    l.append(11, 1.1)
    assert l == [11], l
    l.append(22, 2.2)
    assert l == [11, 22]
    l.append(33, 3.3)
    assert l == [11, 22]
    l.append(3, 0.3)
    #print(l, l.lDst, l.maxD)
    assert l == [3, 11]
    l.append(33, 3.3)
    assert l == [3, 11]
    l.append(3.1, 0.3)
    assert l == [3, 3.1]
    l.append(4, 0.4)
    assert l == [3, 3.1]
    l.append(2, 0.2)
    assert l == [2, 3]
    l.append(2.5, 0.25)
    assert l == [2, 2.5]
    assert l.getDistance(0) == 0.2
    assert l.getDistance(1) == 0.25
    assert l.getMinMaxDistance() == (0.2,0.25)
    assert list(l.zipOD()) == [(2, 0.2), (2.5, 0.25)]
    
    k = 1
    l = kNNList(k)
    assert len(l) == 0
    l.append(11, 1.1)
    assert l == [11], l
    l.append(22, 2.2)
    assert l == [11]
    l.append(33, 3.3)
    assert l == [11]
    l.append(3, 0.3)
    #print(l, l.lDst, l.maxD)
    assert l == [3]
    l.append(33, 3.3)
    assert l == [3]
    l.append(3.1, 0.3)
    assert l == [3]
    l.append(4, 0.4)
    assert l == [3]
    l.append(2, 0.2)
    assert l == [2]
    l.append(2.5, 0.25)
    assert l == [2]
 
    k = 3
    l = kNNList(k)
    assert len(l) == 0
    
    l.append(11, 1.1)
    assert l == [11], l
    l.append(22, 2.2)
    assert l == [11, 22]
    l.append(33, 3.3)
    assert l == [11, 22, 33]
    l.append(3, 0.3)
    #print(l, l.lDst, l.maxD)
    assert l == [3, 11, 22]
    l.append(33, 3.3)
    assert l == [3, 11, 22]
    l.append(3.1, 0.3)
    assert l == [3, 3.1, 11]
    l.append(4, 0.4)
    assert l == [3, 3.1, 4]
    l.append(2, 0.2)
    assert l == [2, 3, 3.1]
    l.append(2.5, 0.25)
    assert l == [2, 2.5, 3]

    l.remove(1,2)
    assert l == [2, 3]
    
    l.remove(0,2)
    assert l == []

    l.append(10, 1)
    l.append(20, 2)
    l.append(40, 4)
    assert l == [10, 20, 40]
    l.remove(2,3)
    assert l == [10, 20]
    l.append(30, 3)
    assert l == [10, 20, 30]
    
    assert l.getDistance(0) == 1
    assert l.getDistance(1) == 2
    assert l.getDistance(2) == 3
    assert l.getMinMaxDistance() == (1,3)
