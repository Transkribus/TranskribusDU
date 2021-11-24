# coding: utf8

"""
Cluster class for the storing the result of segmentation (by Conjugate graphs for now)

JL Meunier, Nov 2019


Copyright Naver 2019
"""

class ClusterList(list):
    """
    A list of Cluster
    """
    def __init__(self, lItem=list(), sAlgo="_undefined_"):
        super(ClusterList, self).__init__(lItem)
        self.sAlgo = sAlgo
    

class Cluster(set):
    """
    A cluster is a set of objects or object's id, with a fe specific attribute and methods
    """
    def __init__(self, lItem=set(), fProba=1.0):
        super(Cluster, self).__init__(lItem)
        self.fProba = fProba
        

def test_ClusterList():
    cl = ClusterList()
    assert cl == []

    cl = ClusterList([1, 99, 23], "toto")
    assert cl == [1, 99, 23]
    
    assert cl.sAlgo == "toto"
    
    cl.append(0)
    assert cl == [1, 99, 23, 0]
    cl.sort()
    assert cl == [0, 1, 23, 99]
    
def test_ClusterList2():
    cl = ClusterList()
    assert cl == []

    cl = ClusterList([[1, 99, 23], [1,2,3], [55]], "bibi")
    assert cl == [[1, 99, 23], [1,2,3], [55]]
    assert cl[1] == [1,2,3]
    
    assert cl.sAlgo == "bibi"
            
def test_Cluster():
    c = Cluster()
    assert c == set([])

    c = Cluster([1, 99, 23])
    assert c == set([1, 23, 99, 23])
    
    assert c.fProba == 1.0
    
    c.add(0)
    assert c == set([1, 23, 99, 0])
    