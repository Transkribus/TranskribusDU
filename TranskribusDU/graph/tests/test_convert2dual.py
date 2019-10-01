'''
Created on 5 avr. 2019

@author: meunier
'''

import numpy as np

from graph.Graph import Graph 


def test_one_edge():
    
    o = Graph()
    
    # 2 nodes linked by 1 edge
    nf = np.array([
            [0, 0]
          , [1, 11]
          ])
    e = np.array([
            [0, 1]
         ])
    ef = np.array([
            [-0]
        ])
    
    X = (nf, e, ef)
    
    Xd = o.convert_X_to_LineDual(X)
    nfd, ed, efd = Xd
    assert (nfd == ef).all()
    assert (ed == np.array([
        ])).all()
    assert (efd== np.array([
        ])).all()
        

def test_two_edge():
    
    o = Graph()
    
    # 2 nodes linked by 1 edge
    nf = np.array([
            [0, 0]
          , [1, 11]
          , [2, 22]
          ])
    e = np.array([
            [0, 1]
          , [1, 2]
         ])
    ef = np.array([
            [-0]
          , [-1]
        ])
    
    X = (nf, e, ef)
    
    Xd = o.convert_X_to_LineDual(X)
    nfd, ed, efd = Xd
    assert (nfd == ef).all()
    assert (ed == np.array([
            [0, 1]
        ])).all()
    assert (efd== np.array([
            [1, 11]
        ])).all()
    
def test_three_edge():
    
    o = Graph()
    
    # 2 nodes linked by 1 edge
    nf = np.array([
            [0, 0]
          , [1, 11]
          , [2, 22]
          ])
    e = np.array([
            [0, 1]
          , [1, 2]
          , [2, 0]
         ])
    ef = np.array([
            [-0]
          , [-1]
          , [-2]
        ])
    
    X = (nf, e, ef)
    
    Xd = o.convert_X_to_LineDual(X)
    nfd, ed, efd = Xd
    assert (nfd == ef).all()
    assert (ed == np.array([  [0, 1]
                            , [0, 2]
                            , [1, 2]
                            ])).all(), ed
    assert (efd== np.array([  [1, 11]
                            , [0, 0]
                            , [2, 22]
                            ])).all(), efd

def test_three_edge_and_lonely_node():
    
    o = Graph()
    
    # 2 nodes linked by 1 edge
    nf = np.array([
            [0, 0]
          , [1, 11]
          , [2, 22]
          , [9, 99] #lonely node
          ])
    e = np.array([
            [0, 1]
          , [1, 2]
          , [2, 0]
         ])
    ef = np.array([
            [-0]
          , [-1]
          , [-2]
        ])
    
    X = (nf, e, ef)
    
    Xd = o.convert_X_to_LineDual(X)
    nfd, ed, efd = Xd
    assert (nfd == ef).all()
    assert (ed == np.array([  [0, 1]
                            , [0, 2]
                            , [1, 2]
                            ])).all(), ed
    assert (efd== np.array([  [1, 11]
                            , [0, 0]
                            , [2, 22]
                            ])).all(), efd
 

def test_basic_numpy_stuff():
    # to make sure I do duplicate and revert edges properly...
    
    a = np.array(range(6)).reshape((3,2))
    refaa = np.array([
       [0, 1],
       [2, 3],
       [4, 5],
       [1, 0],
       [3, 2],
       [5, 4]])
    assert (np.vstack((a, a[:,[1,0]])) == refaa).all()
    
    
    
    
    
    
    
    
    