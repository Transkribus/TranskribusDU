# coding: utf8

"""
Convenience code to load gziped pickled files 

Main functions of an instance of Converter
- .len() gives the number of graphs
- .getGraphPageBlocks(n) -->  the list of page blocks of the Nth graph
- .getGraphXY(n)         --> the X,Y of the Nth graph
- .convert(n)            --> convert to a lightweight dictionary based graph-representation 

The order of the blocks is always the same, and consistent among the different functions.

Jean-Luc Meunier
April 2021

Copyright NAVER France, 2021
"""


import os
    
import numpy as np

from util.gzip_pkl import gzip_pickle_load


class Block:
    def __init__(self, txt, t_x1_y1_x2_y2, cls):
        self.txt = txt
        (self.x1, self.y1, self.x2, self.y2) = t_x1_y1_x2_y2
        self.cls = cls


class Converter:
    """
    the Converter needs the nodes definition file (defining the nodes of  all graphs)
    , and goes over individual graph pickle files.
    
    It is for example the file: 
    - BAR-N.1.tst.node.pkl for the graph pickled in BAR-N.1.tst_tXY
    or 
    - BAR-E.1.tst.node.pkl for the graph pickled in BAR-E.1.tst_tXY
    """    
    def __init__(self, sNodeDefFile):
        """
        sNodeDefFile if the pickle file name containing the node definition for a fold for a task
        """
        sNodeDefFile = sNodeDefFile.strip()
        self.lGraphNodes = gzip_pickle_load(sNodeDefFile)
        # lGraphNode contain for each graph a list of tuples: 
        #  txt, (x1, y1, x2, y2), lbl                

        # also compute the name of associated folder that contains each graph as a pickle
        sSfx = ".node.pkl"
        assert sNodeDefFile.endswith(sSfx), "Pass the node definition file name ending in '%s'"%sSfx
        self.sPickleFolder = sNodeDefFile[:-len(sSfx)] + "_tXY"
    
        self._iter_n = None
        
    def __len__(self):
        """
        return the number of graphs in this dataset
        """
        return len(self.lGraphNodes)

    def convert(self, n):
        """
        return the Nth graph as a dictionary:  { node: {neighbour-nodes},  ...}
        
        node identifiers are integers
        """
        X, _Y = self.getGraphXY(n)
        (NF, E, _EF) = X
        nbNode, nbEdge = NF.shape[0], E.shape[0]

        dGraph = dict()
        for i in range(nbNode): 
            dGraph[i] = set()
        for j in range(nbEdge):
            A, B = E[j]
            dGraph[A].add(B) 
            dGraph[B].add(A) 
        return dGraph
    
    def getGraphPageBlocks(self, n):
        """
        return the list of graph blocks for Nth graph (N starts at 0)
        """
        lGraphNode = self.lGraphNodes[n]
        lBlock = [ Block(txt, (x1, y1, x2, y2), lbl)  # lbl is a cluster_id for segmentation tasks
                for txt, (x1, y1, x2, y2), lbl 
                in lGraphNode
                ]
        # each block needs to know its index
        for _i, blk in enumerate(lBlock): blk._index = _i
        return lBlock

    def getGraphXY(self, n):
        """
        return the (X, Y) of the nth graph (0 is the first one)
        """
        assert n >= 0        , "Index of first graph is 0"
        assert n < len(self) , "Index of last graph is %d" % (len(self)-1)
        (X, Y) = gzip_pickle_load(os.path.join(self.sPickleFolder, "%06d.pkl"% (n+1)))
        return (X, Y)
    
 
if __name__ == "__main__":

    # load the tagging graphs, convert to 4-NN and compare to pre-computed 4-NN
    doerN = Converter("tests/N._.ecn.trn.node.pkl")
    doerE = Converter("tests/E._.ecn.trn.node.pkl")

    assert len(doerN) == len(doerE), "tagging and segmenting tasks operate on different number of graphs??"
    
    for n in range(len(doerN)):
        # load the nodes from the tagging task to get the graph structure
        (X   , Y)    = doerN.getGraphXY(n)
        # load the blocks from the segmentation task to get the cluster id
        lGraphBlock  = doerE.getGraphPageBlocks(n)
        Y = np.array([blk.cls for blk in lGraphBlock], dtype=np.int)
        
        NF, E, EF = X

        print("Graph %6d   NF:%s  E:%s  EF:%s  Y:%s" % (n    , NF.shape, E.shape, EF.shape, Y.shape))
