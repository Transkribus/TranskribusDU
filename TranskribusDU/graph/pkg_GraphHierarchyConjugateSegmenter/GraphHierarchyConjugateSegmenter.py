# -*- coding: utf-8 -*-

"""
    Train, test, predict steps for a graph-based model using a binary conjugate 
    (two classes on the primal edges)

    Structured machine learning, currently using graph-CRF or Edge Convolution Network

    Copyright NAVER(C) 2019 JL. Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""
import numpy as np

from common.trace                   import traceln
from graph.Cluster                  import ClusterList

from graph.GraphConjugate           import GraphConjugate
from graph.GraphConjugateSegmenter  import GraphConjugateSegmenter

from .I_GraphHierarchyConjugateClusterer import I_GraphHierarchyConjugateClusterer


class GraphHierarchyConjugateSegmenter(I_GraphHierarchyConjugateClusterer, GraphConjugateSegmenter):
    """
    What is independent from the input format (XML or JSON, currently)
    """
    _balancedWeights = False   #  Uniform does same or better, in general

    iHierarchyDepth = -1
    _ltsLabelByCls  = []  # mapping class index to hierarchical labels  
    
    def __init__(self, sOuputXmlAttribute=None):
        """
        a CRF model, with a name and a folder where it will be stored or retrieved from
        
        the cluster index of each object with be store in an Xml attribute of given name
        """
        I_GraphHierarchyConjugateClusterer.__init__(self, sOuputXmlAttribute)
        GraphConjugateSegmenter.__init__(self)
        
    @classmethod    
    def setHierarchyDepth(cls, N):
        cls.iHierarchyDepth = N
        cls._setlEdgeLabel(["continue"] + ["break_%d"%i for i in range(N)])
        
    @classmethod    
    def getHierarchyDepth(cls):
        return cls.iHierarchyDepth        

    # -------------------------------------------------------------------
    def parseDocLabels(self):
        """
        Parse the label of the graph from the dataset, and set the node label
        return the set of observed class (set of integers in N+)
        
        Here, no check at all, because we just need to see if two labels are the same or not.
        """
        setSeensLabels = set()
        for nd in self.lNode:
            nodeType = nd.type 
            tsLabel = tuple(nodeType.parseDocNodeLabel(nd))
            assert len(tsLabel) == self.iHierarchyDepth, "Bad hierarchy depth. Expected %d levels, got %s"%(self.iHierarchyDepth, tsLabel)
            try:
                cls = self._dClsByLabel[tsLabel]  #Here, if a node is not labelled, and no default label is set, then KeyError!!!
            except KeyError:
                cls = len(self._ltsLabelByCls)
                self._dClsByLabel[tsLabel] = cls
                self._ltsLabelByCls.append(tsLabel)
            nd.cls = cls
            setSeensLabels.add(cls)
        return setSeensLabels    
    
    def computeEdgeLabels(self):
        """
        Given the loaded graph with labeled nodes, compute the edge labels.
        
            For N levels, we have N+1 edge labels: continue, break-level-1, ... break-level-N
             ["continue"] + ["break_%d"%i for i in range(len(self.getLabelAttribute()))]

        This results in each edge having a .cls attribute.
    
        return the set of observed class (set of integers in N+)
        """
        setSeensLabels = set()
        for edge in self.lEdge:
            if edge.A.cls == edge.B.cls:
                edge.cls = 0  # continue
            else:
                for i, (Albl,Blbl) in enumerate(zip(self._ltsLabelByCls[edge.A.cls]
                                                  , self._ltsLabelByCls[edge.B.cls])
                                                ):
                    if Albl != Blbl: break
                # i+1 indicates at which level there is a break
                edge.cls = i + 1
            # print(edge.cls, self._ltsLabelByCls[edge.A.cls], self._ltsLabelByCls[edge.B.cls])
            #edge.cls = 0 if (edge.A.cls == edge.B.cls) else 1
            setSeensLabels.add(edge.cls)
        return setSeensLabels    
    
    def setDocLabelsOriginal(self, Y, TH=0.5):
        """
        Set the labels of the graph nodes from the Y _probability_ matrix
        
        Here we work level by level, forming cluster at each level
        """
        GraphConjugate.setDocLabels(self, Y) # store .cls attribute of edge objects

        Ylvl = np.zeros(shape=(Y.shape[0], 2))  # make it a continue/break probability
        # for lvl in range(self.iHierarchyDepth):
        for lvl in range(self.iHierarchyDepth-1, -1, -1):  # [self.iHierarchyDepth-1, .., 0]
            # compute a continue or break label for each edge, considering a certain level
            # continue are continue for all levels
            # at level L:
            # - a break at above levels is a break
            # - a break at level L is a break
            # - a break at below levels is a continue
            sptr = lvl+2
            Ylvl[:,0] = Y[:,0] + Y[:,sptr:].sum(axis=1)
            Ylvl[:,1] = Y[:,1:sptr].sum(axis=1)
#             print("-------", lvl)
#             print(np.hstack([Y, Ylvl]))
            self.lCluster.append(self.form_cluster(Ylvl, TH))
    
            # just the last one just generated 
            self.addClusterToDoc(lvl, self.lCluster[-1], sAlgo="GraphHierarchyConjugateSegmenter")
            traceln(" level %d: %d cluster(s) found" % (lvl, len(self.lCluster[-1])))

    
    def setDocLabelHierarchical(self, Y, TH=0.5):
        """
        proceed at deepest level as usual and then works on cluster by merging them
        """
        GraphConjugate.setDocLabels(self, Y) # store .cls attribute of edge objects

        if True:
            llCluster = self.form_cluster_hier(Y, self.iHierarchyDepth)
        elif False: 
            llCluster = self.form_cluster_hypothesis(Y, self.iHierarchyDepth) # , nBest=5)
        else:
            llCluster = self.form_cluster_hierarchical_hypothesis(Y, self.iHierarchyDepth)
        
        for lvl in range(0, self.iHierarchyDepth): 
            self.addClusterToDoc(lvl, llCluster[lvl], sAlgo="GraphHierarchyConjugateSegmenter")
            traceln(" level %d: %d cluster(s) found" % (lvl, len(llCluster[lvl])))

        self.lCluster = [ClusterList(l) for l in llCluster]
        #now iterate, but working on clusters, not nodes
        return self.lCluster
    
    if True:
        # usual algo, applied at each level independently
        setDocLabels = setDocLabelsOriginal
    else:
        # forcing the nesting
        setDocLabels = setDocLabelHierarchical    
        
    @classmethod
    def setOriginal(cls):
        cls.setDocLabels = cls.setDocLabelsOriginal

    @classmethod
    def setHierarchical(cls):
        cls.setDocLabels = cls.setDocLabelHierarchical
    