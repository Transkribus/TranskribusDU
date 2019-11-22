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

from graph.GraphConjugateSegmenter import GraphConjugateSegmenter
from graph.Cluster import ClusterList, Cluster

class GraphBinaryConjugateSegmenter(GraphConjugateSegmenter):
    """
    What is independent from the input format (XML or JSON, currently)
    """
    _balancedWeights = False   #  Uniform does same or better, in general

    lEdgeLabel  = ["continue", "break"]
    nbEdgeLabel = 2

    sOutputAttribute = "DU_cluster"
    
    def __init__(self, sOuputXmlAttribute=None):
        """
        a CRF model, with a name and a folder where it will be stored or retrieved from
        
        the cluster index of each object with be store in an Xml attribute of given name
        """
        super(GraphBinaryConjugateSegmenter, self).__init__()
        
        self.dCluster = None
        self.sClusteringAlgo = None
        
        if not sOuputXmlAttribute is None: 
            GraphBinaryConjugateSegmenter.sOutputAttribute = sOuputXmlAttribute

    def parseDocLabels(self):
        """
        Parse the label of the graph from the dataset, and set the node label
        return the set of observed class (set of integers in N+)
        
        Here, no check at all, because we just need to see if two labels are the same or not.
        """
        setSeensLabels = set()
        for nd in self.lNode:
            nodeType = nd.type 
            sLabel = nodeType.parseDocNodeLabel(nd)
            try:
                cls = self._dClsByLabel[sLabel]  #Here, if a node is not labelled, and no default label is set, then KeyError!!!
            except KeyError:
                cls = len(self._dClsByLabel)
                self._dClsByLabel[sLabel] = cls
            nd.cls = cls
            setSeensLabels.add(cls)
        return setSeensLabels    
    
    def computeEdgeLabels(self):
        """
        Given the loaded graph with labeled nodes, compute the edge labels.
        
        This results in each edge having a .cls attribute.
    
        return the set of observed class (set of integers in N+)
        """
        setSeensLabels = set()
        for edge in self.lEdge:
            edge.cls = 0 if (edge.A.cls == edge.B.cls) else 1
            setSeensLabels.add(edge.cls)
        return setSeensLabels    

    # --- Clusters -------------------------------------------------------
    def form_cluster(self, Y_proba, fThres=0.5, bAgglo=True):
        """
        Do a connected component algo
        Return a dictionary: cluster_num --> [list of node index in lNode]
        """
        
        if bAgglo:
            lCluster = self.agglomerative_clustering(0.99,Y_proba)
        else:
            # need binary edge labels
            Y = Y_proba.argmax(axis=1)
            lCluster = self.connected_component(Y, fThres)
            
        return lCluster
        
    def connected_component(self, Y, fThres):
        import sys
        recursion_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(2*recursion_limit)

        # create clusters of node based on edge binary labels
        try:
            def DFS(i):
                visited[i] = 1
                visited_index.append(i)
                for j in range(nb_node):
                    if visited[j] != 1 and ani[i][j]==1:
                        visited_index.append(j)
                        DFS(j)
                return visited_index
    
            lCluster = ClusterList([], "CC")  # CC stands for COnnected Component
            
            nb_node = len(self.lNode)
            
            # create an adjacency matrix
            ani = np.zeros(shape=(nb_node, nb_node), dtype='int64')
            for i, edge in enumerate(self.lEdge):
                if Y[i] < fThres:
                    # connected!
                    iA, iB = edge.A._index, edge.B._index
                    ani[iA,iB] = 1
                    ani[iB,iA] = 1
            
            visited = np.zeros(nb_node, dtype='int64')
            for i in range(nb_node):
                visited_index = []
                if visited[i] == 0:
                    lCluster.append(Cluster(DFS(i)))
        finally:  
            sys.setrecursionlimit(recursion_limit)
            
        return lCluster
        

    def getEdges(self,lEdges,Y_proba):
        """
            return a dictionary of edges
        """
        #lLabels=['continue','break']
        dEdges={0:{},1:{} }
        Y = Y_proba.argmax(axis=1)
        for i, edge in enumerate(lEdges):
            #type(edge) in [HorizontalEdge, VerticalEdge]:
            #cls = Y[i]
            dEdges[ Y[i] ] [(edge.A,edge.B)]= Y_proba[i,  Y[i]]
        return dEdges
    
    def distance(self,c1,c2,relmat):
        """
            compute the "distance" between an element and a cluster
            distance = nbOk, nbBad edges
            return distance
        """
        iBad = 0
        iOK  = 0
        for p in c1:
            for pp in c2:
                try: 
                    if relmat[0][(p,pp)] >= 0.5:
                        iOK += relmat[0][(p,pp)]
                    else:iBad += relmat[0][(p,pp)]
                except KeyError:#no edge
                    pass
                try: 
                    if relmat[1][(p,pp)] >= 0.5:
                        iBad += relmat[1][(p,pp)]
    #                     else:iOK += 1  # possible?
                except KeyError:#no edge
                    pass                
        return iOK,iBad
    
    def mergeCluster(self,lc,relmat):
        """
            for each cluster: compute score with all other clusters
            need to differentiate between H and V ??29/08/2019
        """
        
        lDist = {}
        for i,c in enumerate(lc):
            for j,c2 in enumerate(lc[i+1:]):
                dist = self.distance(c,c2,relmat)
                if dist != (0,0):
    #                 print(c,c2,dist)
                    if dist[0] > dist[1]:
                        lDist[(i,i+j+1)] = dist[0] - dist[1]
        # sort 
        # merge if dist
        sorted_x = sorted(lDist.items(), key=lambda v:v[1],reverse=True)
        ltbdel=[]
        lSeen=[] 
        for p,score in sorted_x:
            a=p[0];b=p[1]
            if lc[b] not in ltbdel:
                if lc[a] in lSeen or lc[b] in lSeen:
                    pass
                else:
                    lSeen.append(lc[a])
                    lSeen.append(lc[b])
                    lc[a].extend(lc[b][:])
                    ltbdel.append(lc[b])
        [lc.remove(x) for x in ltbdel]
        
        
        return lc, ltbdel !=[] 
    
    
    def assessCluster(self,c,relmat):
        """
            coherent score
        """
        iBad = 0
        iOK  = 0
        for i,p in enumerate(c):
            for pp in c[i+1:]:
                try: 
                    if relmat['continue'][(p,pp)] >= 0.5:
                        iOK += 1 #relmat['continue'][(p.getAttribute('id'),pp.getAttribute('id'))]
                    else:iBad += 1 #relmat['continue'][(p.getAttribute('id'),pp.getAttribute('id'))]
                except KeyError:#no edge
                    pass
                try: 
                    if relmat['break'][(p,pp)] >= 0.5:
                        iBad += 1 #relmat['break'][(p.getAttribute('id'),pp.getAttribute('id'))]
    #                     else:iOK += 1
                except KeyError:#no edge
                    pass                
        return iOK,iBad    
    
    def clusterPlus(self,lCluster,dEdges):
        """
            merge cluster as long as new clusters are created
        """
        bGo=True
        while bGo:
            lCluster,bGo = self.mergeCluster(lCluster,dEdges)
       
        lCluster.sAlgo = 'agglo'
        
        return lCluster
        
    def agglomerative_clustering(self,fTH,Y_proba):
        """
            fTH       : threshold used for initial connected components run
            Y_proba   : edge prediction
            Algo:   perform a cc with fTH
                    merge clusters which share coherent set of edges (iteratively)
    
            return: set of clusters
        """
        Y = Y_proba.argmax(axis=1)
        ClusterList = self.connected_component(Y,fTH)
       
        dEdges = self.getEdges(self.lEdge,Y_proba)
        lCluster = self.clusterPlus(ClusterList,dEdges)
        
        return lCluster
    


