# -*- coding: utf-8 -*-

"""
    Interface: Clustering function for Conjugate graph

    Structured machine learning, currently using graph-CRF or Edge Convolution Network

    Copyright NAVER(C) 2019 JL. Meunier

    Developed  for the EU project READ. The READ project has received funding 
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""
import numpy as np

from graph.Cluster      import ClusterList, Cluster
from util.cluster_node  import HypothesisClusterer


class I_GraphBinaryConjugateClusterer:
    """
    What is independent from the input format (XML or JSON, currently)
    """

    sOutputAttribute = "DU_cluster"
    
    def __init__(self, sOuputXmlAttribute=None):
        """
        a CRF model, with a name and a folder where it will be stored or retrieved from
        
        the cluster index of each object with be store in an Xml attribute of given name
        """
        if not sOuputXmlAttribute is None: 
            I_GraphBinaryConjugateClusterer.sOutputAttribute = sOuputXmlAttribute

        self.dCluster = None
        self.sClusteringAlgo = None
        self.fTH=0.5
        
    # --- Clusters -------------------------------------------------------
    def form_cluster(self, Y_proba, fThres=0.5, bAgglo=True):
        """
        Do a connected component algo
        Return a dictionary: cluster_num --> [list of node index in lNode]
        """
        if bAgglo:
            self.fTH=fThres
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
            return a dictionary of 
            {
            0 -> { (index, index) -> Y_proba_of_0 }
            1 -> { (index, index) -> Y_proba_of_1 }
            }
        """
        #lLabels=['continue','break']
        dEdges={0:{},1:{} }
        Y = Y_proba.argmax(axis=1)
        for i, edge in enumerate(lEdges):
            #type(edge) in [HorizontalEdge, VerticalEdge]:
            #cls = Y[i]
            dEdges[ Y[i] ] [(edge.A._index,edge.B._index)]= Y_proba[i,  Y[i]]
            dEdges[ Y[i] ] [(edge.B._index,edge.A._index)]= Y_proba[i,  Y[i]]
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
                    if relmat[0][(p,pp)] >= self.fTH:
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
                    lc[a] = lc[a].union(lc[b])
                    ltbdel.append(lc[b])
        #for x in ltbdel                     
        [lc.remove(x) for x in ltbdel if x in lc]
                 
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
        bGo=True;
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
        ClusterList = self.connected_component(Y_proba[:,1],1-0.99)
         
        dEdges = self.getEdges(self.lEdge,Y_proba)
        lCluster = self.clusterPlus(ClusterList,dEdges)
        return lCluster
    
    # ---------------------------------------------------------------------------------------
    # December 2020 new methods
   
    def _form_cluster_hypothesis_prepare_data(self, Y_proba, N):
        nn = len(self.lNode)
        ne = len(self.lEdge)
        assert Y_proba.shape == (ne, N+1)
        aAB = np.zeros((ne, 2), dtype=np.int)
        for i, e in enumerate(self.lEdge): aAB[i,:] = (e.A._index, e.B._index)
        return nn, ne, aAB
    
    def form_cluster_hypothesis(self, Y_proba, N, nBest=5):
        """
        run 5 hypothesis in parallel, for a single level
        """
        nn, _ne, aAB = self._form_cluster_hypothesis_prepare_data(Y_proba, N)
        
        doer = HypothesisClusterer(nn, N, None, aAB, Y_proba, nBest=nBest)
        llCluster = []
        for lvl in range(N):
            lt = doer.clusterNBest(lvl)     # list of (score, lCluster)
            lCluster = lt[0][1]             # gets first solution, and ignore its score
            llCluster.append(lCluster)
            
        return llCluster


