# -*- coding: utf-8 -*-

"""
    Train, test, predict steps for a graph-based model using a binary conjugate 
    (two classes on the primal edges)

    Structured machine learning, currently using graph-CRF or Edge Convolution Network

    Copyright NAVER(C) 2019 JL. Meunier

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""
import numpy as np
from shapely.ops import cascaded_union        
import lxml.etree as etree

from common.trace import  traceln
from xml_formats.PageXml import PageXml 
from util.Shape import ShapeLoader
from graph.Graph import Graph


class GraphBinaryConjugateSegmenter(Graph):

    _balancedWeights = False   #  Uniform does same or better, in general

    # conjugate mode
    bConjugate  = True
    lEdgeLabel  = ["continue", "break"]
    nbEdgeLabel = 2

    sXmlAttribute = "DU_cluster"
    
    def __init__(self, sOuputXmlAttribute=None):
        """
        a CRF model, with a name and a folder where it will be stored or retrieved from
        
        the cluster index of each object with be store in an Xml attribute of given name
        """
        super(GraphBinaryConjugateSegmenter, self).__init__()
        
        if not sOuputXmlAttribute is None: 
            GraphBinaryConjugateSegmenter.sXmlAttribute = sOuputXmlAttribute

    def parseDomLabels(self):
        """
        Parse the label of the graph from the dataset, and set the node label
        return the set of observed class (set of integers in N+)
        
        Here, no check at all, because we just need to see if two labels are the same or not.
        """
        setSeensLabels = set()
        for nd in self.lNode:
            nodeType = nd.type 
            sLabel = nodeType.parseDomNodeLabel(nd.node)
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

    def exploitEdgeLabels(self, Y_proba):
        """
        Do whatever is required on the (primal) graph, given the edge labels
            Y_proba is the predicted edge label probability array of shape (N_edges, N_labels)
        
        return None
        
        The node and edge indices corresponding to the order of the lNode
        and lEdge attribute of the graph object.
        
        Here we choose to set an XML attribute DU_cluster="<cluster_num>"
        """
        fThres = 0.5
        
        # create clusters of node based on edge binary labels
        Y = Y_proba.argmax(axis=1)
        dCluster = self.form_cluster(Y, fThres)

        for num, lNodeIdx in dCluster.items():
            for ndIdx in lNodeIdx:
                node = self.lNode[ndIdx]
                node.node.set(self.sXmlAttribute, "%d"%num)
        traceln(" %d cluster(s) found. See @%s XML attribute" % (len(dCluster), self.sXmlAttribute))
        
        if True:
            # for RV grognon... :)
            # done if bGraph... self.addEdgeToDOM(Y_proba)
            self.addClusterToDom(dCluster)
            
        return
    
    # --------------------------------------------------------------------
    def form_cluster(self, Y, fThres):
        import sys
        recursion_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(2*recursion_limit)
        try:
            def DFS(i):
                visited[i] = 1
                visited_index.append(i)
                for j in range(nb_node):
                    if visited[j] != 1 and ani[i][j]==1:
                        visited_index.append(j)
                        DFS(j)
                return visited_index
    
            dCluster = dict()
            count = 0
            
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
                    dfs_i = DFS(i) 
                    dCluster[count] = list(set(dfs_i))
                    count += 1
        finally:  
            sys.setrecursionlimit(recursion_limit)
        return dCluster

    def addEdgeToDOM(self, Y_proba):
        """
        To display the graph conveniently we add new Edge elements
        
        # for y_p, x_u, in zip(lY_pred, [X]):
            # edges = x_u[1][:int(len(x_u[1])/2)]
            # for i, (p,ie)  in enumerate(zip(y_p, edges)):     
                # print(p,  g.lNode[ie[0]].text,g.lNode[ie[1]].text, g.lEdge[i])
        """
        Y = Y_proba.argmax(axis=1)
        
        ndPage = self.lNode[0].page.node    
        #w = ndPage.get("imageFilename")
        ndPage.append(etree.Comment("Edges labeled by the conjugate graph"))
        for i, edge in enumerate(self.lEdge):
            cls = Y[i]
            if True or cls > 0 :    #type(edge) in [HorizontalEdge, VerticalEdge]:
                A, B = edge.A ,edge.B   #shape.centroid, edge.B.shape.centroid
                ndEdge = PageXml.createPageXmlNode("Edge")
                ndEdge.set("label" , self.lEdgeLabel[cls])
                ndEdge.set("proba", "%.3f" % Y_proba[i, cls])
                ndEdge.set("src", edge.A.node.get("id"))
                ndEdge.set("tgt", edge.B.node.get("id"))
                ndEdge.set("type", edge.__class__.__name__)
                ndPage.append(ndEdge)
                PageXml.setPoints(ndEdge, [(A.x1, A.y1), (B.x1, B.y1)]) 
                           
        return         

    def addClusterToDom(self, dCluster, bMoveContent=False):
        """
        Add Cluster elements to the Page DOM node
        """
        pageNode = None
        for x, lnidx in dCluster.items():    
            #self.analysedCluster()                             
            if pageNode is None:
                pageNode= self.lNode[lnidx[0]].page.node    
                pageNode.append(etree.Comment("Clusters created by the conjugate graph"))

            # lp = [ShapeLoader.node_to_Polygon(self.lNode[_i].node) for _i in lnidx]
            # Make it robust to bad data...
            lp = []
            for _i in lnidx:
                try:
                    lp.append(ShapeLoader.node_to_Polygon(self.lNode[_i].node))
                except ValueError:
                    pass
            contour = cascaded_union([p if p.is_valid else p.convex_hull for p in lp ])     
            # print(contour.wkt)
            try:spoints = ' '.join("%s,%s"%(int(x[0]),int(x[1])) for x in contour.minimum_rotated_rectangle.exterior.coords)
            except:spoints = ' '.join("%s,%s"%(int(x[0]),int(x[1])) for x in contour.minimum_rotated_rectangle.coords)
            #print (spoints)
            ndCluster = PageXml.createPageXmlNode('Cluster')     
            # add the space separated list of node ids
            ndCluster.set("content", " ".join(self.lNode[_i].node.get("id") for _i in lnidx))   
            coords = PageXml.createPageXmlNode('Coords')        
            ndCluster.append(coords)
            coords.set('points',spoints)                     
            pageNode.append(ndCluster)   
                 
            if bMoveContent:
                # move the DOM node of the content to the cluster
                for _i in lnidx:                               
                    ndCluster.append(self.lNode[_i].node)
            
        return


    # --------------------------------------------------------------------
    @classmethod
    def form_cluster_like_animesh(cls, THRES, lY_pred_proba, lX):
        """
        lX1, lY1 = self.get_lX_lY(lGraph)
        lX, lY = self.convert_lX_lY_to_LineDual(lX1,lY1)        
        """         
        ldCluster = []
        def DFS(i):
            visited[i] = 1
            visited_index.append(i)
            for j in range(nb_node):
                if visited[j] != 1 and ani[i][j]==1:
                    visited_index.append(j)
                    DFS(j)
            return visited_index

        for X, Y in zip(lX, lY_pred_proba):
            dCluster = dict()
            count = 0
            NF, E, _EF = X
            nb_node = NF.shape[0]
            
            # create an adjacency matrix
            ani = np.zeros(shape=(nb_node, nb_node), dtype='int64')
            for i, (a,b) in enumerate(E):
                if Y[i] < THRES:
                    # connected!
                    ani[a,b] = 1
                    ani[b,a] = 1
            
            visited = np.zeros(nb_node, dtype='int64')
            for i in range(nb_node):
                visited_index = []

                if visited[i] == 0:
                    dfs_i = DFS(i) 
                    dCluster[count] = list(set(dfs_i))
                    count += 1
            ldCluster.append(dCluster)
            
        return ldCluster
         

#     @classmethod
#     def form_cluster_animesh(cls,THRES,lY_pred_proba,lX):
#         """
#            lX1, lY1 = self.get_lX_lY(lGraph)
#         lX, lY = self.convert_lX_lY_to_LineDual(lX1,lY1)        
#         """         
#         out_cluster = []
#         # out_clutser_true = []
# 
#         def create_classedge_matrix(dict_rel, page_resol):
#             ret = np.zeros(shape=(len(page_resol), len(page_resol)), dtype='int64')
#             for node_id in dict_rel:
#                 for i in dict_rel[node_id]:
#                     ret[node_id, i] = 1
#             return ret
# 
#         for y_p, x_u, in zip(lY_pred_proba, lX):
#                     
#             dict_rel = {}
#             dict_p_x1x2={}            
#             page_resol = {}
#             edges = x_u[1][:int(len(x_u[1])/2)]
# 
#             #assert(len(y_t) == len(edges))
#             for p,ie  in zip(y_p, edges):
#               try:dict_p_x1x2[ie[0]][ie[1]]=p            
#               except KeyError:
#                   dict_p_x1x2[ie[0]]={}
#                   dict_p_x1x2[ie[0]][ie[1]]=p              
#               try:dict_p_x1x2[ie[1]][ie[0]]=p            
#               except KeyError:
#                   dict_p_x1x2[ie[1]]={}
#                   dict_p_x1x2[ie[1]][ie[0]]=p                      
#               try:   
#                 # dict_p_x1x2[ie[0]][ie[1]]=p
#   
#                 if p[0] > THRES:
#                     if ie[0] in dict_rel:
#                         dict_rel[ie[0]].append(ie[1]) 
#                     else:
#                         dict_rel[ie[0]] = [ie[1]]
#                     if ie[1] in dict_rel:
#                         dict_rel[ie[1]].append(ie[0])
#                     else:
#                         dict_rel[ie[1]] = [ie[0]]
#               except:
#                # needed for lY  = GT               
#                if p  < THRES:
#                 if ie[0] in dict_rel:
#                     dict_rel[ie[0]].append(ie[1])
#                 else:
#                     dict_rel[ie[0]] = [ie[1]]
#                 if ie[1] in dict_rel:
#                     dict_rel[ie[1]].append(ie[0])
#                 else:
#                     dict_rel[ie[1]] = [ie[0]]
# 
# 
#             # for x in dict_p_x1x2: 
#                 # for x2 in dict_p_x1x2[x]:
#                     # print (x,x2,dict_p_x1x2[x][x2])    
#             #print (dict_p_x1x2)                     
#             page_resol = {i:i for i in range(len(x_u[0]))}
#             ani = create_classedge_matrix(dict_rel, page_resol)      
# 
#             done_f = {} 
#             count  = 0
# 
#             visited = np.zeros(len(page_resol),dtype='int64')
#             for i in range(len(page_resol)):
#                 visited_index = []
#                 def DFS(i):
#                     if visited[i] == 1:
#                         return
#                     visited[i]=1
#                     visited_index.append(i)
#                     for j in range(len(page_resol)):
#                         if visited[j]!=1 and ani[i][j]==1:
#                             visited_index.append(j)
#                             DFS(j)
#                     return visited_index
# 
#                 dfs_i = DFS(i) 
# 
#                 if dfs_i is not None:
#                     done_f[count] = list(set(dfs_i))
#                     #print ( done_f[count])                       
#                     count += 1
#                     # for ii,ni in enumerate(dfs_i):
#                         # for nij in dfs_i[ii+1:]:
#                             # #print (ni,nij)
#                             # try:
#                               # print(ni,nij,dict_p_x1x2[ni][nij])                         
#                             # except:pass                              
#                     #sss                                            
#                         # print (                     
#             # keys_for_print = {}
#             # for k1, v1 in sorted((value, key) for (key, value) in page_resol.items()):
#                 # if k1 not in keys_for_print:
#                     # keys_for_print[k1] = []
#                 # keys_for_print[k1].append(v1)
#         
#             #print (done_f, '\n', keys_for_print)
#             #print (done_f)       
#             out_cluster.append(done_f)
#             # out_clutser_true.append(keys_for_print)
# 
#         return (out_cluster)    
        
def test_form_cluster_like_animesh():
    NF = np.zeros((9,3))
    lE_conn =    [  [0,1]
                  , [4,3]
                  , [4,2]
                  , [6,7]
                  , [2,3]
                  , [0,5]]
    #lE_conn.extend( [None]*len(lE_conn) )
    lE_disc = [ [0,2], [2,1], [2,7], [3,6], [6,8], [5,3] ]
    # lE_disc.extend( [None]*len(lE_disc) )
    lE, Y = [], []
    N = 6
    assert len(lE_conn) == N
    assert len(lE_disc) == N
    for i in range(N):
        lE.append(lE_conn[i])
        lE.append(lE_disc[i])
        Y.extend([0,1])
    E = np.array(lE)
    Y = np.array(Y)
    lX = [ [NF, E, None] ]
    lY = [Y]
#     ret = GraphBinaryConjugateModel.form_cluster_animesh(0.5, [Y], [ [NF, E, None] ])   
#     print(ret)
    ret = GraphBinaryConjugateSegmenter.form_cluster_like_animesh(0.5, [Y], [ [NF, E, None] ])   
    print(ret)
    assert ret == [{0: [0, 1, 5], 1: [2, 3, 4], 2: [6, 7], 3: [8]}]

            
if __name__ == "__main__":
    test_form_cluster_like_animesh()
        

