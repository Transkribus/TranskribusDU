# -*- coding: utf-8 -*-

"""
how to pickle the node true characteristics for any table

JL Meunier 
15/1/2020
Copyright Naver Labs Europe
"""

def getDataToPickle_for_table(doer, mdl, lGraph):
    """
    data that is specific to this task, which we want to pickle when --pkl is used
    for each node of each graph, we want to store the node text + row and column numbers + rowSpan and colSpan
    
    ( (text, (x1, y1, x2, y2), (row, col, rowSpan, colSpan) )
    ...
    """
    def attr_to_int(domnode, sAttr, default_value=None):
        s = domnode.get(sAttr)
        try:
            return int(s)
        except (ValueError, TypeError):
            return default_value
    
    lDataByGraph = []
    for g in lGraph:
        lNodeData = []
        for nd in g.lNode:
            ndCell = nd.node.getparent()
            data = (nd.text
                      , (nd.x1, nd.y1, nd.x2, nd.y2)
                      , (attr_to_int(ndCell, "row")
                         , attr_to_int(ndCell, "col")
                         , attr_to_int(ndCell, "rowSpan")
                         , attr_to_int(ndCell, "colSpan"))
                      )
            lNodeData.append(data)
        lDataByGraph.append(lNodeData)
    return lDataByGraph
    
def getSortedDataToPickle_for_table(doer, mdl, lGraph,key="row"):
    """
    similar to getDataToPickle_for_table but  sorted by "key" (row or col) 
    
    """
    if key == "row": key2 = "col" 
    else: key2 = "row"
    
    def attr_to_int(domnode, sAttr, default_value=None):
        s = domnode.get(sAttr)
        try:
            return int(s)
        except (ValueError, TypeError):
            return default_value
    
    lDataByGraph = []
    for g in lGraph:
        dKey = {}
        lNodeData = []
        for i,nd in enumerate(g.lNode):
            ndCell = nd.node.getparent()
            try:dKey[attr_to_int(ndCell,key,-1)].append((i,nd))
            except KeyError: dKey[attr_to_int(ndCell,key,-1)]=[(i,nd)]
        
        gt = 0
        for ikey in sorted(dKey):
            #print (ikey,[x[0] for x in dKey[ikey]])
            dKey[ikey].sort(key=lambda x:attr_to_int(x[1].node.getparent(),key,-1))
            for i,nd in dKey[ikey]:
                ndCell = nd.node.getparent()
                data = (i,nd.text
                          , (nd.x1, nd.y1, nd.x2, nd.y2)
                          , (attr_to_int(ndCell, "row")
                             , attr_to_int(ndCell, "col")
                             , attr_to_int(ndCell, "rowSpan")
                             , attr_to_int(ndCell, "colSpan"))
                          )
                gt+= 1
                lNodeData.append(data)
        print (len(g.lNode),len(lNodeData))
        assert len(lNodeData) == len(g.lNode)
        lDataByGraph.append(lNodeData)
    assert len(lGraph) == len(lDataByGraph)
    return lDataByGraph



def getSortedDataToPickle(doer, mdl, lGraph,key="row"):
    """
    similar to getDataToPickle_for_table but  NOT FOR TABLE and sorted by "key" (row or col) 
    
    keys: TextRegion number 
    Assume order in RO in a TextRegion
    """
    key = "number"
    
    def attr_to_int(domnode, sAttr, default_value=None):
        s = domnode.get(sAttr)
        try:
            return int(s)
        except (ValueError, TypeError):
            return default_value
    
    lDataByGraph = []
    for g in lGraph:
        dKey = {}
        lNodeData = []
        for i,nd in enumerate(g.lNode):
            ndRegion = nd.node.getparent()
#             print (i,nd.node.get(key),ndRegion.get(key))
            try:dKey[attr_to_int(ndRegion,key,-1)].append((i,nd))
            except KeyError: dKey[attr_to_int(ndRegion,key,-1)]=[(i,nd)]
#         print ('__')
        gt = 0
        for ikey in sorted(dKey):
#             print (ikey,[x[0] for x in dKey[ikey]])
            dKey[ikey].sort(key=lambda x:attr_to_int(x[1].node.getparent(),key,-1))
            for i,nd in dKey[ikey]:
                ndRegion = nd.node.getparent()
#                 print (i,nd.node.get('number'),ndRegion.get('number'))
                data = (i,nd.text
                          , (nd.x1, nd.y1, nd.x2, nd.y2)
                          )
                gt+= 1
                lNodeData.append(data)
        print (len(g.lNode),len(lNodeData))
        assert len(lNodeData) == len(g.lNode)
        lDataByGraph.append(lNodeData)
    assert len(lGraph) == len(lDataByGraph)
    return lDataByGraph
