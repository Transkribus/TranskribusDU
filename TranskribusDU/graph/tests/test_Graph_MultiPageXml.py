# -*- coding: utf-8 -*-


'''
Some ad-hoc test - not very easy to reproduce, I'm afraid

Created on 1 Dec 2016

@author: meunier
'''




import os

from lxml import etree

import graph.Graph_MultiPageXml as Graph_MultiPageXml
from graph.NodeType_PageXml   import NodeType_PageXml

import graph.Edge as Edge

import graph.Graph


def test_RectangleFitting():
    graph.Graph.Graph.resetNodeTypes()


    nt = NodeType_PageXml("TR"                   #some short prefix because labels below are prefixed with it
                              , ['catch-word', 'header', 'heading', 'marginalia', 'page-number']   #EXACTLY as in GT data!!!!
                              , []      #no ignored label/ One of those above or nothing, otherwise Exception!!
                              , True    #no label means OTHER
                              )
    nt.setXpathExpr( (".//pc:TextRegion"        #how to find the nodes
                      , "./pc:TextEquiv")       #how to get their text
                       )
    Graph_MultiPageXml.Graph_MultiPageXml.addNodeType(nt)
        
    print("- classes: ", Graph_MultiPageXml.Graph_MultiPageXml.getLabelNameList())

    obj = Graph_MultiPageXml.Graph_MultiPageXml()

    filename = os.path.join(os.path.dirname(__file__), "7749.mpxml")
    doc = etree.parse(filename)
    
    #load the block of each page, keeping the list of blocks of previous page
    lPrevPageNode = None

    for pnum, page, domNdPage in obj._iter_Page_DomNode(doc):
        #now that we have the page, let's create the node for each type!
        lPageNode = list()
        setPageNdDomId = set() #the set of DOM id
        # because the node types are supposed to have an empty intersection
                        
        lPageNode = [nd for nodeType in obj.getNodeTypeList() for nd in nodeType._iter_GraphNode(doc, domNdPage, page) ]

        obj.lNode.extend(lPageNode)
        
        lPageEdge = Edge.Edge.computeEdges(lPrevPageNode, lPageNode)
        
        obj.lEdge.extend(lPageEdge)
        print("\tPage %5d    %6d nodes    %7d edges" %(pnum, len(lPageNode), len(lPageEdge)))
        
        nc = 20
        fmt = "%%s p%%s %%%ss  --->  p%%s %%%ss"%(nc, nc)
        for edge in lPageEdge:
            sa, sb = edge.A.text, edge.B.text
            sa, sb = sa[:min(len(sa), nc)], sb[:min(len(sb), nc)]
            assert edge.A.node.get("id") != edge.B.node.get("id")
            print(fmt%(edge.__class__.__name__[0], edge.A.pnum, sa, edge.B.pnum, sb))
        lPrevPageNode = lPageNode
    
#     sOut = "TEST_getPageXmlBlock.mpxml"
#     
#     assert os.path.exists(sOut)
    
    
if __name__ == "__main__":
    test_RectangleFitting()
    
    