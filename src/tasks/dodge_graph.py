from crf.Graph_DSXml import Graph_DSXml
from crf.NodeType_DSXml   import NodeType_DS


DU_GRAPH = Graph_DSXml
nt = NodeType_DS("Ddg"                   #some short prefix because labels below are prefixed with it
                      , ['title', 'pnum']   #EXACTLY as in GT data!!!!
                      , []      #no ignored label/ One of those above or nothing, otherwise Exception!!
                      , True    #no label means OTHER
                      )
nt.setXpathExpr( ".//BLOCK"        #how to find the nodes
               )
DU_GRAPH.addNodeType(nt)

"""
The constraints must be a list of tuples like ( <operator>, <NodeType>, <states>, <negated> )
where:
- operator is one of 'XOR' 'XOROUT' 'ATMOSTONE' 'OR' 'OROUT' 'ANDOUT' 'IMPLY'
- states is a list of unary state names, 1 per involved unary. If the states are all the same, you can pass it directly as a single string.
- negated is a list of boolean indicated if the unary must be negated. Again, if all values are the same, pass a single boolean value instead of a list
"""
if False:
    DU_GRAPH.setPageConstraint( [    ('ATMOSTONE', nt, 'pnum' , False)    #0 or 1 catch_word per page
                                   , ('ATMOSTONE', nt, 'title'    , False)    #0 or 1 heading pare page
                                 ] )

# ===============================================================================================================
