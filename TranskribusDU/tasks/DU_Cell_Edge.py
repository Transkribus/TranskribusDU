# -*- coding: utf-8 -*-

"""
    DU task for segmenting text in cells using the conjugate graph after the SW
    re-engineering by JLM
    
    Copyright Xerox(C)  2019  Jean-Luc Meunier
    
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
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""
 
import sys, os

import numpy as np
import shapely.geometry as geom
from shapely.prepared import prep
from rtree import index

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from common.trace import traceln
from util.Shape import ShapeLoader

from tasks.DU_Task_Factory                          import DU_Task_Factory
from tasks.DU_Row_Edge                              import ConjugateSegmenterGraph_MultiSinglePageXml
from graph.Transformer                              import Transformer, FeatureUnion
from graph.NodeType_PageXml                         import defaultBBoxDeltaFun
from graph.NodeType_PageXml                         import NodeType_PageXml_type_woText


from graph.GraphBinaryConjugateSegmenter    import GraphBinaryConjugateSegmenter
from graph.Graph_Multi_SinglePageXml        import Graph_MultiSinglePageXml
from tasks.DU_Task_Features                 import Features_June19_Simple


class ConjugateSegmenterGraph_MultiSinglePageXml(GraphBinaryConjugateSegmenter, Graph_MultiSinglePageXml):
    """
    standard graph but in conjugate mode
    """
    def __init__(self):
        super(ConjugateSegmenterGraph_MultiSinglePageXml, self).__init__()
    

# ---------------------   EXPLOITING GRAPHICAL SEPARATORS   ----------------------------    
class ConjugateSegmenterGraph_MultiSinglePageXml_Separator(ConjugateSegmenterGraph_MultiSinglePageXml):
    bVerbose = True
    
    """
    near standard graph, used in conjugate mode, an aware of SeparatorRegion
    """
    def __init__(self):
        super(ConjugateSegmenterGraph_MultiSinglePageXml, self).__init__()
    
    
    def _index(self):
        """
        This method is called before computin the Xs
        We call it and right after, we compute the intersection of edge with SeparatorRegions
        Then, feature extraction can reflect the crossing of edges and separators 
        """
        bFirstCall = super(ConjugateSegmenterGraph_MultiSinglePageXml_Separator, self)._index()
        
        if bFirstCall:
            # indexing was required
            # , so first call
            # , so we need to make the computation of edges crossing separators!
            self.addSeparatorFeature()

    def addSeparatorFeature(self):
        """
        We load the graphical separators
        COmpute a set of shapely object
        In turn, for each edge, we compute the intersection
        """
        
        # graphical separators
        from xml_formats.PageXml import PageXml
        dNS = {"pc":PageXml.NS_PAGE_XML}
        ndRoot = self.lNode[0].node.getroottree()
        lNdSep = ndRoot.xpath(".//pc:SeparatorRegion", namespaces=dNS)
        loSep = [ ShapeLoader.node_to_LineString(_nd) for _nd in lNdSep]
        
        if self.bVerbose: traceln(" %d graphical separators"%len(loSep))
        
        # make an indexed rtree
        idx = index.Index()
        for i, oSep in enumerate(loSep):
            idx.insert(i, oSep.bounds)
            
        # take each edge in turn and list the separators it crosses
        nCrossing = 0
        for edge in self.lEdge:
            # bottom-left corner to bottom-left corner
            oEdge = geom.LineString([(edge.A.x1, edge.A.y1), (edge.B.x1, edge.B.y1)])
            prepO = prep(oEdge)
            bCrossing = False
            for i in idx.intersection(oEdge.bounds):
                # check each candidate in turn
                if prepO.intersects(loSep[i]):
                    bCrossing = True
                    nCrossing += 1
                    break
            edge.bCrossingSep = bCrossing
        
        if self.bVerbose: 
            traceln(" %d (/ %d) edges crossing at least one graphical separator"%(nCrossing, len(self.lEdge)))
        
        
class Separator_boolean(Transformer):
    """
    a boolean encoding indicating if the edge crosses a separator
    """
    def transform(self, lO):
        nb = len(lO)
        a = np.zeros((nb, 1), dtype=np.float64)
        for i, o in enumerate(lO):
            if o.bCrossingSep: a[i,0] = 1
        return a

    def __str__(self):
        return "- Separator_boolean %s (#1)" % (self.__class__)

        
class Features_June19_Simple_Separator(Features_June19_Simple):
    """
    Simple features:
        NODE: geometry
        EDGE: type and geometry  + boolean for separator crossing
    """

    def __init__(self): 
        Features_June19_Simple.__init__(self)
        
        node_transformer, edge_transformer = self.getTransformers()
        edge_transformer =  FeatureUnion([                                            \
                  ('Features_June19_Simple'   , edge_transformer) # Edge class 1 hot encoded (PUT IT FIRST)
                , ('sprtr_bool', Separator_boolean())
                            ])
        self.setTransformers(node_transformer, edge_transformer)
        
        
# -----------------------------------------------------------------------------

        
class My_ConjugateNodeType(NodeType_PageXml_type_woText):
    """
    We need this to extract properly the label from the @cell attribute of the TableCell element.
    """
    def __init__(self, sNodeTypeName, lsLabel, lsIgnoredLabel=None, bOther=True, BBoxDeltaFun=defaultBBoxDeltaFun):
        super(My_ConjugateNodeType, self).__init__(sNodeTypeName, lsLabel, lsIgnoredLabel, bOther, BBoxDeltaFun)

    def parseDomNodeLabel(self, domnode, defaultCls=None):
        """
        Parse and set the graph node label and return its class index
        raise a ValueError if the label is missing while bOther was not True, or if the label is neither a valid one nor an ignored one
        """
        ndParent = domnode.getparent()
        sLabel = str(ndParent.get("row")) + "_" + str(ndParent.get("col"))

        return sLabel
        
    def setDomNodeLabel(self, domnode, sLabel):
        raise Exception("This should not occur in conjugate mode")    
    
    
def getConfiguredGraphClass(doer):
    """
    In this class method, we must return a configured graph class
    """
    global options

    # each graph reflects 1 page
    if options.bSeparator:
        DU_GRAPH = ConjugateSegmenterGraph_MultiSinglePageXml_Separator
    else:
        DU_GRAPH = ConjugateSegmenterGraph_MultiSinglePageXml

    # ntClass = NodeType_PageXml_type
    ntClass = My_ConjugateNodeType

    nt = ntClass("cell"                   #some short prefix because labels below are prefixed with it
                  , []                   # in conjugate, we accept all labels, andNone becomes "none"
                  , []
                  , False                # unused
                  , BBoxDeltaFun=lambda v: max(v * 0.066, min(5, v/3))  #we reduce overlap in this way
                  )    
    nt.setLabelAttribute("row")
    nt.setXpathExpr( (".//pc:TextLine"        #how to find the nodes            
                      #, "./pc:TextEquiv")       #how to get their text
                      , ".//pc:Unicode")       #how to get their text
                   )
    DU_GRAPH.addNodeType(nt)
    
    return DU_GRAPH


    
if __name__ == "__main__":
    #     import better_exceptions
    #     better_exceptions.MAX_LENGTH = None
    global options
    
    # standard command line options for CRF- ECN- GAT-based methods
    usage, parser = DU_Task_Factory.getStandardOptionsParser(sys.argv[0])

    parser.add_option("--separator"     , dest='bSeparator'   , action="store_true"
                      , default=False,help="Use the graphical spearators, if any, as edge features.") 
    traceln("VERSION: %s" % DU_Task_Factory.getVersion())

    # --- 
    #parse the command line
    (options, args) = parser.parse_args()

    try:
        sModelDir, sModelName = args
    except Exception as e:
        traceln("Specify a model folder and a model name!")
        DU_Task_Factory.exit(usage, 1, e)

    # === SETTING the graph type (and its node type) a,d the feature extraction pipe
    doer = DU_Task_Factory.getDoer(sModelDir, sModelName
                                   , options                    = options
                                   , fun_getConfiguredGraphClass= getConfiguredGraphClass
                                   , cFeatureDefinition         = Features_June19_Simple_Separator if options.bSeparator else Features_June19_Simple
                                   )
    
    # == LEARNER CONFIGURATION ===
    # setting the learner configuration, in a standard way 
    # (from command line options, or from a JSON configuration file)
    dLearnerConfig = doer.getStandardLearnerConfig(options)
    
    
#     # force a balanced weighting
#     print("Forcing balanced weights")
#     dLearnerConfig['balanced'] = True
    
    # of course, you can put yours here instead.
    doer.setLearnerConfiguration(dLearnerConfig)

    # === CONJUGATE MODE ===
    doer.setConjugateMode()
    
    # === GO!! ===
    # act as per specified in the command line (--trn , --fold-run, ...)
    doer.standardDo(options)
    
    del doer

