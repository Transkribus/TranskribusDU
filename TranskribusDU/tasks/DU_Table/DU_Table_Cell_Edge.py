# -*- coding: utf-8 -*-

"""
    DU task for segmenting text in cell, or col, or row using the conjugate 
    graph after the SW re-engineering by JLM during the 2019 summer.
    
    As of June 5th, 2015, this is the exemplary code
    
    Copyright NAVER(C)  2019  Jean-Luc Meunier
    

    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""
 
import sys, os

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) ))) )
    import TranskribusDU_version
TranskribusDU_version

from common.trace import traceln

from graph.NodeType_PageXml     import defaultBBoxDeltaFun
from graph.NodeType_PageXml     import NodeType_PageXml_type
from tasks.DU_Task_Factory      import DU_Task_Factory
from tasks.DU_Task_Features     import Features_June19_Simple
from tasks.DU_Task_Features     import Features_June19_Simple_Separator
from tasks.DU_Task_Features     import Features_June19_Simple_Shift
from tasks.DU_Task_Features     import Features_June19_Simple_Separator_Shift
from tasks.DU_Task_Features     import Features_June19_Full
from tasks.DU_Task_Features     import Features_June19_Full_Separator
from tasks.DU_Task_Features     import Features_June19_Full_Shift
from tasks.DU_Task_Features     import Features_June19_Full_Separator_Shift

from graph.pkg_GraphBinaryConjugateSegmenter.MultiSinglePageXml \
    import MultiSinglePageXml \
    as ConjugateSegmenterGraph_MultiSinglePageXml

from graph.pkg_GraphBinaryConjugateSegmenter.MultiSinglePageXml_Separator \
    import MultiSinglePageXml_Separator \
    as ConjugateSegmenterGraph_MultiSinglePageXml_Separator


# ----------------------------------------------------------------------------
# class My_ConjugateNodeType(NodeType_PageXml_type_woText):
class My_ConjugateNodeType(NodeType_PageXml_type):
    """
    We need this to extract properly the label from the label attribute of the (parent) TableCell element.
    """
    def __init__(self, sNodeTypeName, lsLabel, lsIgnoredLabel=None, bOther=True, BBoxDeltaFun=defaultBBoxDeltaFun):
        super(My_ConjugateNodeType, self).__init__(sNodeTypeName, lsLabel, lsIgnoredLabel, bOther, BBoxDeltaFun)

    def parseDocNodeLabel(self, graph_node, defaultCls=None):
        """
        Parse and set the graph node label and return its class index
        We rely on the standard self.sLabelAttr
        raise a ValueError if the label is missing while bOther was not True
         , or if the label is neither a valid one nor an ignored one
        """
        domnode = graph_node.node
        ndParent = domnode.getparent()
        sLabel = "%s__%s" % ( ndParent.getparent().get("id")  # TABLE ID !
                            , ndParent.get(self.sLabelAttr)   # e.g. "row" or "col"
                            )
        
        return sLabel

    def setDocNodeLabel(self, graph_node, sLabel):
        raise Exception("This should not occur in conjugate mode")    
    

class My_ConjugateNodeType_Cell(My_ConjugateNodeType):
    """
    For cells, the label is formed by the row  __and__  col numberss
    """
    def __init__(self, sNodeTypeName, lsLabel, lsIgnoredLabel=None, bOther=True, BBoxDeltaFun=defaultBBoxDeltaFun):
        super(My_ConjugateNodeType_Cell, self).__init__(sNodeTypeName, lsLabel, lsIgnoredLabel, bOther, BBoxDeltaFun)

    def parseDocNodeLabel(self, graph_node, defaultCls=None):
        """
        Parse and set the graph node label and return its class index
        raise a ValueError if the label is missing while bOther was not True, or if the label is neither a valid one nor an ignored one
        """
        domnode = graph_node.node
        ndParent = domnode.getparent()
        sLabel = "%s__%s__%s" % (  ndParent.getparent().get("id")  # TABLE ID !
                               , ndParent.get("row")
                               , ndParent.get("col")
                               )
        return sLabel


# ----------------------------------------------------------------------------
def main(sys_argv_0, sLabelAttribute, cNodeType=My_ConjugateNodeType):

    
    def getConfiguredGraphClass(_doer):
        """
        In this class method, we must return a configured graph class
        """
        # each graph reflects 1 page
        if options.bSeparator:
            DU_GRAPH = ConjugateSegmenterGraph_MultiSinglePageXml_Separator
        else:
            DU_GRAPH = ConjugateSegmenterGraph_MultiSinglePageXml
    
        ntClass = cNodeType
    
        nt = ntClass(sLabelAttribute         #some short prefix because labels below are prefixed with it
                      , []                   # in conjugate, we accept all labels, andNone becomes "none"
                      , []
                      , False                # unused
                      , BBoxDeltaFun=lambda v: max(v * 0.066, min(5, v/3))  #we reduce overlap in this way
                      )    
        nt.setLabelAttribute(sLabelAttribute)
        nt.setXpathExpr( (".//pc:TextLine"        #how to find the nodes            
                          #, "./pc:TextEquiv")       #how to get their text
                          , ".//pc:Unicode")       #how to get their text
                       )
        DU_GRAPH.addNodeType(nt)
        
        return DU_GRAPH

    # standard command line options for CRF- ECN- GAT-based methods
    usage, parser = DU_Task_Factory.getStandardOptionsParser(sys_argv_0)
    parser.add_option("--separator", dest='bSeparator', action="store_true"
                      , default=False, help="Use the graphical spearators, if any, as edge features.") 
    parser.add_option("--text"       , dest='bText'     , action="store_true"
                      , default=False, help="Use textual information if any, as node and edge features.") 
    parser.add_option("--edge_vh", "--edge_hv"    , dest='bShift'    , action="store_true"
                      , default=False, help="Shift edge feature by range depending on edge type.") 
    parser.add_option("--jsonocr", dest='bJsonOcr',  action="store_true"
                          , help="I/O is in json")   
    traceln("VERSION: %s" % DU_Task_Factory.getVersion())

    # --- 
    #parse the command line
    (options, args) = parser.parse_args()

    try:
        sModelDir, sModelName = args
    except Exception as e:
        traceln("Specify a model folder and a model name!")
        DU_Task_Factory.exit(usage, 1, e)
    if options.bText     : traceln(" - using textual data, if any")
    if options.bSeparator: traceln(" - using graphical separators, if any")
    if options.bShift    : traceln(" - shift edge features by edge type")
    
    if options.bText:
        if options.bSeparator:
            if options.bShift:
                cFeatureDefinition = Features_June19_Full_Separator_Shift
            else:
                cFeatureDefinition = Features_June19_Full_Separator
        else: 
            if options.bShift:
                cFeatureDefinition = Features_June19_Full_Shift
            else:  
                cFeatureDefinition = Features_June19_Full 
    else:
        if options.bSeparator:
            if options.bShift:
                cFeatureDefinition = Features_June19_Simple_Separator_Shift
            else:  
                cFeatureDefinition = Features_June19_Simple_Separator
        else: 
            if options.bShift:
                cFeatureDefinition = Features_June19_Simple_Shift 
            else:  
                cFeatureDefinition = Features_June19_Simple 

    # === SETTING the graph type (and its node type) a,d the feature extraction pipe
    doer = DU_Task_Factory.getDoer(sModelDir, sModelName
                                   , options                    = options
                                   , fun_getConfiguredGraphClass= getConfiguredGraphClass
                                   , cFeatureDefinition         = cFeatureDefinition
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

    # === GO!! ===
    # act as per specified in the command line (--trn , --fold-run, ...)
    doer.standardDo(options)
    
    del doer

    
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    #     import better_exceptions
    #     better_exceptions.MAX_LENGTH = None
    
    main(sys.argv[0], "cell", My_ConjugateNodeType_Cell)
