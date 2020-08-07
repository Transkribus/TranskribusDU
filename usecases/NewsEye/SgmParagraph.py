# -*- coding: utf-8 -*-

"""
    DU task for segmenting words into Menu-Items using the conjugate graph
    
    Copyright NAVER LABS Europe(C)  2019  Jean-Luc Meunier
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""
 
import sys, os
from graph.FeatureDefinition_PageXml_std_noText import FeatureDefinition_PageXml_StandardOnes_noText

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version
TranskribusDU_version

from common.trace import traceln

from graph.NodeType_PageXml     import defaultBBoxDeltaFun

from tasks.DU_Task_Factory                          import DU_Task_Factory
from graph.pkg_GraphBinaryConjugateSegmenter.MultiSinglePageXml  import MultiSinglePageXml as ConjugateSegmenterGraph_MultiSinglePageXml 
from graph.pkg_GraphBinaryConjugateSegmenter.MultiSinglePageXml_Separator \
    import MultiSinglePageXml_Separator \
    as ConjugateSegmenterGraph_MultiSinglePageXml_Separator

#from graph.pkg_ReifiedEdge.MultiSinglePageXml_Segmenter_Separator_DOM import Graph_MultiSinglePageXml_Segmenter_Separator_DOM


from graph.NodeType_PageXml                         import NodeType_PageXml_type
from graph.NodeType_jsonOCR                         import NodeType_jsonOCR
from graph.FeatureDefinition_Generic_noText         import FeatureDefinition_Generic_noText
from graph.FeatureDefinition_Generic                import FeatureDefinition_Generic
from tasks.DU_Task_Features                         import Features_June19_Full, Features_June19_Full_Separator


# ----------------------------------------------------------------------------

class My_ConjugateNodeType(NodeType_PageXml_type):
    """
    We need this to extract properly the label from the label attribute of the (parent) TableCell element.
    """
    def __init__(self, sNodeTypeName, lsLabel, lsIgnoredLabel=None, bOther=True, BBoxDeltaFun=defaultBBoxDeltaFun
                 , bPreserveWidth=False):
        super(My_ConjugateNodeType, self).__init__(sNodeTypeName, lsLabel, lsIgnoredLabel, bOther, BBoxDeltaFun
                                                   , bPreserveWidth=bPreserveWidth)

    def parseDocNodeLabel(self, graph_node, defaultCls=None):
        """
        Parse and set the graph node label and return its class index
        raise a ValueError if the label is missing while bOther was not True, or if the label is neither a valid one nor an ignored one
        """
        #sLabel = domnode.getparent().get(self.sLabelAttr)
        domnode = graph_node.node.getparent()
        sLabel = domnode.get(self.sLabelAttr)
        
        return sLabel if not sLabel is None else "__none__"

    def setDocNodeLabel(self, graph_node, sLabel):
        raise Exception("This should not occur in conjugate mode")    

    
def getConfiguredGraphClass(doer):
    """
    In this class method, we must return a configured graph class
    """
#     if options.bReified:
#         DU_GRAPH = Graph_MultiSinglePageXml_Segmenter_Separator_DOM
    if options.bSeparator:
        DU_GRAPH = ConjugateSegmenterGraph_MultiSinglePageXml_Separator
    else:
        DU_GRAPH = ConjugateSegmenterGraph_MultiSinglePageXml
    ntClass = My_ConjugateNodeType

    if options.bBB2:
        nt = ntClass("mi_clstr"                   #some short prefix because labels below are prefixed with it
                      , []                   # in conjugate, we accept all labels, andNone becomes "none"
                      , []
                      , False                # unused
                      , BBoxDeltaFun  = None
                      , bPreserveWidth=True
                  )    
    else:
        nt = ntClass("mi_clstr"                   #some short prefix because labels below are prefixed with it
                      , []                   # in conjugate, we accept all labels, andNone becomes "none"
                      , []
                      , False                # unused
                      , BBoxDeltaFun  =lambda v: max(v * 0.066, min(5, v/3))  #we reduce overlap in this way
                  )    
    nt.setLabelAttribute("id")
    
    ## HD added 23/01/2020: needed for output generation
    DU_GRAPH.clusterType='paragraph'
    nt.setXpathExpr((  ".//pc:TextLine"
                     , "./pc:TextEquiv")       #how to get their text
                     )
    DU_GRAPH.addNodeType(nt)
    
    return DU_GRAPH


if __name__ == "__main__":
    traceln("VERSION: %s" % DU_Task_Factory.getVersion())
    
    # standard command line options for CRF- ECN- GAT-based methods
    usage, parser = DU_Task_Factory.getStandardOptionsParser(sys.argv[0])
#     parser.add_option("--spm"       , dest='sSPModel'    , action="store", type="string"
#                       , help="Textual features are computed based on the given SentencePiece model. e.g. model/toto.model.")     
    parser.add_option("--unigram"       , dest='unigram'    , action="store"
                      ,type="int", default = 0
                      , help="Textual features as unigram: Max uni")     
    parser.add_option("--pxmlfeatures"       , dest='pxmlfeatures'    , action="store_true"
                      , default=False
                      , help="Use pageXml features (page h,w)")         
    parser.add_option("--separator"       , dest='bSeparator'    , action="store_true"
                      , default=False
                      , help="Use separators")   
    parser.add_option("--BB2", dest='bBB2'    , action="store_true"
                      , help="New style BB (same width as baseline, no resize)")     
    (options, args) = parser.parse_args()
    

    if options.bSeparator:
        cFeatureDefinition = Features_June19_Full_Separator
        dFeatureConfig = { }               
    elif options.pxmlfeatures:
        cFeatureDefinition = Features_June19_Full
        dFeatureConfig = { }               
    else:
        cFeatureDefinition = FeatureDefinition_Generic
        dFeatureConfig = { }               
        
    try:
        sModelDir, sModelName = args
    except Exception as e:
        traceln("Specify a model folder and a model name!")
        DU_Task_Factory.exit(usage, 1, e)
    
    doer = DU_Task_Factory.getDoer(sModelDir, sModelName
                                   , options                    = options
                                   , fun_getConfiguredGraphClass= getConfiguredGraphClass
                                   , cFeatureDefinition         = cFeatureDefinition
                                   , dFeatureConfig             = dFeatureConfig                                           
                                   )
    
    # setting the learner configuration, in a standard way 
    # (from command line options, or from a JSON configuration file)
    dLearnerConfig = doer.getStandardLearnerConfig(options)
    
    
#     # force a balanced weighting
#     print("Forcing balanced weights")
#     dLearnerConfig['balanced'] = True
    
    # of course, you can put yours here instead.
    doer.setLearnerConfiguration(dLearnerConfig)

    # act as per specified in the command line (--trn , --fold-run, ...)
    doer.standardDo(options)
    
    del doer

