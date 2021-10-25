# -*- coding: utf-8 -*-

"""
    DU task for the STAB dataset: TextRegion segmentation task
    
    Copyright NAVER LABS Europe(C)  2021 Hervé Déjean, JL Meunier
    
    
"""
 
import sys, os

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version
TranskribusDU_version

from common.trace import traceln

from tasks.DU_Task_Factory                          import DU_Task_Factory
from graph.NodeType_PageXml                         import NodeType_PageXml_type
from graph.NodeType_PageXml                         import defaultBBoxDeltaFun
from graph.pkg_GraphBinaryConjugateSegmenter.MultiSinglePageXml  \
            import MultiSinglePageXml as ConjugateSegmenterGraph_MultiSinglePageXml 

from graph.FeatureDefinition_PageXml_std_noText  import FeatureDefinition_PageXml_StandardOnes_noText
from graph.FeatureDefinition_PageXml_std            import FeatureDefinition_PageXml_StandardOnes


#from funsd import parserAddSharedOptions, getDataToPickle, selectFeatureStuff


# ----------------------------------------------------------------------------
class My_ConjugateSegmenterGraph_MultiSinglePageXml(ConjugateSegmenterGraph_MultiSinglePageXml):
    """
    HACK to add a @line_id attribute to words, since we did not add it when creating the XML
    """
    def __init__(self):
        super(My_ConjugateSegmenterGraph_MultiSinglePageXml, self).__init__()
    
    def isEmpty(self):
        traceln("\tHACK to create a @line_id on nodes")
        for nd in self.lNode:
            nd.node.set("line_id", nd.node.getparent().get("id"))
        return super(My_ConjugateSegmenterGraph_MultiSinglePageXml, self).isEmpty()


# ----------------------------------------------------------------------------
class My_ConjugateNodeType(NodeType_PageXml_type):
    """
    We need this to extract properly the label from the label attribute of the (parent) TableCell element.
    """
    def __init__(self, sNodeTypeName, lsLabel, lsIgnoredLabel=None, bOther=True, BBoxDeltaFun=defaultBBoxDeltaFun, bPreserveWidth=False):
        super(My_ConjugateNodeType, self).__init__(sNodeTypeName, lsLabel, lsIgnoredLabel, bOther, BBoxDeltaFun=BBoxDeltaFun, bPreserveWidth=bPreserveWidth)

    def parseDocNodeLabel(self, graph_node, defaultCls=None):
        """
        Parse and set the graph node label and return its class index
        raise a ValueError if the label is missing while bOther was not True, or if the label is neither a valid one nor an ignored one
        """
        domnode = graph_node.node
        sLabel = domnode.getparent().get(self.sLabelAttr)
        return str(sLabel)

    def setDocNodeLabel(self, graph_node, sLabel):
        raise Exception("This should not occur in conjugate mode")    

    
def getConfiguredGraphClass(doer):
    """
    In this class method, we must return a configured graph class
    """
    DU_GRAPH = ConjugateSegmenterGraph_MultiSinglePageXml #My_ConjugateSegmenterGraph_MultiSinglePageXml
    ntClass = My_ConjugateNodeType

    nt = ntClass("region"            #some short prefix because labels below are prefixed with it
                  , []                   # in conjugate, we accept all labels, andNone becomes "none"
                  , []
                  , False                # unused
                  #, BBoxDeltaFun=lambda v: max(v * 0.066, min(5, v/3))  #we reduce overlap in this way
                  , BBoxDeltaFun=None if options.bBB2 else lambda v: max(v * 0.066, min(5, v/3))  #we reduce overlap in this way
                  , bPreserveWidth=True if options.bBB2 else False
                  )    
    nt.setLabelAttribute("id")
    
    ## HD added 23/01/2020: needed for output generation
    DU_GRAPH.clusterType='region'
    nt.setXpathExpr((  ".//pc:TextLine"
                     , ".//pc:TextEquiv")       #how to get their text
                     )
    DU_GRAPH.addNodeType(nt)
    
    return DU_GRAPH


if __name__ == "__main__":
    traceln("VERSION: %s" % DU_Task_Factory.getVersion())
    
    # standard command line options for CRF- ECN- GAT-based methods
    usage, parser = DU_Task_Factory.getStandardOptionsParser(sys.argv[0])
    
    #parserAddSharedOptions(parser)
    
    (options, args) = parser.parse_args()
    options.bBB2=True 
    #cFeatureDefinition, dFeatureConfig = selectFeatureStuff(options)

    bText=True
    if not bText:
        cFeatureDefinition = FeatureDefinition_PageXml_StandardOnes_noText
        dFeatureConfig = {}
    else:
        cFeatureDefinition = FeatureDefinition_PageXml_StandardOnes
        if False:
            # this is the best configuration when using the text
            dFeatureConfig = {  'n_tfidf_node':400, 't_ngrams_node':(1,1), 'b_tfidf_node_lc':False
                            , 'n_tfidf_edge':400, 't_ngrams_edge':(1,1), 'b_tfidf_edge_lc':False
            }
        elif True:
            dFeatureConfig = {  'n_tfidf_node':512, 't_ngrams_node':(1,2), 'b_tfidf_node_lc':False
                            , 'n_tfidf_edge':512, 't_ngrams_edge':(1,2), 'b_tfidf_edge_lc':False
            }
        else:
            dFeatureConfig = {  'n_tfidf_node':1024, 't_ngrams_node':(1,3), 'b_tfidf_node_lc':False
                            , 'n_tfidf_edge':1024, 't_ngrams_edge':(1,3), 'b_tfidf_edge_lc':False
            }


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
    #doer.setAdditionalDataProvider(getDataToPickle)


    # act as per specified in the command line (--trn , --fold-run, ...)
    doer.standardDo(options)
    
    del doer

