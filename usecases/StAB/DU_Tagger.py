# -*- coding: utf-8 -*-

"""
    Tagging TextLine for Annemieke's StABccollection
    
    Copyright Xerox(C) 2016 JL. Meunier
    Copyright Naver(C) 2019 H. Déjean
"""
import sys, os

import TranskribusDU_version  # if import error, updade the PYTHONPATH environment variable

from common.trace import traceln
from tasks.DU_Task_Factory                          import DU_Task_Factory
from graph.Graph_Multi_SinglePageXml                import Graph_MultiSinglePageXml
from graph.NodeType_PageXml                         import NodeType_PageXml_type
from graph.NodeType_PageXml                         import NodeType_PageXml_type_woText
from graph.FeatureDefinition_PageXml_std            import FeatureDefinition_PageXml_StandardOnes
from graph.FeatureDefinition_PageXml_std_noText     import FeatureDefinition_PageXml_StandardOnes_noText
from graph.FeatureDefinition_Generic                import FeatureDefinition_Generic
from graph.FeatureDefinition_Generic_noText         import FeatureDefinition_Generic_noText

# =============================================================================
bTextRegion = False         # do we tag TextRegion? Otherwise we will tag TextLine
bText       = False         # do we use text?  (set to False for TextRegion)
bGenericFeature = False      # do not use page width and height
# =============================================================================


class NodeType_for_TextLine(NodeType_PageXml_type):
    """
    We specialize it because we get the label (of TextLine nodes) from the parent node (TextRegion)
    """
    def getLabel(self, domnode):
        return domnode.getparent().get(self.sLabelAttr)


def getConfiguredGraphClass_for_TextLine(doer):
    """
    we return a configured graph class for tagging TextLine
    """
    #DU_GRAPH = ConjugateSegmenterGraph_MultiSinglePageXml  # consider each age as if indep from each other
    DU_GRAPH = Graph_MultiSinglePageXml
    
    # ntClass = NodeType_PageXml_type_woText #NodeType_PageXml_type
    ntClass = NodeType_for_TextLine

    lLabels        = ['heading', 'paragraph', 'page-number']
    lIgnoredLabels = ['caption', 'marginalia']

    nt = ntClass("tag"                   #some short prefix because labels below are prefixed with it
                  , lLabels                   # in conjugate, we accept all labels, andNone becomes "none"
                  , lIgnoredLabels
                  , True                # unused
                  , BBoxDeltaFun=lambda v: max(v * 0.066, min(5, v/3))  #we reduce overlap in this way
                  )    
    nt.setLabelAttribute("type")

    nt.setXpathExpr((".//pc:TextLine"
                      , ".//pc:TextEquiv")       #how to get their text
                   )
    DU_GRAPH.addNodeType(nt)
    
    return DU_GRAPH


def getConfiguredGraphClass_for_TextRegion(doer):
    """
    In this class method, we must return a configured graph class
    """
    #DU_GRAPH = ConjugateSegmenterGraph_MultiSinglePageXml  # consider each age as if indep from each other
    DU_GRAPH = Graph_MultiSinglePageXml
    
    assert bText == False, "Not implemented because most probably useless: extraction of text of TextRegion."
    ntClass = NodeType_PageXml_type_woText

    lLabels        = ['heading', 'paragraph', 'page-number']
    lIgnoredLabels = ['caption', 'marginalia']

    nt = ntClass("TR_tag"                   #some short prefix because labels below are prefixed with it
                  , lLabels                   # in conjugate, we accept all labels, andNone becomes "none"
                  , lIgnoredLabels
                  , True                # unused
                  , BBoxDeltaFun=lambda v: max(v * 0.066, min(5, v/3))  #we reduce overlap in this way
                  )    
    nt.setLabelAttribute("type")
    nt.setXpathExpr((".//pc:TextRegion"
                      , "")
                   )
    DU_GRAPH.addNodeType(nt)
    
    return DU_GRAPH


if __name__ == "__main__":
    #     import better_exceptions
    #     better_exceptions.MAX_LENGTH = None
    
    # standard command line options for CRF- ECN- GAT-based methods
    usage, parser = DU_Task_Factory.getStandardOptionsParser(sys.argv[0])

    traceln("VERSION: %s" % DU_Task_Factory.getVersion())

    # --- 
    #parse the command line
    (options, args) = parser.parse_args()

    if not bText:
        cFeatureDefinition = FeatureDefinition_Generic_noText if bGenericFeature else FeatureDefinition_PageXml_StandardOnes_noText
        dFeatureConfig = {}
    else:
        cFeatureDefinition = FeatureDefinition_Generic if bGenericFeature else FeatureDefinition_PageXml_StandardOnes
        if True:
            # this is the best configuration when using the text
            dFeatureConfig = {  'n_tfidf_node':400, 't_ngrams_node':(1,1), 'b_tfidf_node_lc':False
                            , 'n_tfidf_edge':400, 't_ngrams_edge':(1,1), 'b_tfidf_edge_lc':False
            }
        elif False:
            dFeatureConfig = {  'n_tfidf_node':1024, 't_ngrams_node':(1,2), 'b_tfidf_node_lc':False
                            , 'n_tfidf_edge':1024, 't_ngrams_edge':(1,2), 'b_tfidf_edge_lc':False
            }
        else:
            dFeatureConfig = {  'n_tfidf_node':400, 't_ngrams_node':(1,3), 'b_tfidf_node_lc':False
                            , 'n_tfidf_edge':400, 't_ngrams_edge':(1,3), 'b_tfidf_edge_lc':False
            }

    try:
        sModelDir, sModelName = args
    except Exception as e:
        traceln("Specify a model folder and a model name!")
        DU_Task_Factory.exit(usage, 1, e)

    doer = DU_Task_Factory.getDoer(sModelDir, sModelName
                                   , options                    = options
                                   , fun_getConfiguredGraphClass= getConfiguredGraphClass_for_TextRegion if bTextRegion else getConfiguredGraphClass_for_TextLine
                                   , cFeatureDefinition         = cFeatureDefinition
                                   , dFeatureConfig             = dFeatureConfig
                                   )
    
    # setting the learner configuration, in a standard way 
    # (from command line options, or from a JSON configuration file)
    dLearnerConfig = doer.getStandardLearnerConfig(options)
    
    # of course, you can put yours here instead.
    doer.setLearnerConfiguration(dLearnerConfig)

    #doer.setConjugateMode()
    
    # act as per specified in the command line (--trn , --fold-run, ...)
    doer.standardDo(options)
    
    del doer


    




