# -*- coding: utf-8 -*-

"""
    First DU task for StAZH
    
    Copyright Xerox(C) 2016 JL. Meunier
    Copyright Naver (C) 2019 H. Déjean
"""
import sys, os

import TranskribusDU_version  # if import error, updade the PYTHONPATH environment variable

from common.trace import traceln
from tasks.DU_Task_Factory                          import DU_Task_Factory
from graph.Graph_Multi_SinglePageXml                import Graph_MultiSinglePageXml
from graph.NodeType_PageXml                         import   NodeType_PageXml_type
from graph.FeatureDefinition_PageXml_std            import FeatureDefinition_PageXml_StandardOnes
from graph.NodeType_PageXml                         import  NodeType_PageXml_type_woText, NodeType_PageXml_type
from graph.FeatureDefinition_PageXml_std_noText  import FeatureDefinition_PageXml_StandardOnes_noText


def getConfiguredGraphClass(doer):
    """
    In this class method, we must return a configured graph class
    """
    #DU_GRAPH = ConjugateSegmenterGraph_MultiSinglePageXml  # consider each age as if indep from each other
    DU_GRAPH = Graph_MultiSinglePageXml
    
    ntClass = NodeType_PageXml_type_woText #NodeType_PageXml_type

    #lIgnoredLabels = ['menu-section-heading','Item-number']

    lLabels = ['catch-word', 'header', 'heading', 'marginalia', 'page-number'] 

    nt = ntClass("TR"                   #some short prefix because labels below are prefixed with it
                  , lLabels                   # in conjugate, we accept all labels, andNone becomes "none"
                  , []
                  , True                # unused
                  , BBoxDeltaFun=lambda v: max(v * 0.066, min(5, v/3))  #we reduce overlap in this way
                  )    
    nt.setLabelAttribute("type")
    #DU_GRAPH.sEdgeLabelAttribute="TR"
    nt.setXpathExpr((".//pc:TextRegion"
                      , ".//pc:TextEquiv")       #how to get their text
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

    cFeatureDefinition = FeatureDefinition_PageXml_StandardOnes_noText
#     dFeatureConfig = {  
#                                'n_tfidf_node':400, 't_ngrams_node':(1,3), 'b_tfidf_node_lc':False
#                               , 'n_tfidf_edge':400, 't_ngrams_edge':(1,3), 'b_tfidf_edge_lc':False }
    try:
        sModelDir, sModelName = args
    except Exception as e:
        traceln("Specify a model folder and a model name!")
        DU_Task_Factory.exit(usage, 1, e)

    doer = DU_Task_Factory.getDoer(sModelDir, sModelName
                                   , options                    = options
                                   , fun_getConfiguredGraphClass= getConfiguredGraphClass
                                   , cFeatureDefinition         = cFeatureDefinition
                                   , dFeatureConfig             = {}                                           
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


    




