# -*- coding: utf-8 -*-

"""
    Tagging TextLine for Annemieke's StABccollection
    
    Copyright Xerox(C) 2016 JL. Meunier
    Copyright Naver(C) 2019 H. Déjean
"""
import sys, os

import TranskribusDU_version  # if import error, updade the PYTHONPATH environment variable

from common.trace import traceln
from xml_formats.PageXml import PageXml
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
bText       = True         # do we use text?  (set to False for TextRegion)
bGenericFeature = True      # do not use page width and height
bBB2        = False          # slightly different way to normalize bounding boxes
# =============================================================================


class NodeType_for_TextLine(NodeType_PageXml_type):
    """
    We specialize it because we get the label (of TextLine nodes) from the parent node (TextRegion)
    """
    def getLabel(self, domnode):
        return domnode.getparent().get(self.sLabelAttr)

    def setDocNodeLabel(self, graph_node, sLabel):
        """
        Set the DOM node label in the format-dependent way
        """
        try:
            sXmlLabel = self.dLabel2XmlLabel[sLabel]
        except KeyError:
            sXmlLabel = "other"
        graph_node.node.set(self.sLabelAttr, sXmlLabel)        
        # tagging using the custem XML attribute
        PageXml.setCustomAttr(graph_node.node, "structure", "type", sXmlLabel)


class NodeType_for_TextRegion(NodeType_PageXml_type_woText  ):
    """
    We specialize it, because we want to tag also the inner TextLine
    """
    def setDocNodeLabel(self, graph_node, sLabel):
        """
        Set the DOM node label in the format-dependent way
        """
        super(NodeType_for_TextRegion, self).setDocNodeLabel(graph_node, sLabel)  # tagging the TextRegion with @type

        # tagging inner TextLine
        sLabel = str(graph_node.node.get(self.sLabelAttr))
        dNS = {"pc":"http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
        lNd = graph_node.node.xpath(".//pc:TextLine", namespaces=dNS)
        for nd in lNd:
            nd.set(self.sLabelAttr, sLabel)
            PageXml.setCustomAttr(nd, "structure", "type", sLabel)

        return sLabel


def getConfiguredGraphClass_for_TextLine(doer):
    """
    we return a configured graph class for tagging TextLine
    """
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
    DU_GRAPH = Graph_MultiSinglePageXml
    
    assert bText == False, "Not implemented because most probably useless: extraction of text of TextRegion."
    ntClass = NodeType_for_TextRegion  #  NodeType_PageXml_type_woText

    lLabels        = ['heading', 'paragraph', 'page-number']
    lIgnoredLabels = ['caption', 'marginalia']

    nt = ntClass("TR_tag"                   #some short prefix because labels below are prefixed with it
                  , lLabels                   # in conjugate, we accept all labels, andNone becomes "none"
                  , lIgnoredLabels
                  , True                # unused
                  , BBoxDeltaFun=None if bBB2 else lambda v: max(v * 0.066, min(5, v/3))  #we reduce overlap in this way
                  , bPreserveWidth=True if bBB2 else False
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

    parser.add_option("--TextRegion"     , dest='bTextRegion' , action="store_true"
                        , default=False, help="Tag TextRegion, instead of TextLine") 
    parser.add_option("--no_text"     , dest='bNoText'   , action="store_true"
                        , default=False, help="Do not use text, otherwise use utext unigrams") 

    traceln("VERSION: %s" % DU_Task_Factory.getVersion())

    # --- 
    #parse the command line
    (options, args) = parser.parse_args()
    bTextRegion = options.bTextRegion
    bText       = not(options.bNoText)
    traceln("bTextRegion=%s  bText=%s" % (bTextRegion, bText))

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
        if bGenericFeature: 
            for _s in ['n_tfidf_edge', 't_ngrams_edge', 'b_tfidf_edge_lc']: del dFeatureConfig[_s]

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


    




