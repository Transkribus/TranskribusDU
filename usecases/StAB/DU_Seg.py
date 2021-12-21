# -*- coding: utf-8 -*-

"""
    DU task for the STAB dataset: TextRegion segmentation task
    
    Copyright NAVER LABS Europe(C)  2021 Hervé Déjean, JL Meunier
    
    
"""
 
import sys, os
from collections import Counter

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version
TranskribusDU_version

from common.trace import traceln
from xml_formats.PageXml import PageXml

from tasks.DU_Task_Factory                          import DU_Task_Factory
from graph.NodeType_PageXml                         import NodeType_PageXml_type
from graph.NodeType_PageXml                         import defaultBBoxDeltaFun
from graph.pkg_GraphBinaryConjugateSegmenter.MultiSinglePageXml  \
            import MultiSinglePageXml as ConjugateSegmenterGraph_MultiSinglePageXml 

from graph.FeatureDefinition_PageXml_std            import FeatureDefinition_PageXml_StandardOnes
from graph.FeatureDefinition_PageXml_std_noText     import FeatureDefinition_PageXml_StandardOnes_noText
from graph.FeatureDefinition_Generic                import FeatureDefinition_Generic
from graph.FeatureDefinition_Generic_noText         import FeatureDefinition_Generic_noText

from xml_formats.PageXml                            import PageXml, PageXmlException
from util.Shape                                     import ShapeLoader

#from funsd import parserAddSharedOptions, getDataToPickle, selectFeatureStuff

# =============================================================================
bText       = True         # do we use text?  (set to False for TextRegion)
bGenericFeature = True      # do not use page width and height
bBB2        = False          # slightly different way to normalize bounding boxes
# =============================================================================


# ----------------------------------------------------------------------------
class My_ConjugateSegmenterGraph_MultiSinglePageXml(ConjugateSegmenterGraph_MultiSinglePageXml):
    """
    #HACK to add a @line_id attribute to words, since we did not add it when creating the XML
    """
    def __init__(self):
        super(My_ConjugateSegmenterGraph_MultiSinglePageXml, self).__init__()
    
    # def isEmpty(self):
    #     traceln("\tHACK to create a @line_id on nodes")
    #     for nd in self.lNode:
    #         nd.node.set("line_id", nd.node.getparent().get("id"))
    #     return super(My_ConjugateSegmenterGraph_MultiSinglePageXml, self).isEmpty()

    def form_cluster(self, Y_proba, fThres=0.5, bAgglo=True):
        """
        We specialize this method because we want to preserve the page reading order.
        So, whenever a cluster does not respect the reading order, we split it in sub-clusters
        """
        lCluster = super(My_ConjugateSegmenterGraph_MultiSinglePageXml, self).form_cluster(Y_proba, fThres=fThres, bAgglo=bAgglo)

        # lCLuster is a partition of the graph nodes
        # let's create a reverse mapping node_index --> cluster_index
        dN2C = {}
        # make sure we have a list, not a Set, to have an index
        lCLuster = list(lCluster)
        for ic, c in enumerate(lCLuster):
            for ind in c:
                dN2C[ind] = ic

        # Now re-create the clusters
        # note: self.lnode does respect the document order
        lNewCluster = list()
        cur_ic, cur_c = -1, None
        for ind, nd in enumerate(self.lNode):
            ic = dN2C[ind]
            if ic != cur_ic:   # need to create a new cluster
                cur_ic, cur_c = ic, [ind]
                lNewCluster.append(cur_c)
            else:
                cur_c.append(ind)  # add node at end of the cluster being fed
        
        # now we have cluster and cluster's content in same order as document
        return lNewCluster
 
    def addClusterToDom(self, lCluster, sAlgo=""):
        """
        From the predicted clusters, we create new TextRegions.
        The original TextRegion, if any, are either removed or renamed TextRegion_GT 
        """
        if hasattr(options, "bEvalRegion") and options.bEvalRegion:
            # in order to evaluate, we must keep the original TextRegion and the Cluster elements that are produced
            return super(My_ConjugateSegmenterGraph_MultiSinglePageXml, self).addClusterToDom(lCluster, sAlgo=sAlgo)
        
        lNdCluster = []
        dNS = {"pc":"http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}

        # the graph has been constructed for a certain page
        pageNode = self.lNode[0].page.node
        # guess its position to create unique XML ID
        #pageNum = 1 #+ len( pageNode.xpath(".//preceding::pc:Page", namespaces=dNS) )
        #traceln(" Page: ",pageNum)
        # enumerate TextAreas to remove
        lNdOldTextRegion = pageNode.xpath(".//pc:TextRegion", namespaces=dNS)
        
        # we copy the type attribute to the inner TextLine, to preserve the GT info, if any
        for ndTR in lNdOldTextRegion:
            sType = str(ndTR.get("type"))
            for ndTL in ndTR.xpath(".//pc:TextLine", namespaces=dNS):
                    ndTL.set("type_gt", sType)


        # replace the ReadingOrder section by a new one
        ndReadingOrder = pageNode.xpath(".//pc:ReadingOrder", namespaces=dNS)[0]
        pageNode.remove(ndReadingOrder)
        ndReadingOrder = PageXml.createPageXmlNode('ReadingOrder')  
        ndOrderedGroup = PageXml.createPageXmlNode('OrderedGroup')
        ndOrderedGroup.set("id", "ro_1")
        ndOrderedGroup.set("caption", "Regions reading order")
        ndReadingOrder.append(ndOrderedGroup)
        pageNode.append(ndReadingOrder)

        # loop over clusters
        for ic, c in enumerate(lCluster):
            ndCluster = PageXml.createPageXmlNode('TextRegion')  
            #scid = "cluster_p%d_%d" % (pageNum, ic+1)
            scid = "cluster_%d" % (ic+1)
            ndCluster.set("id", scid)  
            ndCluster.set("custom", "readingOrder {index:%d;}" % ic)  

            # TextRegion bounding box
            coords = PageXml.createPageXmlNode('Coords')        
            ndCluster.append(coords)
            spoints = ShapeLoader.minimum_rotated_rectangle([self.lNode[_i].node for _i in c])
            coords.set('points',spoints)   

            # if the inner TextLine are tagged, let's do a vote to tag the Cluster
            lsType = [self.lNode[_i].node.get('type') for _i in c]
            dType = Counter([o for o in lsType if o is not None])
            mc = dType.most_common(1)
            if mc:
                sXmlLabel = mc[0][0]
                ndCluster.set("type", sXmlLabel)
                PageXml.setCustomAttr(ndCluster, "structure", "type", sXmlLabel)

            #TextLine: move the DOM node of the content to the cluster
            for _i in c:                               
                ndCluster.append(self.lNode[_i].node)
            
            pageNode.append(ndCluster)
            ndCluster.tail = "\n"
            lNdCluster.append(ndCluster)

            ndRegionRefIndexed = PageXml.createPageXmlNode('RegionRefIndexed')
            ndRegionRefIndexed.set("index", str(ic))
            ndRegionRefIndexed.set("regionRef", scid)
            ndRegionRefIndexed.tail = "\n"
            ndOrderedGroup.append(ndRegionRefIndexed)

        # remove or rename the old TextRegion
        for nd in lNdOldTextRegion: 
            if False:
                nd.tag = "TextRegion_GT"
            else:
                #pageNode.remove(nd)        
                nd.getparent().remove(nd)
                
        return lNdCluster

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
        sLabel = domnode.getparent().get(self.sLabelAttr)  # this is an XML id, infact!

        return str(sLabel)

    def setDocNodeLabel(self, graph_node, sLabel):
        raise Exception("This should not occur in conjugate mode")    

    
def getConfiguredGraphClass(doer):
    """
    In this class method, we must return a configured graph class
    """
    #DU_GRAPH = ConjugateSegmenterGraph_MultiSinglePageXml
    DU_GRAPH = My_ConjugateSegmenterGraph_MultiSinglePageXml

    ntClass = My_ConjugateNodeType

    nt = ntClass("region"            #some short prefix because labels below are prefixed with it
                  , []                   # in conjugate, we accept all labels, andNone becomes "none"
                  , []
                  , False                # unused
                  #, BBoxDeltaFun=lambda v: max(v * 0.066, min(5, v/3))  #we reduce overlap in this way
                  , BBoxDeltaFun=None if bBB2 else lambda v: max(v * 0.066, min(5, v/3))  #we reduce overlap in this way
                  , bPreserveWidth=True if bBB2 else False
                  )    
    nt.setLabelAttribute("id")
    
    ## HD added 23/01/2020: needed for output generation
    DU_GRAPH.clusterType='region'
    nt.setXpathExpr((  ".//pc:TextLine"
                     , "./pc:TextEquiv")       #how to get their text
                     )
    DU_GRAPH.addNodeType(nt)
    
    return DU_GRAPH


if __name__ == "__main__":
    traceln("VERSION: %s" % DU_Task_Factory.getVersion())
    
    # standard command line options for CRF- ECN- GAT-based methods
    usage, parser = DU_Task_Factory.getStandardOptionsParser(sys.argv[0])
    
    
    (options, args) = parser.parse_args()
    #cFeatureDefinition, dFeatureConfig = selectFeatureStuff(options)

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
            dFeatureConfig = {  'n_tfidf_node':512, 't_ngrams_node':(1,2), 'b_tfidf_node_lc':False
                            , 'n_tfidf_edge':512, 't_ngrams_edge':(1,2), 'b_tfidf_edge_lc':False
            }
        else:
            dFeatureConfig = {  'n_tfidf_node':1024, 't_ngrams_node':(1,3), 'b_tfidf_node_lc':False
                            , 'n_tfidf_edge':1024, 't_ngrams_edge':(1,3), 'b_tfidf_edge_lc':False
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

