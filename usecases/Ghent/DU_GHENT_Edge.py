# -*- coding: utf-8 -*-

"""
    DU task for segmenting text in resoluation using conjugate 
    graph after the SW re-engineering by JLM during the 2019 summer.
    
    As of June 5th, 2015, this is the exemplary code
    
    Copyright NAVER(C)  2019 Hervé Déjean 
    
    



    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""
 
import sys, os
import numpy as np 
import TranskribusDU_version

from common.trace import traceln

from graph.NodeType_PageXml     import defaultBBoxDeltaFun
from graph.NodeType_PageXml     import NodeType_PageXml_type
from graph.NodeType_PageXml     import NodeType_PageXml_type_woText
from tasks.DU_Task_Factory      import DU_Task_Factory
from tasks.DU_Task_Features     import Features_June19_Simple
from tasks.DU_Task_Features     import Features_June19_Simple_Separator
from tasks.DU_Task_Features     import Features_June19_Simple_Shift
from tasks.DU_Task_Features     import Features_June19_Simple_Separator_Shift
from tasks.DU_Task_Features     import Features_June19_Full
from tasks.DU_Task_Features     import Features_June19_Full_Separator
from tasks.DU_Task_Features     import Features_June19_Full_Shift
from tasks.DU_Task_Features     import Features_June19_Full_Separator_Shift
from tasks.DU_Task_Features     import FeatureDefinition
from tasks.DU_Task_Features     import *
from graph.Transformer       import Transformer

from graph.pkg_GraphBinaryConjugateSegmenter.MultiSinglePageXml \
    import MultiSinglePageXml \
    as ConjugateSegmenterGraph_MultiSinglePageXml

from graph.pkg_GraphBinaryConjugateSegmenter.MultiSinglePageXml_Separator \
    import MultiSinglePageXml_Separator \
    as ConjugateSegmenterGraph_MultiSinglePageXml_Separator


class NodeSemanticLabels(Transformer): 
    """
    Semantic labels as features (@type)
    """
    def transform(self, lNode):
#         labels = [ 'resolution-marginalia', 'resolution-number', 'resolution-paragraph',
#                    'page-number',  'heading', 'header', 'other', 'IGNORE']
        labels= ['paragraph','heading','paragraph_left','paragraph_right','None','IGNORE']
        #labels= ['paragraph','heading','footnote,'None','IGNORE']

        aa = np.zeros( ( len(lNode), len(labels) ) , dtype=np.float64)  

        for i, blk in enumerate(lNode):
            try:
                s = blk.node.attrib['type']
            except:
                s = 'IGNORE'
            aa[i, labels.index(s)] = 1.0
        return aa
class Features_BAR_Full(FeatureDefinition):
    """
    All features we had historically (some specific to CRF):
        NODE: geometry, neighbor count, text
        EDGE: type, constant 1, geometry, text of source and target nodes
        The features of the edges are shifted by class, apart the 1-hot ones.
    """
    
    n_QUANTILES = 16
    
    bShiftEdgeByClass = False
    bSeparator        = False
    
    def __init__(self): 
        FeatureDefinition.__init__(self)
        
        # NODES
        self.lNodeFeature = [                               \
              ("geometry"           , Node_Geometry())         # one can set nQuantile=...
            , ("neighbor_count"     , Node_Neighbour_Count())  # one can set nQuantile=...
            , ("text"               , Node_Text_NGram( 'char'    # character n-grams
                                                       , 500     # number of N-grams
                                                       , (1,3)    # N
                                                       , False    # lowercase?))
                                                       ))
            #,('sem',NodeSemanticLabels())
            
            ]
        node_transformer = FeatureUnion(self.lNodeFeature)
        
        # EDGES
        # which types of edge can we get??
        # It depends on the type of graph!!
        lEdgeClass = [HorizontalEdge, VerticalEdge]
        # standard set of features, including a constant 1 for CRF
        self.lEdgeFeature = [                                            \
                  ('1hot'   , Edge_Type_1Hot(lEdgeClass=lEdgeClass)) # Edge class 1 hot encoded (PUT IT FIRST)
                , ('1'      , Edge_1())                              # optional constant 1 for CRF
                , ('geom'   , Edge_Geometry())                       # one can set nQuantile=...
                , ('src_txt', Edge_Source_Text_NGram( 'char'    # character n-grams
                                               , 50     # number of N-grams
                                               , (1,2)    # N
                                               , False    # lowercase?))
                                               ))
                , ('tgt_txt', Edge_Target_Text_NGram( 'char'    # character n-grams
                                               , 50     # number of N-grams
                                               , (1,2)    # N
                                               , False    # lowercase?))
                                               ))
                
                            ]
        if self.bSeparator:
            self.lEdgeFeature = self.lEdgeFeature + [ 
                  ('sprtr_bool', Separator_boolean())
                , ('sprtr_num' , Separator_num())
                            ]
        fu =  FeatureUnion(self.lEdgeFeature)        
        
        # you can use directly this union of features!
        edge_transformer = fu
        
        # OPTIONNALLY, you can have one range of features per type of edge.
        # the 1-hot encoding must be the first part of the union and it will determine
        #   by how much the rest of the feature are shifted.
        #
        # IMPORTANT: 1hot is first of union   AND   the correct number of edge classes
        if self.bShiftEdgeByClass:
            ppl = Pipeline([
                      ('fu', fu)
                    , ('shifter', EdgeClassShifter(len(lEdgeClass)))
                    ])
            edge_transformer = ppl
        
        self.setTransformers(node_transformer, edge_transformer)
# ----------------------------------------------------------------------------
# class My_ConjugateNodeType(NodeType_PageXml_type_woText):
class My_ConjugateNodeType(NodeType_PageXml_type):
    """
    We need this to extract properly the label from the label attribute of the (parent) TableCell element.
    """
    def __init__(self, sNodeTypeName, lsLabel, lsIgnoredLabel=None, bOther=True, BBoxDeltaFun=defaultBBoxDeltaFun):
        super(My_ConjugateNodeType, self).__init__(sNodeTypeName, lsLabel, lsIgnoredLabel, bOther, BBoxDeltaFun)

    def parseDomNodeLabel(self, domnode, defaultCls=None):
        """
        Parse and set the graph node label and return its class index
        We rely on the standard self.sLabelAttr
        raise a ValueError if the label is missing while bOther was not True
         , or if the label is neither a valid one nor an ignored one
        """
        ndParent = domnode.getparent()
        sLabel = "%s" % (
                            ndParent.get(self.sLabelAttr)   #  id
                            )
        #domnode.set('type',sLabel)
        return sLabel

    def setDomNodeLabel(self, domnode, sLabel):
        raise Exception("This should not occur in conjugate mode")    
    


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
        #nt.setXpathExpr( (".//pc:TextRegion/pc:TextLine[@DU_num]"        #how to find the nodes
        #nt.setXpathExpr( (".//pc:TextLine"        #how to find the nodes            
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
                cFeatureDefinition = Features_BAR_Full #Features_June19_Full 
                
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
    dLearnerConfig['balanced'] = True
    
    # of course, you can put yours here instead.
    doer.setLearnerConfiguration(dLearnerConfig)

    # === CONJUGATE MODE ===
    doer.setConjugateMode()
    
    # === GO!! ===
    # act as per specified in the command line (--trn , --fold-run, ...)
    doer.standardDo(options)
    
    del doer

    
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    #     import better_exceptions
    #     better_exceptions.MAX_LENGTH = None
    
    main(sys.argv[0], "id", My_ConjugateNodeType)
