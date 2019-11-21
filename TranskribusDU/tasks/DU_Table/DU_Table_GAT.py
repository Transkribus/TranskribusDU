# -*- coding: utf-8 -*-

"""
    DU task for Table based on ECN
    
    Copyright NAVER(C) 2018, 2019  Hervé Déjean, Jean-Luc Meunier, Animesh Prasad


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""




import sys, os

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from common.trace import traceln

from crf.Graph_MultiPageXml import Graph_MultiPageXml
from crf.Graph_Multi_SinglePageXml import Graph_MultiSinglePageXml
from crf.NodeType_PageXml   import NodeType_PageXml_type_woText, NodeType_PageXml_type
from crf.FeatureDefinition_PageXml_std import FeatureDefinition_PageXml_StandardOnes
from crf.FeatureDefinition_PageXml_std_noText import FeatureDefinition_PageXml_StandardOnes_noText
from tasks.DU_ECN_Task import DU_ECN_Task


class DU_Table_GAT(DU_ECN_Task):
    """
    ECN Models
    """
    bHTR     = True  # do we have text from an HTR?
    bPerPage = True # do we work per document or per page?
    bTextLine = True # if False then act as TextRegion

    sMetadata_Creator = "NLE Document Understanding GAT"


    sXmlFilenamePattern = "*.bar_mpxml"

    # sLabeledXmlFilenamePattern = "*.a_mpxml"
    sLabeledXmlFilenamePattern = "*.bar_mpxml"

    sLabeledXmlFilenameEXT = ".bar_mpxml"


    dLearnerConfigOriginalGAT ={
        'nb_iter': 500,
        'lr': 0.001,
        'num_layers': 2,#2 Train Acc is lower 5 overfit both reach 81% accuracy on Fold-1
        'nb_attention': 5,
        'stack_convolutions': True,
        # 'node_indim': 50   , worked well 0.82
        'node_indim': -1,
        'dropout_rate_node': 0.0,
        'dropout_rate_attention': 0.0,
        'ratio_train_val': 0.15,
        "activation_name": 'tanh',
        "patience": 50,
        "mu": 0.00001,
        "original_model" : True

    }


    dLearnerConfigNewGAT = {'nb_iter': 500,
                      'lr': 0.001,
                      'num_layers': 5,
                      'nb_attention': 5,
                      'stack_convolutions': True,
                      'node_indim': -1,
                      'dropout_rate_node': 0.0,
                      'dropout_rate_attention'  : 0.0,
                      'ratio_train_val': 0.15,
                      "activation_name": 'tanh',
                      "patience":50,
                      "original_model": False,
                      "attn_type":0
       }
    dLearnerConfig = dLearnerConfigNewGAT
    #dLearnerConfig = dLearnerConfigOriginalGAT
    # === CONFIGURATION ====================================================================
    @classmethod
    def getConfiguredGraphClass(cls):
        """
        In this class method, we must return a configured graph class
        """
        lLabels = ['heading', 'header', 'page-number', 'resolution-number', 'resolution-marginalia', 'resolution-paragraph', 'other']

        lIgnoredLabels = None

        """
        if you play with a toy collection, which does not have all expected classes, you can reduce those.
        """

        lActuallySeen = None
        if lActuallySeen:
            traceln("REDUCING THE CLASSES TO THOSE SEEN IN TRAINING")
            lIgnoredLabels = [lLabels[i] for i in range(len(lLabels)) if i not in lActuallySeen]
            lLabels = [lLabels[i] for i in lActuallySeen]
            traceln(len(lLabels), lLabels)
            traceln(len(lIgnoredLabels), lIgnoredLabels)


        # DEFINING THE CLASS OF GRAPH WE USE
        if cls.bPerPage:
            DU_GRAPH = Graph_MultiSinglePageXml  # consider each age as if indep from each other
        else:
            DU_GRAPH = Graph_MultiPageXml

        if cls.bHTR:
            ntClass = NodeType_PageXml_type
        else:
            #ignore text
            ntClass = NodeType_PageXml_type_woText


        nt = ntClass("bar"  # some short prefix because labels below are prefixed with it
                                          , lLabels
                                          , lIgnoredLabels
                                          , False  # no label means OTHER
                                          , BBoxDeltaFun=lambda v: max(v * 0.066, min(5, v / 3))
                                          # we reduce overlap in this way
                                          )
        nt.setLabelAttribute("DU_sem")
        if cls.bTextLine:
            nt.setXpathExpr( (".//pc:TextRegion/pc:TextLine"        #how to find the nodes
                      , "./pc:TextEquiv")
                   )
        else:
            nt.setXpathExpr( (".//pc:TextRegion"        #how to find the nodes
                      , "./pc:TextEquiv")       #how to get their text
                   )


        DU_GRAPH.addNodeType(nt)

        return DU_GRAPH

    def __init__(self, sModelName, sModelDir, sComment=None,dLearnerConfigArg=None):
        if self.bHTR:
            cFeatureDefinition = FeatureDefinition_PageXml_StandardOnes
            dFeatureConfig = { 'bMultiPage':False, 'bMirrorPage':False
                          , 'n_tfidf_node':500, 't_ngrams_node':(2,4), 'b_tfidf_node_lc':False
                          , 'n_tfidf_edge':250, 't_ngrams_edge':(2,4), 'b_tfidf_edge_lc':False }
        else:
            cFeatureDefinition = FeatureDefinition_PageXml_StandardOnes_noText
            dFeatureConfig = { 'bMultiPage':False, 'bMirrorPage':False
                          , 'n_tfidf_node':None, 't_ngrams_node':None, 'b_tfidf_node_lc':None
                          , 'n_tfidf_edge':None, 't_ngrams_edge':None, 'b_tfidf_edge_lc':None }


        if sComment is None: sComment  = sModelName


        DU_ECN_Task.__init__(self
                             , sModelName, sModelDir
                             , dFeatureConfig=dFeatureConfig
                             , dLearnerConfig= dLearnerConfigArg if dLearnerConfigArg is not None else self.dLearnerConfig
                             , sComment=sComment
                             , cFeatureDefinition=cFeatureDefinition
                             , cModelClass=DU_Model_GAT
                             )

        if options.bBaseline:
            self.bsln_mdl = self.addBaseline_LogisticRegression()  # use a LR model trained by GridSearch as baseline

    # === END OF CONFIGURATION =============================================================
    def predict(self, lsColDir):
        """
        Return the list of produced files
        """
        self.sXmlFilenamePattern = "*.bar_mpxml"
        return DU_ECN_Task.predict(self, lsColDir)

