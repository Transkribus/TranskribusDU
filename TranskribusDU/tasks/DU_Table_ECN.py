# -*- coding: utf-8 -*-

"""
    DU task for Table based on ECN
    
    Copyright Xerox(C) 2018, 2019  Hervé Déjean, Jean-Luc Meunier, Animesh Prasad

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
import gcn.DU_Model_ECN  
from tasks.DU_ECN_Task import DU_ECN_Task


from crf.FeatureDefinition_PageXml_std_noText_v4 import FeatureDefinition_PageXml_StandardOnes_noText_v4
 
class DU_Table_ECN(DU_ECN_Task):
    """
    ECN Models
    """
    bHTR     = False  # do we have text from an HTR?
    bPerPage = False # do we work per document or per page?
    #bTextLine = False # if False then act as TextRegion

    sMetadata_Creator = "NLE Document Understanding ECN"
    sXmlFilenamePattern = "*.mpxml"

    # sLabeledXmlFilenamePattern = "*.a_mpxml"
    sLabeledXmlFilenamePattern = "*.mpxml" #"*mpxml"


    sLabeledXmlFilenameEXT = ".mpxml"

    dLearnerConfig = None

    #dLearnerConfig = {'nb_iter': 50,
    #                  'lr': 0.001,
    #                  'num_layers': 3,
    #                  'nconv_edge': 10,
    #                  'stack_convolutions': True,
    #                  'node_indim': -1,
    #                  'mu': 0.0,
    #                  'dropout_rate_edge': 0.0,
    #                  'dropout_rate_edge_feat': 0.0,
    #                  'dropout_rate_node': 0.0,
    #                  'ratio_train_val': 0.15,
    #                  #'activation': tf.nn.tanh, Problem I can not serialize function HERE
    #   }
    # === CONFIGURATION ====================================================================
    @classmethod
    def getConfiguredGraphClass(cls):
        """
        In this class method, we must return a configured graph class
        """
        #lLabels = ['heading', 'header', 'page-number', 'resolution-number', 'resolution-marginalia', 'resolution-paragraph', 'other']
        
        #lLabels = ['IGNORE', '577', '579', '581', '608', '32', '3431', '617', '3462', '3484', '615', '49', '3425', '73', '3', '3450', '2', '11', '70', '3451', '637', '77', '3447', '3476', '3467', '3494', '3493', '3461', '3434', '48', '3456', '35', '3482', '74', '3488', '3430', '17', '613', '625', '3427', '3498', '29', '3483', '3490', '362', '638a', '57', '616', '3492', '10', '630', '24', '3455', '3435', '8', '15', '3499', '27', '3478', '638b', '22', '3469', '3433', '3496', '624', '59', '622', '75', '640', '1', '19', '642', '16', '25', '3445', '3463', '3443', '3439', '3436', '3479', '71', '3473', '28', '39', '361', '65', '3497', '578', '72', '634', '3446', '627', '43', '62', '34', '620', '76', '23', '68', '631', '54', '3500', '3480', '37', '3440', '619', '44', '3466', '30', '3487', '45', '61', '3452', '3491', '623', '633', '53', '66', '67', '69', '643', '58', '632', '636', '7', '641', '51', '3489', '3471', '21', '36', '3468', '4', '576', '46', '63', '3457', '56', '3448', '3441', '618', '52', '3429', '3438', '610', '26', '609', '3444', '612', '3485', '3465', '41', '20', '3464', '3477', '3459', '621', '3432', '60', '3449', '626', '628', '614', '47', '3454', '38', '3428', '33', '12', '3426', '3442', '3472', '13', '639', '3470', '611', '6', '40', '14', '3486', '31', '3458', '3437', '3453', '55', '3424', '3481', '635', '64', '629', '3460', '50', '9', '18', '42', '3495', '5', '580']
        #lLabels = [ str(i) for i in range(0,5000)]
        lLabels = [ str("%d_%d"%(t,i)) for t in range(0,10) for i in range(0,5000)]            
        lLabels.append('IGNORE')
        # traceln (lLabels)

        lIgnoredLabels = None

        """
        if you play with a toy collection, which does not have all expected classes, you can reduce those.
        """
        if cls.bPerPage:
            DU_GRAPH = Graph_MultiSinglePageXml  # consider each age as if indep from each other
        else:
            DU_GRAPH = Graph_MultiPageXml

        lActuallySeen = None
        if lActuallySeen:
            traceln("REDUCING THE CLASSES TO THOSE SEEN IN TRAINING")
            lIgnoredLabels = [lLabels[i] for i in range(len(lLabels)) if i not in lActuallySeen]
            lLabels = [lLabels[i] for i in lActuallySeen]
            traceln(len(lLabels), lLabels)
            traceln(len(lIgnoredLabels), lIgnoredLabels)
        
        if cls.bHTR:
            ntClass = NodeType_PageXml_type
        else:
            #ignore text
            ntClass = NodeType_PageXml_type_woText

        # DEFINING THE CLASS OF GRAPH WE USE
        nt = ntClass("cell"  # some short prefix because labels below are prefixed with it
                      , lLabels
                      , lIgnoredLabels
                      , False  # no label means OTHER
                      , BBoxDeltaFun=lambda v: max(v * 0.066, min(5, v / 3))
                      # we reduce overlap in this way
                      )

        nt.setLabelAttribute("cell")
        nt.setXpathExpr( (".//pc:TextLine"        #how to find the nodes            
        #nt.setXpathExpr( (".//pc:TableCell//pc:TextLine"        #how to find the nodes
                      , "./pc:TextEquiv")
                   )

        DU_GRAPH.addNodeType(nt)

        return DU_GRAPH

    def __init__(self, sModelName, sModelDir, sComment=None,dLearnerConfigArg=None):
        traceln ( self.bHTR)

        if self.bHTR:
            cFeatureDefinition = FeatureDefinition_PageXml_StandardOnes
            dFeatureConfig = { 'bMultiPage':False, 'bMirrorPage':False
                          , 'n_tfidf_node':300, 't_ngrams_node':(2,4), 'b_tfidf_node_lc':False
                          , 'n_tfidf_edge':300, 't_ngrams_edge':(2,4), 'b_tfidf_edge_lc':False }
        else:
            cFeatureDefinition = FeatureDefinition_PageXml_StandardOnes_noText_v4
            # cFeatureDefinition = FeatureDefinition_PageXml_NoNodeFeat_v3                 
            dFeatureConfig = {}

        if sComment is None: sComment  = sModelName

        if  dLearnerConfigArg is not None and "ecn_ensemble" in dLearnerConfigArg:
            traceln('ECN_ENSEMBLE')
            DU_ECN_Task.__init__(self
                                 , sModelName, sModelDir
                                 , dFeatureConfig=dFeatureConfig
                                 , dLearnerConfig=self.dLearnerConfig if dLearnerConfigArg is None else dLearnerConfigArg
                                 , sComment=sComment
                                 , cFeatureDefinition= cFeatureDefinition
                                 , cModelClass=gcn.DU_Model_ECN.DU_Ensemble_ECN
                                 )
        else:
            #Default Case Single Model
            DU_ECN_Task.__init__(self
                                 , sModelName, sModelDir
                                 , dFeatureConfig=dFeatureConfig
                                 , dLearnerConfig=self.dLearnerConfig if dLearnerConfigArg is None else dLearnerConfigArg
                                 , sComment= sComment
                                 , cFeatureDefinition=cFeatureDefinition
                                 )

        #if options.bBaseline:
        #    self.bsln_mdl = self.addBaseline_LogisticRegression()  # use a LR model trained by GridSearch as baseline

    # === END OF CONFIGURATION =============================================================
    def predict(self, lsColDir):
        """
        Return the list of produced files
        """
        self.sXmlFilenamePattern = "*.mpxml"
        return DU_ECN_Task.predict(self, lsColDir)
    
    

