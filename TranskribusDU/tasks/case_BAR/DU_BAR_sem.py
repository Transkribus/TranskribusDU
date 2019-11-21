# -*- coding: utf-8 -*-

"""
    DU task for BAR - see https://read02.uibk.ac.at/wiki/index.php/Document_Understanding_BAR
    
    Copyright Xerox(C) 2017 JL Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""

import sys, os

import json

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from common.trace import traceln

from crf.Graph_MultiPageXml import Graph_MultiPageXml
from crf.Graph_Multi_SinglePageXml import Graph_MultiSinglePageXml
from crf.NodeType_PageXml   import NodeType_PageXml_type_woText, NodeType_PageXml_type
from tasks.DU_CRF_Task import DU_CRF_Task
from crf.FeatureDefinition_PageXml_std import FeatureDefinition_PageXml_StandardOnes
from graph.FeatureDefinition_PageXml_std_noText import FeatureDefinition_PageXml_StandardOnes_noText
from tasks import _checkFindColDir, _exit


from crf.FeatureDefinition_PageXml_std_noText import FeatureDefinition_PageXml_StandardOnes_noText

from gcn.DU_Model_ECN import DU_Model_GAT


from tasks.DU_BAR import main as m
 
class DU_BAR_sem(DU_CRF_Task):
    """
    We will do a typed CRF model for a DU task
    , with the below labels 
    """
    sLabeledXmlFilenamePattern = "*.mpxml"  #"*.bar_mpxml"

    bHTR     = False  # do we have text from an HTR?
    bPerPage = False # do we work per document or per page?
    bTextLine = True # if False then act as TextRegion
    
    #=== CONFIGURATION ====================================================================
    @classmethod
    def getConfiguredGraphClass(cls):
        """
        In this class method, we must return a configured graph class
        """
        #DEFINING THE CLASS OF GRAPH WE USE
        if cls.bPerPage:
            DU_GRAPH = Graph_MultiSinglePageXml  # consider each age as if indep from each other
        else:
            DU_GRAPH = Graph_MultiPageXml

        #lLabels1 = ['heading', 'header', 'page-number', 'resolution-number', 'resolution-marginalia', 'resolution-paragraph', 'other']
  
        lLabels1 = ['IGNORE', '577', '579', '581', '608', '32', '3431', '617', '3462', '3484', '615', '49', '3425', '73', '3', '3450', '2', '11', '70', '3451', '637', '77', '3447', '3476', '3467', '3494', '3493', '3461', '3434', '48', '3456', '35', '3482', '74', '3488', '3430', '17', '613', '625', '3427', '3498', '29', '3483', '3490', '362', '638a', '57', '616', '3492', '10', '630', '24', '3455', '3435', '8', '15', '3499', '27', '3478', '638b', '22', '3469', '3433', '3496', '624', '59', '622', '75', '640', '1', '19', '642', '16', '25', '3445', '3463', '3443', '3439', '3436', '3479', '71', '3473', '28', '39', '361', '65', '3497', '578', '72', '634', '3446', '627', '43', '62', '34', '620', '76', '23', '68', '631', '54', '3500', '3480', '37', '3440', '619', '44', '3466', '30', '3487', '45', '61', '3452', '3491', '623', '633', '53', '66', '67', '69', '643', '58', '632', '636', '7', '641', '51', '3489', '3471', '21', '36', '3468', '4', '576', '46', '63', '3457', '56', '3448', '3441', '618', '52', '3429', '3438', '610', '26', '609', '3444', '612', '3485', '3465', '41', '20', '3464', '3477', '3459', '621', '3432', '60', '3449', '626', '628', '614', '47', '3454', '38', '3428', '33', '12', '3426', '3442', '3472', '13', '639', '3470', '611', '6', '40', '14', '3486', '31', '3458', '3437', '3453', '55', '3424', '3481', '635', '64', '629', '3460', '50', '9', '18', '42', '3495', '5', '580']       

 
        #the converter changed to other unlabelled TextRegions or 'marginalia' TRs
        lIgnoredLabels1 = None
        
        """
        if you play with a toy collection, which does not have all expected classes, you can reduce those.
        """
        
#         lActuallySeen = None
#         if lActuallySeen:
#             print( "REDUCING THE CLASSES TO THOSE SEEN IN TRAINING")
#             lIgnoredLabels  = [lLabels[i] for i in range(len(lLabels)) if i not in lActuallySeen]
#             lLabels         = [lLabels[i] for i in lActuallySeen ]
#             print( len(lLabels)          , lLabels)
#             print( len(lIgnoredLabels)   , lIgnoredLabels)
        if cls.bHTR:
            ntClass = NodeType_PageXml_type
        else:
            #ignore text
            ntClass = NodeType_PageXml_type_woText
                         
        nt1 = ntClass("bar"                   #some short prefix because labels below are prefixed with it
                              , lLabels1
                              , lIgnoredLabels1
                              , False    #no label means OTHER
                              , BBoxDeltaFun=lambda v: max(v * 0.066, min(5, v/3))  #we reduce overlap in this way
                              )
        nt1.setLabelAttribute("DU_num")
        if cls.bTextLine: 
            nt1.setXpathExpr( (".//pc:TextRegion/pc:TextLine[@DU_num]"        #how to find the nodes
                          , "./pc:TextEquiv") 
                       )
        else:
            nt1.setXpathExpr( (".//pc:TextRegion"        #how to find the nodes
                          , "./pc:TextEquiv")       #how to get their text
                       )
        DU_GRAPH.addNodeType(nt1)
            
        return DU_GRAPH

    
    # ===============================================================================================================
    

    
    # """
    # if you play with a toy collection, which does not have all expected classes, you can reduce those.
    # """
    # 
    # lActuallySeen = None
    # if lActuallySeen:
    #     print "REDUCING THE CLASSES TO THOSE SEEN IN TRAINING"
    #     lIgnoredLabels  = [lLabels[i] for i in range(len(lLabels)) if i not in lActuallySeen]
    #     lLabels         = [lLabels[i] for i in lActuallySeen ]
    #     print len(lLabels)          , lLabels
    #     print len(lIgnoredLabels)   , lIgnoredLabels
    #     nbClass = len(lLabels) + 1  #because the ignored labels will become OTHER
    

    
    #=== CONFIGURATION ====================================================================
    def __init__(self, sModelName, sModelDir, sComment=None, C=None, tol=None, njobs=None, max_iter=None, inference_cache=None): 
        
        if self.bHTR:
            cFeatureDefinition = FeatureDefinition_PageXml_StandardOnes
            dFeatureConfig = {  
                               'n_tfidf_node':100, 't_ngrams_node':(1,2), 'b_tfidf_node_lc':False
                              , 'n_tfidf_edge':100, 't_ngrams_edge':(1,2), 'b_tfidf_edge_lc':False }
        else:
            cFeatureDefinition = FeatureDefinition_PageXml_StandardOnes_noText
            dFeatureConfig = { } 
                               #'n_tfidf_node':None, 't_ngrams_node':None, 'b_tfidf_node_lc':None
                              #, 'n_tfidf_edge':None, 't_ngrams_edge':None, 'b_tfidf_edge_lc':None }
        
        DU_CRF_Task.__init__(self
                     , sModelName, sModelDir
                     , dFeatureConfig = dFeatureConfig
                     , dLearnerConfig = {
                                   'C'                : .1   if C               is None else C
                                 , 'njobs'            : 16    if njobs           is None else njobs
                                 , 'inference_cache'  : 50   if inference_cache is None else inference_cache
                                 #, 'tol'              : .1
                                 , 'tol'              : .05  if tol             is None else tol
                                 , 'save_every'       : 50     #save every 50 iterations,for warm start
                                 , 'max_iter'         : 1000 if max_iter        is None else max_iter
                         }
                     , sComment=sComment
                     , cFeatureDefinition=cFeatureDefinition
#                     , cFeatureDefinition=FeatureDefinition_T_PageXml_StandardOnes_noText
#                      , dFeatureConfig = {
#                          #config for the extractor of nodes of each type
#                          "text": None,    
#                          "sprtr": None,
#                          #config for the extractor of edges of each type
#                          "text_text": None,    
#                          "text_sprtr": None,    
#                          "sprtr_text": None,    
#                          "sprtr_sprtr": None    
#                          }
                     )
        
        traceln("- classes: ", self.getGraphClass().getLabelNameList())

        self.bsln_mdl = self.addBaseline_LogisticRegression()    #use a LR model trained by GridSearch as baseline
        
    #=== END OF CONFIGURATION =============================================================

  
    def predict(self, lsColDir):#,sDocId):
        """
        Return the list of produced files
        """
#         self.sXmlFilenamePattern = "*.a_mpxml"
        return DU_CRF_Task.predict(self, lsColDir)#,sDocId)


    def runForExternalMLMethod(self, lsColDir, storeX, applyY, bRevertEdges=False):
        """
        Return the list of produced files
        """
        self.sXmlFilenamePattern = "*.mpxml"
        return DU_CRF_Task.runForExternalMLMethod(self, lsColDir, storeX, applyY, bRevertEdges)




from tasks.DU_ECN_Task import DU_ECN_Task
import gcn.DU_Model_ECN   
class DU_ABPTable_ECN(DU_ECN_Task):
        """
        ECN Models
        """
        bHTR     = False  # do we have text from an HTR?
        bPerPage = False # do we work per document or per page?
        bTextLine = True # if False then act as TextRegion

        sMetadata_Creator = "NLE Document Understanding ECN"
        sXmlFilenamePattern = "*.mpxml"

        # sLabeledXmlFilenamePattern = "*.a_mpxml"
        sLabeledXmlFilenamePattern = "*.mpxml"


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
            
            lLabels = ['IGNORE', '577', '579', '581', '608', '32', '3431', '617', '3462', '3484', '615', '49', '3425', '73', '3', '3450', '2', '11', '70', '3451', '637', '77', '3447', '3476', '3467', '3494', '3493', '3461', '3434', '48', '3456', '35', '3482', '74', '3488', '3430', '17', '613', '625', '3427', '3498', '29', '3483', '3490', '362', '638a', '57', '616', '3492', '10', '630', '24', '3455', '3435', '8', '15', '3499', '27', '3478', '638b', '22', '3469', '3433', '3496', '624', '59', '622', '75', '640', '1', '19', '642', '16', '25', '3445', '3463', '3443', '3439', '3436', '3479', '71', '3473', '28', '39', '361', '65', '3497', '578', '72', '634', '3446', '627', '43', '62', '34', '620', '76', '23', '68', '631', '54', '3500', '3480', '37', '3440', '619', '44', '3466', '30', '3487', '45', '61', '3452', '3491', '623', '633', '53', '66', '67', '69', '643', '58', '632', '636', '7', '641', '51', '3489', '3471', '21', '36', '3468', '4', '576', '46', '63', '3457', '56', '3448', '3441', '618', '52', '3429', '3438', '610', '26', '609', '3444', '612', '3485', '3465', '41', '20', '3464', '3477', '3459', '621', '3432', '60', '3449', '626', '628', '614', '47', '3454', '38', '3428', '33', '12', '3426', '3442', '3472', '13', '639', '3470', '611', '6', '40', '14', '3486', '31', '3458', '3437', '3453', '55', '3424', '3481', '635', '64', '629', '3460', '50', '9', '18', '42', '3495', '5', '580']


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
                print("REDUCING THE CLASSES TO THOSE SEEN IN TRAINING")
                lIgnoredLabels = [lLabels[i] for i in range(len(lLabels)) if i not in lActuallySeen]
                lLabels = [lLabels[i] for i in lActuallySeen]
                print(len(lLabels), lLabels)
                print(len(lIgnoredLabels), lIgnoredLabels)

            if cls.bHTR:
                ntClass = NodeType_PageXml_type
            else:
                #ignore text
                ntClass = NodeType_PageXml_type_woText



            # DEFINING THE CLASS OF GRAPH WE USE
            nt = ntClass("bar"  # some short prefix because labels below are prefixed with it
                                              , lLabels
                                              , lIgnoredLabels
                                              , False  # no label means OTHER
                                              , BBoxDeltaFun=lambda v: max(v * 0.066, min(5, v / 3))
                                              # we reduce overlap in this way
                                              )
                                                 


            nt.setLabelAttribute("DU_num")
            if cls.bTextLine:
                nt.setXpathExpr( (".//pc:TextRegion/pc:TextLine[@DU_num]"        #how to find the nodes
                          , "./pc:TextEquiv")
                       )
            else:
                nt.setXpathExpr( (".//pc:TextRegion"        #how to find the nodes
                          , "./pc:TextEquiv")       #how to get their text
                       )


            DU_GRAPH.addNodeType(nt)

            return DU_GRAPH

        def __init__(self, sModelName, sModelDir, sComment=None,dLearnerConfigArg=None):
            print ( self.bHTR)
            			
            if self.bHTR:
                cFeatureDefinition = FeatureDefinition_PageXml_StandardOnes
                dFeatureConfig = { 'bMultiPage':False, 'bMirrorPage':False
                              , 'n_tfidf_node':300, 't_ngrams_node':(1,4), 'b_tfidf_node_lc':False
                              , 'n_tfidf_edge':300, 't_ngrams_edge':(1,4), 'b_tfidf_edge_lc':False }
            else:
                cFeatureDefinition = FeatureDefinition_PageXml_StandardOnes_noText
                dFeatureConfig = {  }


            if sComment is None: sComment  = sModelName


            if  dLearnerConfigArg is not None and "ecn_ensemble" in dLearnerConfigArg:
                print('ECN_ENSEMBLE')
                DU_ECN_Task.__init__(self
                                     , sModelName, sModelDir
                                     , dFeatureConfig=dFeatureConfig
                                     ,
                                     dLearnerConfig=dLearnerConfigArg if dLearnerConfigArg is not None else self.dLearnerConfig
                                     , sComment=sComment
                                     , cFeatureDefinition= cFeatureDefinition
                                     , cModelClass=gcn.DU_Model_ECN.DU_Ensemble_ECN
                                     )


            else:
                #Default Case Single Model
                DU_ECN_Task.__init__(self
                                     , sModelName, sModelDir
                                     , dFeatureConfig=dFeatureConfig
                                     , dLearnerConfig= dLearnerConfigArg if dLearnerConfigArg is not None else self.dLearnerConfig
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




class DU_ABPTable_GAT(DU_ECN_Task):
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
                print("REDUCING THE CLASSES TO THOSE SEEN IN TRAINING")
                lIgnoredLabels = [lLabels[i] for i in range(len(lLabels)) if i not in lActuallySeen]
                lLabels = [lLabels[i] for i in lActuallySeen]
                print(len(lLabels), lLabels)
                print(len(lIgnoredLabels), lIgnoredLabels)


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




# ----------------------------------------------------------------------------

def main(sModelDir, sModelName, options):
    if options.use_ecn:
        if options.ecn_json_config is not None and options.ecn_json_config is not []:
            f = open(options.ecn_json_config[0])
            djson=json.loads(f.read())

            if "ecn_learner_config" in djson:
                dLearnerConfig=djson["ecn_learner_config"]
                f.close()
                doer = DU_ABPTable_ECN(sModelName, sModelDir,dLearnerConfigArg=dLearnerConfig)
            elif "ecn_ensemble" in djson:
                dLearnerConfig = djson
                f.close()
                doer = DU_ABPTable_ECN(sModelName, sModelDir, dLearnerConfigArg=dLearnerConfig)

        else:
            doer = DU_ABPTable_ECN(sModelName, sModelDir)
    elif options.use_gat:
        if options.gat_json_config is not None and options.gat_json_config is not []:

            f = open(options.gat_json_config[0])
            djson=json.loads(f.read())
            dLearnerConfig=djson["gat_learner_config"]
            f.close()
            doer = DU_ABPTable_GAT(sModelName, sModelDir,dLearnerConfigArg=dLearnerConfig)

        else:
            doer = DU_ABPTable_GAT(sModelName, sModelDir)

    else:
        doer = m(DU_BAR_sem)


    if options.rm:
        doer.rm()
        return




    lTrn, lTst, lRun, lFold = [_checkFindColDir(lsDir) for lsDir in [options.lTrn, options.lTst, options.lRun, options.lFold]]

    traceln("- classes: ", doer.getGraphClass().getLabelNameList())

    ## use. a_mpxml files
    doer.sXmlFilenamePattern = doer.sLabeledXmlFilenamePattern


    if options.iFoldInitNum or options.iFoldRunNum or options.bFoldFinish:
        if options.iFoldInitNum:
            """
            initialization of a cross-validation
            """
            splitter, ts_trn, lFilename_trn = doer._nfold_Init(lFold, options.iFoldInitNum, test_size=0.25, random_state=None, bStoreOnDisk=True)
        elif options.iFoldRunNum:
            """
            Run one fold
            """
            oReport = doer._nfold_RunFoldFromDisk(options.iFoldRunNum, options.warm, options.pkl)
            traceln(oReport)
        elif options.bFoldFinish:
            tstReport = doer._nfold_Finish()
            traceln(tstReport)
        else:
            assert False, "Internal error"
        #no more processing!!
        exit(0)
        #-------------------





    if lFold:
        loTstRpt = doer.nfold_Eval(lFold, 3, .25, None, options.pkl)
        import graph.GraphModel
        sReportPickleFilename = os.path.join(sModelDir, sModelName + "__report.txt")
        traceln("Results are in %s"%sReportPickleFilename)
        graph.GraphModel.GraphModel.gzip_cPickle_dump(sReportPickleFilename, loTstRpt)
    elif lTrn:
        doer.train_save_test(lTrn, lTst, options.warm, options.pkl)
        try:    traceln("Baseline best estimator: %s"%doer.bsln_mdl.best_params_)   #for GridSearch
        except: pass
        traceln(" --- CRF Model ---")
        traceln(doer.getModel().getModelInfo())
    elif lTst:
        doer.load()
        tstReport = doer.test(lTst)
        traceln(tstReport)
        if options.bDetailedReport:
            traceln(tstReport.getDetailledReport())
            import graph.GraphModel
            for test in lTst:
                sReportPickleFilename = os.path.join('..',test, sModelName + "__report.pkl")
                traceln('Report dumped into %s'%sReportPickleFilename)
                graph.GraphModel.GraphModel.gzip_cPickle_dump(sReportPickleFilename, tstReport)

    if lRun:
        if options.storeX or options.applyY:
            try: doer.load()
            except: pass    #we only need the transformer
            lsOutputFilename = doer.runForExternalMLMethod(lRun, options.storeX, options.applyY, options.bRevertEdges)
        else:
            doer.load()
            lsOutputFilename = doer.predict(lRun)

        traceln("Done, see in:\n  %s"%lsOutputFilename)


# ----------------------------------------------------------------------------





if __name__ == "__main__":

    version = "v.01"
    usage, description, parser = DU_CRF_Task.getBasicTrnTstRunOptionParser(sys.argv[0], version)
#     parser.add_option("--annotate", dest='bAnnotate',  action="store_true",default=False,  help="Annotate the textlines with BIES labels")    

    #FOR GCN
    parser.add_option("--revertEdges", dest='bRevertEdges',  action="store_true", help="Revert the direction of the edges")
    parser.add_option("--detail", dest='bDetailedReport',  action="store_true", default=False,help="Display detailled reporting (score per document)")
    parser.add_option("--baseline", dest='bBaseline',  action="store_true", default=False, help="report baseline method")
    parser.add_option("--ecn",dest='use_ecn',action="store_true", default=False, help="wether to use ECN Models")
    parser.add_option("--ecn_config", dest='ecn_json_config',action="append", type="string", help="The Config files for the ECN Model")
    parser.add_option("--gat", dest='use_gat', action="store_true", default=False, help="wether to use ECN Models")
    parser.add_option("--gat_config", dest='gat_json_config', action="append", type="string",
                      help="The Config files for the Gat Model")
    # --- 
    #parse the command line
    (options, args) = parser.parse_args()

    # --- 
    try:
        sModelDir, sModelName = args
    except Exception as e:
        traceln("Specify a model folder and a model name!")
        _exit(usage, 1, e)

    main(sModelDir, sModelName, options)

