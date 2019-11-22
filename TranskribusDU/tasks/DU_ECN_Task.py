# -*- coding: utf-8 -*-

"""
    DU task based on ECN
    
    Copyright NAVER(C) 2018, 2019  Hervé Déjean, Jean-Luc Meunier, Animesh Prasad


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""
import sys, os
import json

try:  # to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))
    import TranskribusDU_version
TranskribusDU_version

from common.trace import traceln
from tasks.DU_Task import DU_Task
import graph.GraphModel

# dynamically imported
DU_Model_ECN = None


class DU_ECN_Task(DU_Task):
    """
    DU learner based on ECN
    """
    VERSION = "ECN_v19"
    
    version          = None     # dynamically computed
    
    def __init__(self, sModelName, sModelDir
                 , sComment             = None
                 , cFeatureDefinition   = None
                 , dFeatureConfig       = {}
                 , cModelClass          = None
                 ):
        super().__init__(sModelName, sModelDir, sComment, cFeatureDefinition, dFeatureConfig)

        global DU_Model_ECN
        DU_Model_ECN = DU_Task.DYNAMIC_IMPORT('.DU_Model_ECN', 'gcn').DU_Model_ECN

        self.cModelClass = DU_Model_ECN if cModelClass == None else cModelClass
        assert issubclass(self.cModelClass, graph.GraphModel.GraphModel), "Your model class must inherit from graph.GraphModel.GraphModel"

    @classmethod
    def getVersion(cls):
        cls.version = "-".join([DU_Task.getVersion(), str(cls.VERSION)])
        return cls.version 

    @classmethod
    def updateStandardOptionsParser(cls, parser):
        usage = """
        --ecn         Enable Edge Convolutional Network learning
        --ecn_config  Path to the JSON configuration file (required!) for ECN learning
        """
        #FOR GCN
        parser.add_option("--ecn"        , dest='bECN'              , action="store_true"
                          , default=False, help="use ECN Models")
        parser.add_option("--ecn_config" , dest='ecn_json_config'   , action="store", type="string"
                          , help="The config file for the ECN Model")
        parser.add_option("--baseline"   , dest='bBaseline'         , action="store_true"
                          , default=False, help="report baseline method")

        return usage, parser

    def getStandardLearnerConfig(self, options):
        """
        Once the command line has been parsed, you can get the standard learner
        configuration dictionary from here.
        """
        if options.ecn_json_config:
            with open(options.ecn_json_config) as f:
                djson = json.loads(f.read())
                if "ecn_learner_config" in djson:
                    dLearnerConfig=djson["ecn_learner_config"]
                else:
                    raise Exception("Invalid config JSON file")
        else:
            dLearnerConfig = {
            "name"                  :"default_8Lay1Conv",
            "dropout_rate_edge"     : 0.2,
            "dropout_rate_edge_feat": 0.2,
            "dropout_rate_node"     : 0.2,
            "lr"                    : 0.0001,
            "mu"                    : 0.0001,
            "nb_iter"               : 1200,
            "nconv_edge"            : 1,
            # until AUg 29, 2019 "node_indim"            : -1,
            "node_indim"            : 64,
            "num_layers"            : 8,
            "ratio_train_val"       : 0.1,
            "patience"              : 50,
            "activation_name"       :"relu",
            "stack_convolutions"    : False
            }
            
        if options.max_iter:
            traceln(" - max_iter=%d" % options.max_iter)
            dLearnerConfig["nb_iter"] = options.max_iter
        
        return dLearnerConfig


class DU_Ensemble_ECN_Task(DU_Task):
    """
    DU learner based on Ensemble ECN
    """
    VERSION = "ECN_v19"
    
    version          = None     # dynamically computed
    
    def __init__(self, sModelName, sModelDir
                 , sComment             = None
                 , cFeatureDefinition   = None
                 , dFeatureConfig       = {}
                 , cModelClass          = None
                 ):
        super().__init__(sModelName, sModelDir, sComment, cFeatureDefinition, dFeatureConfig)

        global DU_Model_ECN
        DU_Model_ECN = DU_Task.DYNAMIC_IMPORT('.DU_Model_ECN', 'gcn').DU_Ensemble_ECN

        self.cModelClass = DU_Model_ECN if cModelClass == None else cModelClass
        assert issubclass(self.cModelClass, graph.GraphModel.GraphModel), "Your model class must inherit from graph.GraphModel.GraphModel"

    @classmethod
    def getVersion(cls):
        cls.version = "-".join([DU_Task.getVersion(), str(cls.VERSION)])
        return cls.version 

    @classmethod
    def updateStandardOptionsParser(cls, parser):
        usage = """
        --ecn_ensemble         Enable Edge Convolutional Network learning
        --ecn_ensemble_config  Path to the JSON configuration file (required!) for ECN learning
        """
        #FOR GCN
        parser.add_option("--ecn_ensemble" , dest='bECNEnsemble'    
                          , action="store_true"
                          , default=False, help="use Ensemble ECN Models")
        parser.add_option("--ecn_ensemble_config" , dest='ecn_ensemble_json_config'   
                          , action="store", type="string"
                          , help="The config file for the Ensemble ECN Model")
        return usage, parser

    def getStandardLearnerConfig(self, options):
        """
        Once the command line has been parsed, you can get the standard learner
        configuration dictionary from here.
        """
        if options.ecn_ensemble_json_config:
            with open(options.ecn_ensemble_json_config) as f:
                djson = json.loads(f.read())
                if "ecn_ensemble" in djson:
                    dLearnerConfig = djson
                else:
                    raise Exception("Invalid config JSON file for ensemble ECN model.")
        else:
            dLearnerConfig = {
        "_comment":"1 relu and 1 tanh models, twice",
        "ratio_train_val": 0.2,
        "ecn_ensemble": [
            {
            "type": "ecn",
            "name"                  :"default_8Lay1Conv_A",
            "dropout_rate_edge"     : 0.2,
            "dropout_rate_edge_feat": 0.2,
            "dropout_rate_node"     : 0.2,
            "lr"                    : 0.0001,
            "mu"                    : 0.0001,
            "nb_iter"               : 1200,
            "nconv_edge"            : 1,
            "node_indim"            : 64,
            "num_layers"            : 8,
            "ratio_train_val"       : 0.1,
            "patience"              : 50,
            "activation_name"       :"relu",
            "stack_convolutions"    : False
            },
            {
            "type": "ecn",
            "name"                  :"default_8Lay1Conv_A",
            "dropout_rate_edge"     : 0.2,
            "dropout_rate_edge_feat": 0.2,
            "dropout_rate_node"     : 0.2,
            "lr"                    : 0.0001,
            "mu"                    : 0.0001,
            "nb_iter"               : 1200,
            "nconv_edge"            : 1,
            "node_indim"            : 64,
            "num_layers"            : 8,
            "ratio_train_val"       : 0.1,
            "patience"              : 50,
            "activation_name"       :"tanh",
            "stack_convolutions"    : False
            },
           {
            "type": "ecn",
            "name"                  :"default_8Lay1Conv_B",
            "dropout_rate_edge"     : 0.2,
            "dropout_rate_edge_feat": 0.2,
            "dropout_rate_node"     : 0.2,
            "lr"                    : 0.0001,
            "mu"                    : 0.0001,
            "nb_iter"               : 1200,
            "nconv_edge"            : 1,
            "node_indim"            : 64,
            "num_layers"            : 8,
            "ratio_train_val"       : 0.1,
            "patience"              : 50,
            "activation_name"       :"relu",
            "stack_convolutions"    : False
            },
            {
            "type": "ecn",
            "name"                  :"default_8Lay1Conv_B",
            "dropout_rate_edge"     : 0.2,
            "dropout_rate_edge_feat": 0.2,
            "dropout_rate_node"     : 0.2,
            "lr"                    : 0.0001,
            "mu"                    : 0.0001,
            "nb_iter"               : 1200,
            "nconv_edge"            : 1,
            "node_indim"            : 64,
            "num_layers"            : 8,
            "ratio_train_val"       : 0.1,
            "patience"              : 50,
            "activation_name"       :"tanh",
            "stack_convolutions"    : False
            }
        ] 
                }         
            
        if options.max_iter:
            traceln(" - max_iter=%d" % options.max_iter)
            dLearnerConfig["nb_iter"] = options.max_iter
        
        return dLearnerConfig
