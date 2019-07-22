# -*- coding: utf-8 -*-

"""
    DU task based on ECN GAT
    
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
import json

try:  # to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))
    import TranskribusDU_version
TranskribusDU_version

from common.trace import traceln

import graph.GraphModel
from tasks.DU_Task import DU_Task

# dynamically imported
DU_Model_GAT = None

class DU_GAT_Task(DU_Task):
    """
    DU learner based on GAT
    """
    VERSION = "GAT_v19"
    
    version          = None     # dynamically computed
    
    def __init__(self, sModelName, sModelDir
                 , sComment             = None
                 , cFeatureDefinition   = None
                 , dFeatureConfig       = {}
                 , cModelClass          = None
                 ):
        super().__init__(sModelName, sModelDir, sComment, cFeatureDefinition, dFeatureConfig)

        global DU_Model_GAT
        DU_Model_GAT = DU_Task.DYNAMIC_IMPORT('.DU_Model_ECN', 'gcn').DU_Model_GAT
        
        self.cModelClass = DU_Model_GAT if cModelClass == None else cModelClass
        assert issubclass(self.cModelClass, graph.GraphModel.GraphModel), "Your model class must inherit from graph.GraphModel.GraphModel"

    @classmethod
    def getVersion(cls):
        cls.version = "-".join([DU_Task.getVersion(), str(cls.VERSION)])
        return cls.version 

    @classmethod
    def updateStandardOptionsParser(cls, parser):
        usage = """
        --gat         Enable Graph Attention learning
        --gat_config  Path to the JSON configuration file (required!) for GAT learning
        """
        #FOR GAT
        parser.add_option("--gat"        , dest='bGAT'           , action="store_true"
                          , default=False, help="wether to use GAT Models")
        parser.add_option("--gat_config" , dest='gat_json_config'   , action="store", type="string",
                          help="The Config files for the Gat Model")
        # parser.add_option("--baseline"   , dest='bBaseline'         , action="store_true"
        #                  , default=False, help="report baseline method")

        return usage, parser


    def getStandardLearnerConfig(self, options):
        """
        Once the command line has been parsed, you can get the standard learner
        configuration dictionary from here.
        """
        try:
            sFile = options.gat_json_config or options.ecn_json_config
        except AttributeError:
            sFile = options.gat_json_config
        if sFile:
            with open(sFile) as f:
                djson = json.loads(f.read())
                if "gat_learner_config" in djson:
                    dLearnerConfig = djson["gat_learner_config"]
                elif "gat_ensemble" in djson:
                    dLearnerConfig = djson
                elif "ecn_learner_config" in djson:
                    dLearnerConfig = djson["ecn_learner_config"]
                elif "ecn_ensemble" in djson:
                    dLearnerConfig = djson
                else:
                    raise Exception("Invalid config JSON file")
        
        else:
            dLearnerConfig = {  'name'                  : "default_5lay5att"
                              , 'nb_iter'               : 500,
                              'lr'                      : 0.001,
                              'num_layers'              : 5,
                              'nb_attention'            : 5,
                              'stack_convolutions'      : True,
                              'node_indim'              : -1,
                              'dropout_rate_node'       : 0.0,
                              'dropout_rate_attention'  : 0.0,
                              'ratio_train_val'         : 0.15,
                              "activation_name"         : 'tanh',
                              "patience"                : 50,
                              "original_model"          : False,
                              "attn_type"               : 0
               }
            
        if options.max_iter:
            traceln(" - max_iter=%d" % options.max_iter)
            dLearnerConfig["nb_iter"] = options.max_iter
        
        return dLearnerConfig