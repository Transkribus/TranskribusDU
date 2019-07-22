# -*- coding: utf-8 -*-

"""
    DU task based on ECN
    
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
                          , help="The Config files for the ECN Model")
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
                elif "ecn_ensemble" in djson:
                    dLearnerConfig = djson
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
            "node_indim"            : -1,
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