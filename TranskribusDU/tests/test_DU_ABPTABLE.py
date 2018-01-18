# -*- coding: utf-8 -*-

'''
Testing various DU function provided by DU_ABPTABLE

Created on 10 janv. 2018

@author: meunier
'''

from __future__ import absolute_import, print_function

import sys
import os.path

# deal with file locations and PYTHONPATH
sTESTS_DIR = os.path.dirname(os.path.abspath(__file__))
sDATA_DIR = os.path.join(sTESTS_DIR, "data")
sys.path.append(os.path.dirname(sTESTS_DIR))

import tasks.DU_ABPTable

#Fake output of a command line parse    
class FakeOption:
    def __init__(self):
        self.lTrn = None
        self.lTst = None
        self.lRun = None
        self.lFold = None
        self.iFoldInitNum = None
        self.iFoldRunNum = None
        self.bFoldFinish = False
        self.warm = False
        self.pkl = False
        self.rm = False
        self.crf_njobs = 2
        self.crf_max_iter = 2
        self.crf_C = None
        self.crf_tol = None
        self.crf_inference_cache = None
        self.best_params = None
        
        self.storeX = None
        self.applyY = None
        
def test_ABPTable_train():
    
    sModelDir = os.path.join(sTESTS_DIR, "models")
    sModelName = "test_ABPTable_train"
    sDataDir = os.path.join(sDATA_DIR, "abp_TABLE_9142_mpxml")
    
    options = FakeOption()
    options.rm = True                       
    # remove any pre-existing model
    tasks.DU_ABPTable.main(sModelDir, sModelName, options)
     
    options = FakeOption()
    options.lTrn = [sDataDir]                       
    tasks.DU_ABPTable.main(sModelDir, sModelName, options)
     
    options = FakeOption()
    options.lTst = [sDataDir]                       
    tasks.DU_ABPTable.main(sModelDir, sModelName, options)

#     options = FakeOption()
#     options.lRun = [sDataDir]                       
#     tasks.DU_ABPTable.main(sModelDir, sModelName, options)
    
if __name__ == "__main__":
    test_ABPTable_train()
    
    