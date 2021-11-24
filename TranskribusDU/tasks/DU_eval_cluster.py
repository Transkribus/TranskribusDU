# -*- coding: utf-8 -*-

"""
    Tentative to get a simple tool to reproduce the evaluation done in DU_Task.
    
    JL Meunier
    Octobre 2020
    Copyright Naver 2020
"""

import sys, os
from optparse import OptionParser

from tasks.DU_Table.DU_Table_Evaluator import listFiles, eval_cluster_of_files

if __name__ == "__main__":
    usage = """
    %s RUN_DIR+   --gtattribute <NAME>   eval the quality of the clusters
    """ % sys.argv[0]

    parser = OptionParser(usage=usage, version="0.1")
    parser.add_option("-v", "--verbose", dest='bVerbose', action="store_true"
                      , default=False) 
    parser.add_option("--gtattribute", dest='sGTAttr', action="store"
                      , type="string"
                      , help="attribute name defining the GT cluster for each node of interest"
                      , default=None) 
    parser.add_option("--xpselector", dest='xpSelector', action="store"
                      , type="string"
                      , help="xpath expression to select the nodes of interest ('.//pc:TextLine' by default)"
                      , default=".//pc:TextLine") 

    (options, args) = parser.parse_args()

    assert options.sGTAttr is not None, "You must specify a GT attribute"
    
    lsDir = args
    
    for sDir in lsDir:
        lsFilename = listFiles(sDir)
        print("----- ", sDir, "  %d files"% len(lsFilename))
        nOk, nErr, nMiss, sRpt, _ = eval_cluster_of_files([os.path.join(sDir, _s) for _s in lsFilename]
                                                      , "cluster"
                                                      , bIgnoreHeader=False
                                                      , bIgnoreOutOfTable=True
                                                      , xpSelector=options.xpSelector
                                                      , sClusterGTAttr=options.sGTAttr
                                                      , bVerbose=options.bVerbose
                                             )
        # fP, fR, fF = computePRF(nOk, nErr, nMiss)
        print(sRpt)
                        
