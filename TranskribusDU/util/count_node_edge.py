# -*- coding: utf-8 -*-

"""
    Count the number of nodes and edges in TRN and VLD from the log file
   
    November 2020
    Copyright Naver Labs Europe 2018
    JL Meunier
"""


import sys, os
import re


# ================================================================
# what we want to match

# ========== the file names
"""
    - computing features on training set
         #nodes=366  #edges=831 
         ECN #nodes=429  #edges=2550 
         ECN #nodes=402  #edges=2306 
         #features nodes=54  edges=539 
     [0.7s] done

    - 2 classes (['continue', 'break'])
    - retrieving or creating model...
         ECN #nodes=920  #edges=5758 
         ECN #nodes=731  #edges=4620 
2 training graphs --  2 validation graphs
"""
# find filenames
cre_FILES_begin     = re.compile("^\s+- computing features on training set\s*$")
cre_FILES_nnod_nedg = re.compile("^\s*ECN #nodes=(\d+)  #edges=(\d+)\s*$")
cre_FILES_end       = re.compile("^\d+ training graphs --  \d+ validation graphs\s*$")
          
def parse_log(fn):
    """
    parse the log
    return nb_graph, nb_node, nb_edge
    """
    nG, nN, nE = 0, 0, 0

    bIn = 0
    
    with open(fn, "r") as fd:
        # state 0
        for s in fd:
            if bIn:
                if cre_FILES_end.match(s):
                    bIn = 2
                    break
                o = cre_FILES_nnod_nedg.match(s)
                if o is not None:
                    nG += 1
                    nN += int(o.group(1))
                    nE += int(o.group(2))
                    #print(int(o.group(1)), int(o.group(2)))
            elif cre_FILES_begin.match(s):
                bIn = 1
                continue
    if bIn != 2: 
        raise ValueError("ERROR: end of data series not observed???")
    return nG, nN, nE
         

# ----------------------------------------------------------------------------    
if __name__ == "__main__":
    if not(sys.argv[1:]):
        print("Usage: %s <LOGFILE>"%sys.argv[0])
        print("print number of node and edge in trn+vld set.")
        sys.exit(1)
        
       
    fn = sys.argv[1]
    nG, nN, nE = parse_log(fn)
    #print("%d graphs =   %d nodes   %d edges" % (nG, nN, nE))
    print("%d graphs =   %s nodes   %s edges" % (nG
                                                 , '{:,d}'.format(nN)
                                                 , '{:,d}'.format(nE)
                                                 ))
