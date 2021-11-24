# -*- coding: utf-8 -*-

"""
    Determine if one tagging is significantly better than another, per tag

    Assume (and check) that the 2 clustering are based on the same files
    
    November 2020
    Copyright Naver Labs Europe 2018
    JL Meunier
"""


import sys, os
import re
from collections import defaultdict
import math

from statistics import mean, stdev

import scipy.stats

DEBUG = 0

alpha = 0.05

# ================================================================
# what we want to match

cre_blank_line = re.compile("^\s*$")
sre_brckt      = "\[[^\]]*\]"   # match things between square brackets

# ========== the file names
"""
- loading test graphs
    1 - /tmp-network/user/meunier/menus/all_20201027/tst/col/Maylis-images-ByAlphabet-English-0001.jpg.pxml
        EDGES:  167 horizontal    257 vertical      0 celtic
    Type 0 - 0    186 nodes            424 edges
         (186 nodes,  424 edges)
    2 - /tmp-network/user/meunier/menus/all_20201027/tst/col/Maylis-images-ByAlphabet-English-19647854735_5086e45f84_b.jpg.pxml
        EDGES:  141 horizontal    243 vertical      0 celtic
.....
 --- 197 graphs loaded
"""
# find filenames
cre_FILES_begin     = re.compile("^- loading test graphs\s*$")
cre_FILES_num_fn    = re.compile("^\s*([\d]+)\s+-\s+(.*)")
cre_FILES_end       = cre_blank_line

# ========== the tags
"""
  Line=True class, column=Prediction
           mi_OTHER  [[ 4289   306     2  1532    21    27]
 mi_section-heading   [  232  1397     7  1433    17    25]
   mi_section-price   [    4    12    88    21    61     5]
mi_item-description   [  611   277     1 33010    79   146]
      mi_item-price   [   43    19    34   474  3767    99]
     mi_item-number   [   35    12     2   413   126   671]]

"""
# find confusion matrix
cre_CONFMAT_begin  = re.compile("^\s+Line=True class, column=Prediction\s*")
cre_CONFMAT_tag    = re.compile("^\s*([^\s]+)\s+.*")
cre_CONFMAT_end    = cre_blank_line

# ========== the detailed report
"""
 Detailed Reporting Precision per label, then Recall, then F1,  per document
--------------------
0    [0.733 0.    0.    0.958 0.971 0.857] [0.687 0.    0.    0.983 0.825 0.857] [0.71  0.    0.    0.97  0.892 0.857]
1    [1. 1. 0. 1. 1. 0.] [1. 1. 0. 1. 1. 0.] [1. 1. 0. 1. 1. 0.]
.....
196    [0.808 0.125 0.    0.7   0.875 0.   ] [0.913 0.042 0.    0.792 1.    0.   ] [0.857 0.062 0.    0.743 0.933 0.   ]

"""
cre_PRF_begin   = re.compile("^\s+Detailed Reporting Precision per label, then Recall, then F1,  per document.*")
cre_PRF_num_lf1 = re.compile("^\s*([\d]+)\s*" + sre_brckt + "\s+" + sre_brckt + "\s" + "\[([^\]]*)\]")      
cre_PRF_end     = cre_blank_line

          
def parse_log(fn):
    """
    parse the log
    return a  list of tuple( <filename>
                            , dictionary of (P,R,F1, ok, err, miss) per threshold
                            )
    """
    lFN  = list()
    lTag = list()
    dL   = defaultdict(list)  

    lState = [  "Looking for file names"
              , "looking for tags"
              , "looking for f1 per tag per file"]
    state = 0
    bIn = False
    
    if DEBUG: print(lState[state])
                    
    with open(fn, "r") as fd:
        # state 0
        for s in fd:
            if bIn:
                if cre_FILES_end.match(s):
                    break
                o = cre_FILES_num_fn.match(s)
                if o is not None:
                    num = int(o.group(1))
                    fn  =     o.group(2).strip()
                    if DEBUG: print(num, fn)
                    lFN.append(fn)
            elif cre_FILES_begin.match(s):
                bIn = True
                continue

        state += 1
        bIn = False
        for s in fd:
            if bIn:
                if cre_CONFMAT_end.match(s):
                    break
                o = cre_CONFMAT_tag.match(s)
                tag = o.group(1).strip()
                lTag.append(tag)
                if DEBUG: print(tag)
            elif cre_CONFMAT_begin.match(s):
                bIn = True
                continue
     
        nTag = len(lTag)
        state += 1
        bIn = False
        for s in fd:
            if bIn:
                if cre_PRF_end.match(s):
                    break
                o = cre_PRF_num_lf1.match(s)
                num  = int(o.group(1))
                slf1 =     o.group(2).strip()
                if DEBUG: print(num, slf1, end='  ')  
                lf1 = [float(_s) for _s in slf1.split()]
                if DEBUG: print( lf1)
                assert nTag == len(lf1)
                for t,f1 in zip(lTag, lf1): dL[t].append(100.0*f1)
            elif cre_PRF_begin.match(s):
                fd.readline() # skipping next line"
                bIn = True
                continue
        assert num+1 == len(lFN)

    # making a strict dict, to detect bugs
    return lFN, lTag, dict(dL) 

         

def parse_logs(lfn):
    """
    parse the log so that each file of first list is compared against each of the other list 
    """
    lFN  = list()
    lTag = list()
    dL   = None 

    for fn in lfn:
            try:
                _lfn, _lTag, _dL = parse_log(fn)
            except Exception as e:
                print("ERROR on file '%s'"%fn)
                raise e
            if lTag:
                if lTag != _lTag:
                    raise Exception("ERROR: tags do not match")               
            else:
                lTag = _lTag
                dL = {k:list() for k in lTag}   # strict dict
            lFN.extend(_lfn)
            for t in lTag:
                dL[t].extend(_dL[t])
    return lFN, lTag, dL


def print_f1_wilcox_by_tag(lTag1, dL1, lTag2, dL2, bSkipBothZeros=True):
    """
    paired-test on F1, by tag
    
    if bSkipBothZeros is True:
    We skip zero, because when a tag is absent from a document, our report says 0 while it is not a zero. :-(
    => If for some reason , the two model miss entirely a certain tag on a document, then we miss this information.
    """
    assert lTag1 == lTag2
    
    sTagFmt = "%%%ds" % max(len(t) for t in lTag1)  # longtest tag name
    
    N = None
    for k in lTag1:
        l1 = dL1[k]
        l2 = dL2[k]
        assert len(l1) == len(l2)
        if N is None:
            N = len(l1)
        else:
            assert N == len(l1)

        if bSkipBothZeros:
            l1b, l2b = [], []
            nz1 = 0 # case when at least one of method has 0 as performance
            for v1, v2 in zip(l1, l2):
                if v1 == 0 or v2 == 0: 
                    nz1 += 1
                    if v1 == 0 and v2 == 0: continue
                l1b.append(v1)
                l2b.append(v2)
            l1, l2 = l1b, l2b
            nz = N - len(l1)
        else:
            nz = 0
        sSkippedZeros = "(%5d 'paired' zeros skipped, = %5.1f%% )" % (nz, (100.0*nz)/nz1) if nz > 0 else ""

        m1, m2 = mean(l1), mean(l2)

        lD = [_v2 - _v1 for _v1, _v2 in zip(l1, l2)]  # delta
        mD = mean(lD)
        assert (mD - (m2-m1)) <= 0.001
        
        
        if l1 == l2:
            print("%s |  avg = %6.1f  IDENTICAL LISTS of %d values  %s" % (sTagFmt%k, m1, N, sSkippedZeros))
        else:
            print("%s |  avg1 = %6.1f, avg2 = %6.1f | avg_delta = %+7.2f  %10s" % (sTagFmt%k
                                                                                   , m1, m2, mD, "(± %.2f)"%stdev(lD)), end='')
            _, pv = scipy.stats.wilcoxon(l1, l2, zero_method="pratt")
#             spv = "(%ssign. at %.2f)" % ("not " if pv > alpha else "", alpha)
#             print(" : pvalue = %.3f   %20s   %s" % (pv, spv, sSkippedZeros))            
            spv = " " if pv > alpha else "*"
            print(" : pvalue = %.3f   %s   %s" % (pv, spv, sSkippedZeros))            

    return

# ----------------------------------------------------------------------------    
if __name__ == "__main__":
    if not(sys.argv[1:]):
        print("Usage:")
        print("python %s LOG1")
        print("python %s LOG1             LOG2")
        print("python %s LOG1a,LOG1b,..   LOG2a,LOG2b..")
        print("Do a comparison of paired files by passing a comma-separated list of log file.")
        sys.exit(1)
        
    if len(sys.argv) == 2:
        lfn1 = [sys.argv[1]]
        s1 = ""
        lFN1, lTag1, dL1 = parse_logs(lfn1)
        print("\t1  ", lTag1, "   ", lfn1)
        nFiles = len(dL1[lTag1[0]]) 
        print("%d test files" % nFiles)
        sTagFmt = "%%%ds" % max(len(t) for t in lTag1)  # longtest tag name
    
        for k in lTag1:
            l1 = dL1[k]
            l1z = [_v for _v in l1 if _v != 0]
            nZero = l1.count(0)
            print("%s |  avg = %6.1f  sdev=%6.1f  |  ignoring zeros: avg = %6.1f  sdev=%6.1f" % (sTagFmt%k
                                                     , mean(l1), stdev(l1)
                                                     , mean(l1z), stdev(l1z)
                                                     ))
    else:
        fn1, fn2 = sys.argv[1:3]
        s1, s2 = "", ""
        print("== Comparing series of F1 of tagging =="
              , "  (Paired Wilcoxon test using Pratt method for ties)")
        lfn1, lfn2 = fn1.split(','), fn2.split(',')
        if len(lfn1) > 1 or len(lfn2) > 1:
            print(" = comparing each of the %d first files to each of the %d other files"%(len(lfn1), len(lfn2)))
        else:
            lfn1, lfn2 = [fn1], [fn2]
            
        lFN1, lTag1, dL1 = parse_logs(lfn1)
        lFN2, lTag2, dL2 = parse_logs(lfn2)
    
    #     print(l1[0:5])
    #     print(l2[0:5])
        print("\t1  ", lTag1, "   ", lfn1)
        print("\t2  ", lTag2, "   ", lfn2)
        assert lTag1 == lTag2, "TODO: accept different tagsets, and compute what can be computed on common tags - NOT YET IMPLEMENTED"
        nFiles = len(dL1[lTag1[0]]) 
        print("%d test files" % nFiles)
        assert  nFiles == len(dL2[lTag2[0]])
        print_f1_wilcox_by_tag(lTag1, dL1, lTag2, dL2)
    
        if lFN1 != lFN2:
            raise Exception("Different filenames in each list of files")
        
    #     print(dL1['mi_item-number'][:10])
    #     print(dL1['mi_item-number'][-10:])
    #     print(dL1['mi_section-heading'][:10])
    #     print(dL1['mi_section-heading'][-10:])
    #     print(dL1)
        
        sys.exit(0)
    
