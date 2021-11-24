# -*- coding: utf-8 -*-

"""
    Determine if one clustering is significant better than another

    Assume (and check) that the 2 clustering are based on the same files
    
    June 2020
    Copyright Naver Labs Europe 2018
    JL Meunier
"""
import sys, os
import re
import collections
import math
import csv
from optparse import OptionParser

from statistics import mean, stdev

import scipy.stats

DEBUG = 0

alpha = 0.05

# find sections in case we do joint clustering like SMI
sre_NAMED_SECTION_HEAD  = "ALL_(%s)\s+@simil.*"

# within a section, find files
cre_HEAD = re.compile("PAGE\s+\d+\s+OF FILE\s+(.*)$")

re_tPRFoem = "@simil (\d+\.\d+)" \
        + "".join(["\s+%s\s+(\d+\.\d+)\s+" % _s for _s in ["P", "R", "F1"]])\
        + "".join(["%s=\s*(\d+)\s*" % _s for _s in ["ok", "err", "miss"]])
cre_tPRFoem = re.compile(re_tPRFoem)

          
def parse_log(s, sClusterName=""):
    """
    parse the log
    return a  list of tuple( <filename>
                            , dictionary of (P,R,F1, ok, err, miss) per threshold
                            )
    """
    sName, l = None, []

    if sClusterName == "" :
        sre = sre_NAMED_SECTION_HEAD % "[^\s]+"
    else:
        sre = sre_NAMED_SECTION_HEAD % sClusterName
    # print(sre)
    creSectionHead = re.compile(sre)

    bSectionSeen     = False
    bSectionDataSeen = False
    bOut = True
    sFile, d = None, None
    for sLine in s.splitlines():
        if not bOut:
            o = cre_tPRFoem.match(sLine)
            if o:
                d[o.group(1)] = tuple([float(o.group(i)) for i in range(2,5)] + [int(o.group(i)) for i in range(5,8)])
                bSectionDataSeen = True
            else:
                if d is not None:
                    l.append((sFile, d))
                sFile, d = None, None
                bOut = True
        o = creSectionHead.match(sLine)
        if o: 
            if bSectionSeen:
                if bSectionDataSeen:  # end of a section 
                    break
                #else, ignore consecutive lines like "ALL_ ..."
            else:
                sName = o.group(1)
                bSectionSeen = True
        elif bSectionSeen and bOut:
            o = cre_HEAD.match(sLine)
            if o:
                sFile = o.group(1)
                d = dict()
                bOut = False

    if not bOut:
        if d is not None:
            l.append((sFile, d))
    
    if sClusterName != "" and sName != sClusterName: 
        raise ValueError("No cluster '%s' found"%sClusterName)
    return l


                
def print_f1_wilcox_by_th(sName1, l1, sName2, l2):
    """
    paired-test on F1, by threshold
    """
    
    assert len(l1) == len(l2)
    # list of values, by threhold
    dl1, dl2 = collections.defaultdict(list), collections.defaultdict(list)
    
    for (fn1, d1), (fn2, d2) in zip(l1, l2):
        assert os.path.basename(fn1) == os.path.basename(fn2), "Error filename mismatch: %s %s"%(fn1, fn2)
        for dl, d in [(dl1, d1), (dl2, d2)]:
            for k, v in d.items(): dl[k].append(v)

    lk1 = sorted(dl1.keys())
    lk2 = sorted(dl2.keys())
    # print(lk1)
    assert lk1 == lk2, "different thresholds in each input: %s %s"%(lk1, lk2)
    
    N = None
    for k in lk1:
        l1 = [t[2] for t in dl1[k]]  # taking F1 score
        l2 = [t[2] for t in dl2[k]]
        
        #DEBUG!
#         if k.startswith("1"): l1 = [_v +10 for _v in l1]
#         if k.startswith("0.8"): l1 = [_v +0.00001 for _v in l2]
#         l1 = l1[:10]
#         l2 = l2[:10]
        
        if N is None: 
            N = len(l1)
            # effect size stuff"
            bEffectSize = True if N > 20 else False

            # effect size computation disabled, because buggy
            bEffectSize = None   # *************** TODO: get Z statistic

            # EXPLANATION
            # the Z statistic is not returned
            # I tried to compute it as explained in # https://github.com/scipy/scipy/issues/2625#issuecomment-20848886
            # but it makes no sense as adding 0.00001 to each value leads to a "large effect..."
            # l1 = [_v +0.00001 for _v in l2]
            # simil @ 0.80   avg1 =  66.78, avg2 =  66.78  (  -0.00) : pvalue = 0.000      (significant at 0.05) : effect size (r) = 0.992  (large effect)
            if bEffectSize is None or bEffectSize:
                print("%d values" % N)
            else:
                print("%d values  (Cannot compute effect size)" % N)
        assert len(l1) == N and len(l2) == N, "Different number of values: %d %d"%(len(l1), len(l2))

        
        m1, m2 = mean(l1), mean(l2)
        lD = [_v2 - _v1 for _v1, _v2 in zip(l1, l2)]  # delta
        mD = mean(lD)
        assert (mD - (m2-m1)) <= 0.001
        
        if l1 == l2:
            print("simil @ %s |  avg = %6.2f  IDENTICAL LISTS of %d values" % (k, m1, N))
        else:
            print("simil @ %s |  avg1 = %6.2f, avg2 = %6.2f | avg_delta = %+7.2f  %10s" % (k, m1, m2, mD, "(± %.2f)"%stdev(lD)), end='')
            if bEffectSize:
                # use approx and two-sided to make sure distributions.norm.sf is used 
                _, pv = scipy.stats.wilcoxon(l1, l2, zero_method="pratt"
                                             , mode="approx"    # so that Z is computed using distributions.norm.sf
                                             , correction=True  # due to my readings of http://vassarstats.net/textbook/ch12a.html
                                             )
                if pv > alpha:
                    s = None  # no sense in computing aneffect size
                else:
                    # https://github.com/scipy/scipy/issues/2625#issuecomment-20848886
                    z_abs = scipy.stats.norm.isf(pv / 2.)
                    assert abs(pv - 2. * scipy.stats.distributions.norm.sf(abs(z_abs))) < 0.0001
                    abs_effect_size = z_abs / math.sqrt(N)
                    
                    # effect size conventional scale (rule of thumb!)
                    # Sawilowsky, S. S. (2009). New effect size rules of thumb. Journal of Modern Applied Statistical Methods, 8(2), 597 – 599.
                    if   abs_effect_size >= 2.0 : s = "huge"
                    elif abs_effect_size >= 1.2 : s = "very large"
                    elif abs_effect_size >= 0.8 : s = "large"
                    elif abs_effect_size >= 0.5 : s = "medium"
                    elif abs_effect_size >= 0.2 : s = "small"
                    elif abs_effect_size >= 0.01: s = "very small"
                    else:                         s = "none"
                spv = "(%ssignificant at %.2f)" % ("not " if pv > alpha else "", alpha)
                print(" | pvalue = %5.3f" % pv
                      , " %25s" % spv
                      , "" if s is None else ": effect size (r) = %5.3f  (%s effect)" % (abs_effect_size, s))
            else:
                _, pv = scipy.stats.wilcoxon(l1, l2, zero_method="pratt")
                spv = "(%ssignificant at %.2f)" % ("not " if pv > alpha else "", alpha)
                print(" : pvalue = %.3f   %25s" % (pv, spv))            

    return lk1, dl1, dl2

    
def  createCsv(fCsv, lk, fn1, s1, l1, fn2, s2, l2):
    """
    l are a list of tuple( <filename>
                            , dictionary of (P,R,F1, ok, err, miss) per threshold
                            )
    """
    wrtr = csv.writer(fCsv)
    
    # --- file names
    wrtr.writerow([" ".join([fn1, s1, fn2, s2])])
    
    # --- columns headers
    lv = ["N", "file"]
    for k in lk:
        lv.append(k)
        for s in [s1, s2]:
            lv.extend(["F1_%s"%k, "ok", "err", "miss", "tot"])
    wrtr.writerow(lv)
    
    # --- values
    N = 1
    for (fn1, d1), (fn2, d2) in zip(l1, l2):
        assert os.path.basename(fn1) == os.path.basename(fn2), "Error filename mismatch: %s %s"%(fn1, fn2)
        lv = list([N, os.path.basename(fn1)])
        for k in lk:
            lv.append(k)
            for d in [d1, d2]:
                t6 = d[k]
                f1   = t6[2]
                ok   = t6[3]
                err  = t6[4]
                miss = t6[5]
                tot  = ok + miss
                lv.extend([f1, ok, err, miss, tot])
        wrtr.writerow(lv)
        N += 1
            
    fCsv.close()
    

def product_parse_logs(lfn1, lfn2, s1, s2):
    """
    parse the log so that each file of first list is compared against each of the other list 
    """
    ll1, ll2 = [],[]
    for ll, lfn, s in [(ll1, lfn1, s1), (ll2, lfn2, s2)]:
        for fn in lfn:
            try:
                ll.append(parse_log(open(fn).read(), s))
            except Exception as e:
                print("ERROR on file '%s'"%fn)
                raise e
                
                
    #     ll1 = [parse_log(open(_fn1).read(), s1) for _fn1 in lfn1]
    #     ll2 = [parse_log(open(_fn2).read(), s1) for _fn2 in lfn2]
    
    l1, l2 = [], []
    for _i1, _l1 in enumerate(ll1):
        for _i2, _l2 in enumerate(ll2):
            if DEBUG: print(lfn1[_i1] , " versus ", lfn2[_i2])
            l1.extend(_l1)
            l2.extend(_l2)
    return l1, l2

    
if __name__ == "__main__":

    usage = """
        [--csv CSVFILE]              LOG1              LOG2
        [--csv CSVFILE] CLUSTER_NAME LOG1 CLUSTER_NAME LOG2
Do a pairwise comparison of cluster quality        
CLUSTER_NAME typically is like: cluster or cluster_lvl0 ...

        [--csv CSVFILE]              LOG1a,LOG1b..              LOG2a,LOG2b..
Do a comparison of paired files by passing a comma-separated list of log file.        
Train and test a ECN or EnsembleECN model using data pickled (--lpkl option)

        --avg              LOG
        --avg CLUSTER_NAME LOG
Compute cluster average quality and standard deviation
"""
    
    parser = OptionParser(usage=usage, version=0.1)
    
    parser.add_option("--csv"       , dest='fnCsv'       ,  action="store"  , type="string"
                      , help="Store data in a csv file")    
    parser.add_option("--avg"       , dest='bAvg'       ,  action="store_true"
                      , help="Compute average and standard deviation for a single log file")    
 
    (options, args) = parser.parse_args()

    if not(args) or len(args) not in [1,2,4] or (options.bAvg and len(args)==4):
        print(usage)
        sys.exit(1)
        
    if options.fnCsv:
        fCsv = open(options.fnCsv, "w", newline='')  # CSV file to be created
    else:
        fCsv = None
    
    if options.bAvg:
        s1, fn1 = args if len(args) == 2 else ("", args[0])
        l1 = parse_log(open(fn1).read(), s1)
        print("%d values" % len(l1))
        print("\t1  ", "%20s"%s1, "   ", fn1)
        
        # list of values, by threshold
        dl1 = collections.defaultdict(list)
        for (fn1, d1) in l1:
            for k, v in d1.items(): dl1[k].append(v[2])  #F1
        lk1 = sorted(dl1.keys())
        for k in lk1:
            _l = dl1[k]
            print("simil @ %s |  avg1 = %6.2f   %10s" % (k, mean(_l), "(± %.2f)"%stdev(_l)))            
    else:
        if len(args) == 4:
            s1, fn1, s2, fn2 = args
            print("== Comparing series of F1 of clustering at given threshold =="
                  , "  (Paired Wilcoxon test using Pratt method for ties)")
            lfn1, lfn2 = fn1.split(','), fn2.split(',')
            if len(lfn1) > 1 or len(lfn2) > 1:
                print(" = comparing each of the %d first files to each of the %d other files"%(len(lfn1), len(lfn2)))
                l1, l2 = product_parse_logs(lfn1, lfn2, s1, s2)
            else:
                l1 = parse_log(open(fn1).read(), s1)
                l2 = parse_log(open(fn2).read(), s2)
        else:
            fn1, fn2 = args
            s1, s2 = "", ""
            print("== Comparing series of F1 of clustering at given threshold =="
                  , "  (Paired Wilcoxon test using Pratt method for ties)")
            lfn1, lfn2 = fn1.split(','), fn2.split(',')
            if len(lfn1) > 1 or len(lfn2) > 1:
                print(" = comparing each of the %d first files to each of the %d other files"%(len(lfn1), len(lfn2)))
                l1, l2 = product_parse_logs(lfn1, lfn2, s1, s2)
            else:
                l1 = parse_log(open(fn1).read())
                l2 = parse_log(open(fn2).read())
    
    #     print(l1[0:5])
    #     print(l2[0:5])
        print("\t1  ", "%20s"%s1, "   ", fn1)
        print("\t2  ", "%20s"%s2, "   ", fn2)
        lk, _dl1, _dl2 = print_f1_wilcox_by_th(s1, l1, s2, l2)
        
        if bool(fCsv):
            createCsv(fCsv, lk, fn1, s1, l1, fn2, s2, l2)
            print("See ", options.fnCsv)
        
    sys.exit(0)
    
# =============================================================================

_s_test_parse =  """ done [43.20s]
ALL_cluster  @simil 0.66   P 50.00  R 29.64  F1 37.22         ok=91  err=91  miss=216
ALL_cluster  @simil 0.80   P 41.76  R 24.76  F1 31.08         ok=76  err=106  miss=231
ALL_cluster  @simil 1.00   P 29.12  R 17.26  F1 21.68         ok=53  err=129  miss=254

PAGE     1  OF FILE    /tmp-network/user/meunier/menus/20200526-vf/run.1434466/col/08_22_17-DINNERhsl.pdf.ner_du.mpxml
@simil 0.66   P  0.00  R  0.00  F1  0.00   ok=     0  err=     6  miss=     5
@simil 0.80   P  0.00  R  0.00  F1  0.00   ok=     0  err=     6  miss=     5
@simil 1.00   P  0.00  R  0.00  F1  0.00   ok=     0  err=     6  miss=     5
PAGE     1  OF FILE    /tmp-network/user/meunier/menus/20200526-vf/run.1434466/col/japa_delivery.pdf-9.ner_du.mpxml
@simil 0.66   P 57.14  R 57.14  F1 57.14   ok=     4  err=     3  miss=     3
@simil 0.80   P 57.14  R 57.14  F1 57.14   ok=     4  err=     3  miss=     3
@simil 1.00   P 42.86  R 42.86  F1 42.86   ok=     3  err=     4  miss=     4"""

def test_parse():

    ref = [ ("/tmp-network/user/meunier/menus/20200526-vf/run.1434466/col/08_22_17-DINNERhsl.pdf.ner_du.mpxml"
             , {
"0.66" :  (0.00 ,  0.00 , 0.00   ,     0  ,     6  ,     5),
"0.80" :  (0.00 ,  0.00 , 0.00   ,     0  ,     6  ,     5),
"1.00" :  (0.00 ,  0.00 , 0.00   ,     0  ,     6  ,     5)
                }
             )
        ,   ("/tmp-network/user/meunier/menus/20200526-vf/run.1434466/col/japa_delivery.pdf-9.ner_du.mpxml"
             , {
"0.66" :  (57.14 , 57.14 , 57.14   ,     4  ,     3  ,     3),
"0.80" :  (57.14 , 57.14 , 57.14   ,     4  ,     3  ,     3),
"1.00" :  (42.86 , 42.86 , 42.86   ,     3  ,     4  ,     4)             
                 }
            )
        ]
    
    out = parse_log(_s_test_parse)
    for t in out: print(t)
    assert ref == out, out

_s_test_parse_2 = """ done [43.20s]
ALL_cluster  @simil 0.66   P 50.00  R 29.64  F1 37.22         ok=91  err=91  miss=216
ALL_cluster  @simil 0.80   P 41.76  R 24.76  F1 31.08         ok=76  err=106  miss=231
ALL_cluster  @simil 1.00   P 29.12  R 17.26  F1 21.68         ok=53  err=129  miss=254

PAGE     1  OF FILE    /tmp-network/user/meunier/menus/20200526-vf/run.1434466/col/08_22_17-DINNERhsl.pdf.ner_du.mpxml
@simil 0.66   P  0.00  R  0.00  F1 30.00   ok=     0  err=     6  miss=     5
@simil 0.80   P  0.00  R  0.00  F1 20.00   ok=     0  err=     6  miss=     5
@simil 1.00   P  0.00  R  0.00  F1 10.00   ok=     0  err=     6  miss=     5
PAGE     1  OF FILE    /tmp-network/user/meunier/menus/20200526-vf/run.1434466/col/japa_delivery.pdf-9.ner_du.mpxml
@simil 0.66   P 57.14  R 57.14  F1 57.14   ok=     4  err=     3  miss=     3
@simil 0.80   P 57.14  R 57.14  F1 57.14   ok=     4  err=     3  miss=     3
@simil 1.00   P 42.86  R 42.86  F1 42.86   ok=     3  err=     4  miss=     4"""

_s_test_parse_3 = """ done [43.20s]
ALL_cluster  @simil 0.66   P 50.00  R 29.64  F1 37.22         ok=91  err=91  miss=216
ALL_cluster  @simil 0.80   P 41.76  R 24.76  F1 31.08         ok=76  err=106  miss=231
ALL_cluster  @simil 1.00   P 29.12  R 17.26  F1 21.68         ok=53  err=129  miss=254

PAGE     1  OF FILE    /tmp-network/user/meunier/menus/20200526-vf/run.1434466/col/08_22_17-DINNERhsl.pdf.ner_du.mpxml
@simil 0.66   P  0.00  R  0.00  F1 30.00   ok=     0  err=     6  miss=     5
@simil 0.80   P  0.00  R  0.00  F1 20.00   ok=     0  err=     6  miss=     5
@simil 1.00   P  0.00  R  0.00  F1 10.00   ok=     0  err=     6  miss=     5
PAGE     1  OF FILE    /tmp-network/user/meunier/menus/20200526-vf/run.1434466/col/japa_delivery.pdf-9.ner_du.mpxml
@simil 0.66   P 57.14  R 57.14  F1 67.14   ok=     4  err=     3  miss=     3
@simil 0.80   P 57.14  R 57.14  F1 67.14   ok=     4  err=     3  miss=     3
@simil 1.00   P 42.86  R 42.86  F1 52.86   ok=     3  err=     4  miss=     4"""


def test_same():
    out1 = parse_log(_s_test_parse)
    out2 = parse_log(_s_test_parse_2)
    out3 = parse_log(_s_test_parse_3)
        
    print_f1_wilcox_by_th(out1, out2)
    print_f1_wilcox_by_th(out1*10, out2*10)
    print_f1_wilcox_by_th(out1, out3)
    print_f1_wilcox_by_th(out1+out1, out3+out3)

