# -*- coding: utf-8 -*-

"""
    Find cuts of a page along different slopes 
    and annotate them based on the table row content (which defines a partition)
    
    Copyright Naver Labs Europe 2018
    JL Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""
import sys, os
from optparse import OptionParser
from lxml import etree
from collections import defaultdict

import numpy as np

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    sys.path.append( os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) ))
    import TranskribusDU_version

from common.trace import traceln
from util.metrics import evalAdjustedRandScore, average, stddev
from xml_formats.PageXml import PageXml
from tasks.DU_Table.DU_ABPTableSkewed_CutAnnotator import op_cut, op_eval_row, op_gt_recall
from tasks.DU_Table.DU_ABPTableCutAnnotator import CutAnnotator, op_eval_col
from tasks.DU_Table.DU_Table_CellBorder import op_eval_cell
from util.partitionEvaluation import evalPartitions
from util.jaccard import jaccard_distance

from util.hungarian import evalHungarian
if True:
    # We fix a bug in this way, to keep old code ready to use
    evalPartitions = evalHungarian

from graph.NodeType_PageXml     import defaultBBoxDeltaFun
from graph.NodeType_PageXml     import NodeType_PageXml_type_woText

from graph.pkg_GraphBinaryConjugateSegmenter.MultiSinglePageXml \
    import MultiSinglePageXml \
    as ConjugateSegmenterGraph_MultiSinglePageXml

from graph.pkg_GraphBinaryConjugateSegmenter.MultiSinglePageXml_Separator \
    import MultiSinglePageXml_Separator \
    as ConjugateSegmenterGraph_MultiSinglePageXml_Separator

def listFiles(sDir,ext="_du.mpxml"):
    """
    return 1 list of files
    """
    lsFile = sorted([_fn 
                     for _fn in os.listdir(sDir) 
                     if _fn.lower().endswith(ext) or _fn.lower().endswith(ext)
                     ])
    return lsFile


     
def listParallelFiles(lsDocDir):
    """
    return 1 list of file per folder, as a tuple
    
    Make sure the filenames correspond in each folder
    """
    llsFile = [listFiles(sDir) for sDir in lsDocDir]
    
    if len(lsDocDir) > 1:
        # correspondance tests
        lset = [set(l) for l in llsFile]
        setInter = lset[0].intersection(*lset[1:]) 
        setUnion = lset[0].union(*lset[1:]) 
        if setInter != setUnion:
            for setFile, sDir in zip(lset, lsDocDir):
                setExtra = setFile.difference(setInter)
                if len(setExtra) > 0:
                    traceln("\t %s has %d extra files: %s" % (sDir, len(setExtra), sorted(list(setExtra))))
                else:
                    traceln("\t %s is OK")
            raise Exception("Folders contain different filenames")
    
    return llsFile


def computePRF(nOk, nErr, nMiss):
    eps = 0.00001
    fP = 100 * nOk / (nOk + nErr + eps)
    fR = 100 * nOk / (nOk + nMiss + eps)
    fF = 2 * fP * fR / (fP + fR + eps)
    return fP, fR, fF


class My_ConjugateNodeType(NodeType_PageXml_type_woText):
# class My_ConjugateNodeType(NodeType_PageXml_type):
    """
    We need this to extract properly the label from the label attribute of the (parent) TableCell element.
    """
    def __init__(self, sNodeTypeName, lsLabel, lsIgnoredLabel=None, bOther=True, BBoxDeltaFun=defaultBBoxDeltaFun):
        super(My_ConjugateNodeType, self).__init__(sNodeTypeName, lsLabel, lsIgnoredLabel, bOther, BBoxDeltaFun)

    def parseDocNodeLabel(self, graph_node, defaultCls=None):
        """
        Parse and set the graph node label and return its class index
        raise a ValueError if the label is missing while bOther was not True, or if the label is neither a valid one nor an ignored one
        """
        domnode = graph_node.node
        ndParent = domnode.getparent()
        sLabel = "%s__%s" % (  ndParent.getparent().get("id")  # TABLE ID !
                               , self.sLabelAttr               # e.g. "row" or "col"
                               )
        return sLabel

    def setDocNodeLabel(self, graph_node, sLabel):
        raise Exception("This shoud not occur in conjugate mode")    
    
    
def getConfiguredGraphClass(bSeparator=False):
    """
    In this class method, we must return a configured graph class
    """
    # each graph reflects 1 page
    if bSeparator:
        DU_GRAPH = ConjugateSegmenterGraph_MultiSinglePageXml_Separator
    else:
        DU_GRAPH = ConjugateSegmenterGraph_MultiSinglePageXml

    # ntClass = NodeType_PageXml_type
    ntClass = My_ConjugateNodeType

    nt = ntClass("row"                   #some short prefix because labels below are prefixed with it
                  , []                   # in conjugate, we accept all labels, andNone becomes "none"
                  , []
                  , False                # unused
                , BBoxDeltaFun=lambda v: max(v * 0.066, min(5, v/3))  #we reduce overlap in this way
#                   , BBoxDeltaFun= None

                  )    
    nt.setLabelAttribute("idontcare")
    nt.setXpathExpr( (".//pc:TextLine"        #how to find the nodes            
                      , ".//pc:Unicode")       #how to get their text
                   )
    DU_GRAPH.addNodeType(nt)
    
    return DU_GRAPH

def labelEdges(g,sClusterLevel):
    """
        g: DU_graph
        label g edges as 0 (continue) or 1 (break) according to GT structure
        
    """
    Y = np.zeros((len(g.lEdge),2))
    for i,edge in enumerate(g.lEdge):
        a, b = edge.A.node, edge.B.node
        if sClusterLevel == "cell":
            if (a.getparent().get("row"), a.getparent().get("col")) == (b.getparent().get("row"), b.getparent().get("col")):
                a.set('DU_cluster', "%s__%s" % (a.get("row"), a.get("col")))
                b.set('DU_cluster', "%s__%s" % (a.get("row"), a.get("col")))
                Y[i][0]=1
            else: Y[i][1]=1
        elif sClusterLevel == "col":
            if a.get("col") == b.get("col"):
                a.set('DU_cluster', "%s" % (a.get("col")))
                b.set('DU_cluster', "%s" % (a.get("col")))
                Y[i][0]=1
            else: Y[i][1]=1
        elif sClusterLevel == "row":
            if a.getparent().get("row") == b.getparent().get("row") and  a.getparent().get("row") is not None:
                tablea = a.getparent().getparent().get('id')
                tableb = b.getparent().getparent().get('id')
                a.set('DU_cluster', "%s_%s" % (a.getparent().get("row"),"_" + tablea))
                b.set('DU_cluster', "%s_%s" % (a.getparent().get("row"),"_" + tableb))                    
                if tablea == tableb:
                    Y[i][0]=1
                else:
                    Y[i][1]=1
            else: Y[i][1]=1
        else:        
            raise Exception("Unknown clustering level: %s"%sClusterLevel)
    
    return Y

def eval_oracle(lsRunDir, sClusterLevel
                , bIgnoreHeader=True
                , bIgnoreOutOfTable=True
                , lfSimil=[i / 100.0 for i in [66, 80, 100]]
                , xpSelector=".//pc:TextLine"):
    """
    evaluate the cluster quality from a run folder
    
    We assume to have the groundtruth row and col in the files as well as the predicted clusters
    """
    assert lsRunDir
    dOkErrMiss = { fSimil:(0,0,0) for fSimil in lfSimil }

    DU_GraphClass = getConfiguredGraphClass()
    
    for sRunDir in lsRunDir:
        lsFile = listFiles(sRunDir,ext='.pxml')
        traceln("-loaded %d files from %s" % (len(lsFile), sRunDir))
        
        for sFilename in lsFile:
            
            #
            lg = DU_GraphClass.loadGraphs(DU_GraphClass, [os.path.join(sRunDir, sFilename)], bDetach=False, bLabelled=False, iVerbose=1)
            
            # cluster  -> [node_id]
            dGT  = defaultdict(list)
            dRun = defaultdict(list)
            
#             doc = etree.parse(os.path.join(sRunDir, sFilename))
            # assume 1 page per doc!
            g=lg[0]
            rootNd = g.doc.getroot()
            #assert len(PageXml.xpath(rootNd, "//pc:Page")) == 1, "NOT YET IMPLEMENTED: eval on multi-page files"
            for iPage, ndPage in enumerate(PageXml.xpath(rootNd, "//pc:Page")):
                traceln("PAGE %5d  OF FILE    %s" % (iPage+1, sFilename))
                
                try:g = lg[iPage]
                except IndexError:continue
                Y = labelEdges(g,sClusterLevel)
                g.form_cluster(Y)
                g.addEdgeToDoc()
                
                for nd in PageXml.xpath(ndPage, xpSelector):
                    if bIgnoreHeader and nd.getparent().get("custom") and "table-header" in nd.getparent().get("custom"): continue
#                     if bIgnoreHeader and nd.get("DU_header") != "D": continue
                    
                    ndparent = nd.getparent() 
                    ndid   = nd.get("id")
                    
                    if sClusterLevel == "cell":
                        val_gt = "%s__%s" % (ndparent.get("row"), ndparent.get("col"))
                        if val_gt == 'None__None' and bIgnoreOutOfTable: continue
                    elif sClusterLevel == "col":
                        val_gt = ndparent.get("col")
                        if val_gt == None and bIgnoreOutOfTable: continue
                    elif sClusterLevel == "row":
                        val_gt = ndparent.get("row")
                        if val_gt == None and bIgnoreOutOfTable: continue
                    else:
                        raise Exception("Unknown clustering level: %s"%sClusterLevel)
    
                    # distinguish each table!
                    val_gt = val_gt + "_" + ndparent.getparent().get("id")
                    
                    dGT[val_gt].append(ndid)
     
                    val_run = nd.get("DU_cluster")
                    dRun[val_run].append(ndid)
#                     assert ndparent.tag.endswith("TableCell"), "expected TableCell got %s" % nd.getparent().tag
                    
                for fSimil in lfSimil:
                    _nOk, _nErr, _nMiss = evalPartitions(
#                     _nOk, _nErr, _nMiss, _lFound, _lErr, _lMissed = evalPartitions(
                          list(dRun.values())
                        , list(dGT.values())
                        , fSimil
                        , jaccard_distance)
                    
                    _fP, _fR, _fF = computePRF(_nOk, _nErr, _nMiss)
                    
                    #traceln("simil:%.2f  P %5.2f  R %5.2f  F1 %5.2f   ok=%6d  err=%6d  miss=%6d" %(
                    traceln("@simil %.2f   P %5.2f  R %5.2f  F1 %5.2f   ok=%6d  err=%6d  miss=%6d" %(
                          fSimil
                        , _fP, _fR, _fF
                        , _nOk, _nErr, _nMiss
                        ))
    #                     , os.path.basename(sFilename)))
    #                 sFilename = "" # ;-)
                    
                    # global count
                    nOk, nErr, nMiss = dOkErrMiss[fSimil]
                    nOk   += _nOk
                    nErr  += _nErr
                    nMiss += _nMiss
                    dOkErrMiss[fSimil] = (nOk, nErr, nMiss)
                
            traceln()
            g.doc.write(os.path.join(sRunDir, sFilename)+'.oracle')
        
    for fSimil in lfSimil:
        nOk, nErr, nMiss = dOkErrMiss[fSimil]
        fP, fR, fF = computePRF(nOk, nErr, nMiss)
        traceln("ALL_TABLES  @simil %.2f   P %5.2f  R %5.2f  F1 %5.2f " % (fSimil, fP, fR, fF )
                , "        "
                ,"ok=%d  err=%d  miss=%d" %(nOk, nErr, nMiss))
        
    return (nOk, nErr, nMiss)

def eval_direct(lCriteria, lsDocDir
                , bIgnoreHeader=False
                , bIgnoreOutOfTable=True
                , lfSimil=[i / 100.0 for i in [66, 80, 100]]
                , xpSelector=".//pc:TextLine"):
    """
    use the row, col, DU_row, DU_col XML attributes to form the partitions
    
    lCriteria is a list containg "row" or "col" or both
    """
    assert lsDocDir
    
    llsFile = listParallelFiles(lsDocDir)
    traceln("-loaded %d files for each criteria"%len(llsFile[0]))
    
    dOkErrMiss = { fSimil:(0,0,0) for fSimil in lfSimil }

    def _reverseDictionary(d):
        rd = defaultdict(list)
        for k, v in d.items():
            rd[v].append(k)
        return rd
 
    for i, lsCritFile in enumerate(zip(*llsFile)):
        assert len(lCriteria) == len(lsCritFile)
        
        # node_id -> consolidated_criteria_values
        dIdValue    = defaultdict(str)
        dIdValue_GT = defaultdict(str)
        for crit, sFilename, sDir in zip(lCriteria, lsCritFile, lsDocDir):
            doc = etree.parse(os.path.join(sDir, sFilename))
            rootNd = doc.getroot()
            assert len(PageXml.xpath(rootNd, "//pc:Page")) == 1, "NOT YET IMPLEMENTED: eval on multi-page files"
             
            for nd in PageXml.xpath(rootNd, xpSelector):
                ndid   = nd.get("id")
                val_gt = nd.getparent().get(crit)
                if val_gt is None:
                    if bIgnoreOutOfTable: 
                        continue
                    else:
                        val_gt = "-1"
                #if bIgnoreHeader and nd.get("DU_header") != "D": continue
                if bIgnoreHeader and nd.getparent().get("custom") and "table-header" in nd.getparent().get("custom"): continue
                assert nd.getparent().tag.endswith("TableCell"), "expected TableCell got %s" % nd.getparent().tag
                val    = nd.get("DU_"+crit)
#                 import random
#                 if random.random() < 0.10:
#                     val    = nd.get("DU_"+crit)
#                 else:
#                     val    = nd.getparent().get(crit)
                dIdValue[ndid]    += "_%s_" % val
                dIdValue_GT[ndid] += "_%s_" % val_gt  
#         print("**run ", str(dIdValue))
#         print("**GT  ", str(dIdValue_GT))

        # reverse dicitonaries
        dValue_lId    = _reverseDictionary(dIdValue)
        dValue_lId_GT = _reverseDictionary(dIdValue_GT)
        
#         print("run ", list(dValue_lId.values()))
#         print("GT  ", list(dValue_lId_GT.values()))
        for fSimil in lfSimil:
            _nOk, _nErr, _nMiss = evalPartitions(
                  list(dValue_lId.values())
                , list(dValue_lId_GT.values())
                , fSimil
                , jaccard_distance)
            
            _fP, _fR, _fF = computePRF(_nOk, _nErr, _nMiss)
            
            traceln("simil:%.2f  P %5.2f  R %5.2f  F1 %5.2f   ok=%6d  err=%6d  miss=%6d  %s" %(
                  fSimil
                , _fP, _fR, _fF
                , _nOk, _nErr, _nMiss
                , os.path.basename(sFilename)))
            sFilename = "" # ;-)
            nOk, nErr, nMiss = dOkErrMiss[fSimil]
            nOk   += _nOk
            nErr  += _nErr
            nMiss += _nMiss
            dOkErrMiss[fSimil] = (nOk, nErr, nMiss)
        traceln()
        
    for fSimil in lfSimil:
        nOk, nErr, nMiss = dOkErrMiss[fSimil]
        fP, fR, fF = computePRF(nOk, nErr, nMiss)
        traceln("ALL simil:%.2f  P %5.2f  R %5.2f  F1 %5.2f " % (fSimil, fP, fR, fF )
                , "        "
                ,"ok=%d  err=%d  miss=%d" %(nOk, nErr, nMiss))
        
    return (nOk, nErr, nMiss)
    

def eval_cluster(lsRunDir, sClusterLevel
                , bIgnoreHeader=False
                , bIgnoreOutOfTable=True
                , lfSimil=[i / 100.0 for i in [66, 80, 100]]
                , xpSelector=".//pc:TextLine"
                , sAlgo=None
                , sGroupByAttr=""
                ):
    """
    evaluate the cluster quality from a run folder
    
    We assume to have the groundtruth row and col in the files as well as the predicted clusters
    """
    assert lsRunDir
    traceln(" --- eval_cluster  level=%s"%sClusterLevel)
    nOk, nErr, nMiss = 0,0,0
    for sRunDir in lsRunDir:
        lsFile = listFiles(sRunDir)
        traceln("-loaded %d files from %s" % (len(lsFile), sRunDir))
        if not(lsFile) and not os.path.normpath(sRunDir).endswith("col"):
            # ... checking folders
            sRunDir = os.path.join(sRunDir, "col")
            lsFile = listFiles(sRunDir)
            if lsFile:
                traceln("-loaded %d files from %s" % (len(lsFile), sRunDir))

        if not sAlgo is None:
            traceln("Loading cluster @algo='%s'"%sAlgo)
            
        _nOk, _nErr, _nMiss, sRpt, _ = eval_cluster_of_files([os.path.join(sRunDir, _s) for _s in lsFile], sClusterLevel
            , bIgnoreHeader=bIgnoreHeader
            , bIgnoreOutOfTable=bIgnoreOutOfTable
            , lfSimil=lfSimil
            , xpSelector=xpSelector
            , sAlgo=sAlgo
            , sGroupByAttr=sGroupByAttr)
        nOk += _nOk
        nErr += _nErr
        nMiss += _nMiss
        traceln(sRpt)
        
    return (nOk, nErr, nMiss)

def eval_cluster_of_files(lsFilename
                , sClusterLevel  # either "row", "col", "cell", "region", "cluster"
                , bIgnoreHeader=False
                , bIgnoreOutOfTable=False
                , lfSimil=[i / 100.0 for i in [66, 80, 100]]
                , xpSelector=".//pc:TextLine"
                , sAlgo=None
                , sGroupByAttr=""
                , sClusterGTAttr=None  # used when sClusterLevel=="cluster"
                , bVerbose=False
                , bParseFile=True   # set to false if you directly pass the DOM
                ):
    """
    return nOk, nErr, nMiss, sRpt, dOkErrMiss
        where nOk, nErr, nMiss relate to the last value of lfSimil, yeah...
    """
    bTable = sClusterLevel in ["row", "col", "cell"]
    #if not bTable: assert sClusterLevel in ['region', 'cluster']
    # sCluelsterLevel can be CLuster or clusterlvl1, 2, ...
    if not bTable: assert sClusterLevel == 'region' or sClusterLevel.startswith('cluster')  
    
    dOkErrMiss = { fSimil:(0,0,0) for fSimil in lfSimil }
    dlfARI     = { fSimil:list()  for fSimil in lfSimil }
    lsRpt = []
    for sFilename in lsFilename:
        doc = etree.parse(sFilename) if bParseFile else sFilename
        rootNd = doc.getroot()
        #assert len(PageXml.xpath(rootNd, "//pc:Page")) == 1, "NOT YET IMPLEMENTED: eval on multi-page files"
        for iPage, ndPage in enumerate(PageXml.xpath(rootNd, "//pc:Page")):
            lsRpt.append("PAGE %5d  OF FILE    %s" % (iPage+1, sFilename))
            # cluster  -> [node_id]
            dGT  = defaultdict(list)
            dRun = defaultdict(list)
            for nd in PageXml.xpath(ndPage, xpSelector):
                #if bIgnoreHeader and nd.get("DU_header") != "D": continue
                if bIgnoreHeader and nd.getparent().get("custom") and "table-header" in nd.getparent().get("custom"): continue
                
                ndparent = nd.getparent() 
                ndid   = nd.get("id")
                val_run = nd.get("DU_cluster")  # ok in most cases
                if bTable:
                    if sClusterLevel == "cell":
                        val_gt = "%s__%s" % (ndparent.get("row"), ndparent.get("col"))
                        if val_gt == 'None__None' and bIgnoreOutOfTable: continue
                    else: # "col" or "row"
                        val_gt = ndparent.get(sClusterLevel)
                        if val_gt == None and bIgnoreOutOfTable: continue
                    # distinguish each table!
                    val_gt = "%s_%s" % (val_gt, ndparent.getparent().get("id"))
                else:
                    if sClusterLevel == 'region':
                        val_gt = ndparent.get("id")
                        # WHY???  if val_gt == 'None' and bIgnoreOutOfTable: continue
                    #elif sClusterLevel == 'cluster':
                    elif sClusterLevel == 'cluster':
                        val_gt = nd.get(sClusterGTAttr)
                    elif sClusterLevel.startswith('cluster_lvl'):
                        val_gt = nd.get(sClusterGTAttr)
                        val_run = nd.get("DU_"+sClusterLevel)
                    else:
                        raise Exception("Unknown clustering level: %s"%sClusterLevel)
               
                dGT[val_gt].append(ndid)
 
                dRun[val_run].append(ndid)
                #assert ndparent.tag.endswith("TableCell"), "expected TableCell got %s" % nd.getparent().tag
            
            if not sAlgo is None:
                dRun = defaultdict(list)
                lNdCluster = PageXml.xpath(ndPage, ".//pc:Cluster[@algo='%s']"%sAlgo)
                # lNdCluster = PageXml.xpath(ndPage, ".//pc:Cluster[@algo='%s' and @rowSpan='1']"%sAlgo)
                traceln("Loaded %d cluster @algo='%s'"%(len(lNdCluster), sAlgo))
                for iCluster, ndCluster in enumerate(lNdCluster):
                    sIDs = ndCluster.get("content")
                    lndid = sIDs.split()
                    if lndid: 
                        if sGroupByAttr:
                            # we group them by the value of an attribute
                            dRun[ndCluster.get(sGroupByAttr)].extend(lndid)
                        else:
                            dRun[str(iCluster)] = lndid
            
            if bVerbose:
                lkGT  = sorted(list(dGT.keys()))
                for k in lkGT: print("GT  CLuster", k, dGT[k])
                lkRun = sorted(list(dRun.keys()))
                for k in lkRun: print("RUN CLuster", k, dRun[k])
                
            for fSimil in lfSimil:
                lCluster_run = list(dRun.values())
                lCluster_gt  = list(dGT.values())
                _nOk, _nErr, _nMiss = evalPartitions(lCluster_run, lCluster_gt
                                                     , fSimil
                                                     , jaccard_distance)
                
                _fP, _fR, _fF = computePRF(_nOk, _nErr, _nMiss)
                _fARI = evalAdjustedRandScore(lCluster_run, lCluster_gt)
                #traceln("simil:%.2f  P %5.2f  R %5.2f  F1 %5.2f   ok=%6d  err=%6d  miss=%6d" %(
                lsRpt.append("@simil %.2f   P %5.2f  R %5.2f  F1 %5.2f   ok=%6d  err=%6d  miss=%6d  ARI=%.3f" %(
                      fSimil
                    , _fP, _fR, _fF
                    , _nOk, _nErr, _nMiss
                    , _fARI
                    ))
#                     , os.path.basename(sFilename)))
#                 sFilename = "" # ;-)
                
                # global count
                nOk, nErr, nMiss = dOkErrMiss[fSimil]
                nOk   += _nOk
                nErr  += _nErr
                nMiss += _nMiss
                dOkErrMiss[fSimil] = (nOk, nErr, nMiss)
                dlfARI[fSimil].append(_fARI)

        if bParseFile: del doc  #better memory management
        
    lSummary = []
    for fSimil in lfSimil:
        nOk, nErr, nMiss = dOkErrMiss[fSimil]
        fP, fR, fF = computePRF(nOk, nErr, nMiss)
        try:        fARI_avg, fARI_sdv = average(dlfARI[fSimil]), stddev(dlfARI[fSimil]) 
        except:     fARI_avg, fARI_sdv = -1, -1
        sLine = "ALL_%s  @simil %.2f   P %5.2f  R %5.2f  F1 %5.2f    ARI %.3f (sdv %.3f)" % (
                     "TABLES" if bTable else sClusterLevel, fSimil, fP, fR, fF, fARI_avg, fARI_sdv) \
                + "        "                                                                    \
                +"ok=%d  err=%d  miss=%d" %(nOk, nErr, nMiss)
        lSummary.append(sLine)
    
    sRpt = "\n".join(lsRpt) + "\n\n" + "\n".join(lSummary)
    
    return nOk, nErr, nMiss, sRpt, dOkErrMiss
    
    
# ------------------------------------------------------------------
if __name__ == "__main__":
    usage = """
    cut        INPUT_FILE  OUTPUT_FILE    to cut

    eval_cut_row       FILE+          eval of partitions from cuts 
    eval_cut_col       FILE+
    eval_cut_cell      ROW_DIR COL_DIR
    eval_cut_gt_recall FILE+          maximum obtainablerecall by cutting
    
    eval_direct_row   ROW_DIR           eval of partitions from DU_row index 
    eval_direct_col   COL_DIR
    eval_direct_cell  ROW_DIR COL_DIR

    eval_cluster      RUN_DIR+     eval the quality of the CELLs clusters using the GT clusters defined by table@ID+@row+@col
    eval_cluster_cell RUN_DIR+     eval the quality of the CELLs clusters using the GT clusters defined by table@ID+@row+@col
    eval_cluster_col  RUN_DIR+     eval the quality of the COLs  clusters using the GT clusters defined by table@ID     +@col
    eval_cluster_row  RUN_DIR+     eval the quality of the ROWs  clusters using the GT clusters defined by table@ID+@row
    --group_by_attr <ATTR>  merge cluster according to the value of their ATTR attribute
    
    oracle_cell RUN_DIR+     eval the quality of the CELLs clusters using the GT clusters defined by table@ID+@row+@col
    oracle_col  RUN_DIR+     eval the quality of the COLs  clusters using the GT clusters defined by table@ID     +@col
    oracle_row  RUN_DIR+     eval the quality of the ROWs  clusters using the GT clusters defined by table@ID+@row
    if --algo is specified, then the run output is taken from the CLuster definitions
    """
    parser = OptionParser(usage=usage, version="0.1")
    parser.add_option("--cut-height", dest="fCutHeight", default=10
                      ,  action="store", type=float
                      , help="cut, gt_recall: Minimal height of a cut")
     
    parser.add_option("--simil", dest="lfSimil", default=None
                      ,  action="append", type=float
                      , help="Minimal similarity for associating 2 partitions") 
    
    parser.add_option("--cut-angle", dest='lsAngle'
                      ,  action="store", type="string", default="0"
                        ,help="cut, gt_recall: Allowed cutting angles, in degree, comma-separated") 
    
    parser.add_option("--cut-below", dest='bCutBelow',  action="store_true", default=False
                        ,help="cut, eval_row, eval_cell, gt_recall: (OBSOLETE) Each object defines one or several cuts above it (instead of above as by default)")
     
#     parser.add_option("--cut-above", dest='bCutAbove',  action="store_true", default=None
#                         , help="Each object defines one or several cuts above it (instead of below as by default)") 
    
    parser.add_option("-v", "--verbose", dest='bVerbose',  action="store_true", default=False)

    parser.add_option("--algo", dest='sAlgo',  action="store", type="string"
                      , help="Use the cluster definition by given algo, not the @DU_cluster attribute of the nodes")

    parser.add_option("--ignore-header", dest='bIgnoreHeader',  action="store_true", default=False
                        , help="eval_row: ignore header text (and ignore empty cells, so ignore header cells!)")

    parser.add_option("--group_by_attr", dest='sGroupByAttr',  action="store", type="string"
                      , help=" merge cluster according to the value of their ATTR attribute")

    parser.add_option("--ratio", dest='fRatio', action="store"
                      , type=float
                      , help="eval_col, eval_cell : Apply this ratio to the bounding box. This is normally useless as the baseline becomes a point (the centroid)"
                      , default=CutAnnotator.fRATIO) 

    
    # --- 
    #parse the command line
    (options, args) = parser.parse_args()
    
    options.bCutAbove = not(options.bCutBelow)
    
    #load mpxml 
    try:
        op = args[0]
    except:
        traceln(usage)
        sys.exit(1)
    
    traceln("--- %s ---"%op)
    if op in ["eval", "eval_row", "eval_col", "eval_cell"]:
        if op == "eval": op = "eval_row"
        traceln("DEPRECATED: now use ", op[0:4] + "_cut" + op[4:])
        exit(1)
    # --------------------------------------
    if op == "cut":
        sFilename = args[1]
        sOutFilename = args[2]
        traceln("- cutting : %s  --> %s" % (sFilename, sOutFilename))
        lDegAngle = [float(s) for s in options.lsAngle.split(",")]
        traceln("- Allowed angles (°): %s" % lDegAngle)
        op_cut(sFilename, sOutFilename, lDegAngle, options.bCutAbove, fCutHeight=options.fCutHeight)
    # --------------------------------------
    elif op.startswith("eval_cut"):
        if op == "eval_cut_row":
            lsFilename = args[1:]
            traceln("- evaluating cut-based ROW partitions (fSimil=%s): " % options.lfSimil[0], lsFilename)
            if options.bIgnoreHeader: traceln("- ignoring headers")
            op_eval_row(lsFilename, options.lfSimil[0], options.bCutAbove, options.bVerbose
                        , bIgnoreHeader=options.bIgnoreHeader)
        elif op == "eval_cut_col":
            lsFilename = args[1:]
            traceln("- evaluating cut-based COLUMN partitions (fSimil=%s): " % options.lfSimil[0], lsFilename)
            op_eval_col(lsFilename, options.lfSimil[0], options.fRatio, options.bVerbose)
        elif op == "eval_cut_cell":
            sRowDir,sColDir = args[1:]
            traceln("- evaluating cut-based CELL partitions : " , sRowDir,sColDir)
            op_eval_cell(sRowDir, sColDir, options.fRatio, options.bCutAbove, options.bVerbose)
        elif op == "eval_cut_gt_recall":
            lsFilename = args[1:]
            traceln("- GT recall : %s" % lsFilename)
            lDegAngle = [float(s) for s in options.lsAngle.split(",")]
            traceln("- Allowed angles (°): %s" % lDegAngle)
            op_gt_recall(lsFilename, options.bCutAbove, lDegAngle, fCutHeight=options.fCutHeight)
        else:
            raise Exception("Unknown operation: %s"%op) 
    # --------------------------------------
    elif op.startswith("eval_direct"):
        lCrit, lDir = [], []
        if options.bIgnoreHeader: traceln("- ignoring headers")
        if op == "eval_direct_row":
            sRowDir = args[1]
            traceln("- evaluating ROW partitions (lfSimil=%s): " % options.lfSimil, sRowDir)
            lCrit, lDir = ["row"], [sRowDir]
        elif op == "eval_direct_col":
            sColDir = args[1]
            traceln("- evaluating COLUMN partitions (lfSimil=%s): " % options.lfSimil, sColDir)
            lCrit, lDir = ["col"], [sColDir]
        elif op == "eval_direct_cell":
            sRowDir,sColDir = args[1:3]
            lCrit, lDir = ["row", "col"], [sRowDir, sColDir]
            traceln("- evaluating CELL partitions (lfSimil=%s): " % options.lfSimil, lCrit, " ", [sRowDir, sColDir])
        else:
            raise Exception("Unknown operation: %s"%op) 
        if options.lfSimil:
            eval_direct(lCrit, lDir
                        , bIgnoreHeader=options.bIgnoreHeader
                        , bIgnoreOutOfTable=True
                        , lfSimil=options.lfSimil
                        )
        else:
            eval_direct(lCrit, lDir
                        , bIgnoreHeader=options.bIgnoreHeader
                        , bIgnoreOutOfTable=True
                        )
            
    elif op.startswith("eval_cluster"):
        lsRunDir = args[1:]
        sClusterLevel = {   "eval_cluster"      :"cell"
                         ,  "eval_cluster_cell" :"cell"
                         ,  "eval_cluster_col"  :"col"
                         ,  "eval_cluster_row"  :"row"
                        ,   "eval_cluster_region":"region"
                         }[op]
        traceln("- evaluating cluster partitions (lfSimil=%s): " % options.lfSimil, lsRunDir)
        if options.sGroupByAttr:
            traceln(" - merging run clusters having same value for their @%s attribute"%options.sGroupByAttr)
        if options.lfSimil:
            eval_cluster(lsRunDir, sClusterLevel
                        , bIgnoreHeader=options.bIgnoreHeader
                        , bIgnoreOutOfTable=True
                        , lfSimil=options.lfSimil
                        , sAlgo=options.sAlgo
                        , sGroupByAttr=options.sGroupByAttr
                        )
        else:
            eval_cluster(lsRunDir, sClusterLevel
                        , bIgnoreHeader=options.bIgnoreHeader
                        , bIgnoreOutOfTable=True
                        , sAlgo=options.sAlgo
                        , sGroupByAttr=options.sGroupByAttr
                        )
    elif op.startswith("oracle"):
        lsRunDir = args[1:]
        sClusterLevel = {   "oracle_cell" :"cell"
                         ,  "oracle_col"  :"col"
                         ,  "oracle_row"  :"row"
                         }[op]
        traceln("- evaluating cluster partitions (lfSimil=%s): " % options.lfSimil, lsRunDir)
        if options.lfSimil:
            eval_oracle(lsRunDir, sClusterLevel
                        , bIgnoreHeader=options.bIgnoreHeader
                        , bIgnoreOutOfTable=True
                        , lfSimil=options.lfSimil
                        )
        else:
            eval_oracle(lsRunDir, sClusterLevel
                        , bIgnoreHeader=options.bIgnoreHeader
                        , bIgnoreOutOfTable=True
                        )        
        
    
    # --------------------------------------
    else:
        traceln(usage)
        



