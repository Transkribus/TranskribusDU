# -*- coding: utf-8 -*-

"""
    Find cuts of a page along different slopes 
    and annotate them based on the table row content (which defines a partition)
    
    Copyright Naver Labs Europe 2018
    JL Meunier

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
from optparse import OptionParser
from lxml import etree
from collections import defaultdict

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from common.trace import traceln
from xml_formats.PageXml import PageXml
from tasks.DU_ABPTableSkewed_CutAnnotator import op_cut, op_eval_row, op_gt_recall
from tasks.DU_ABPTableCutAnnotator import CutAnnotator, op_eval_col
from tasks.DU_Table_CellBorder import op_eval_cell
from util.partitionEvaluation import evalPartitions
from util.jaccard import jaccard_distance


def listFiles(sDir):
    """
    return 1 list of files
    """
    lsFile = sorted([_fn 
                     for _fn in os.listdir(sDir) 
                     if _fn.lower().endswith("_du.mpxml") or _fn.lower().endswith("_du.pxml")
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
                if bIgnoreHeader and nd.get("DU_header") != "D": continue
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
            _nOk, _nErr, _nMiss, _lFound, _lErr, _lMissed = evalPartitions(
                  list(dValue_lId.values())
                , list(dValue_lId_GT.values())
                , fSimil
                , jaccard_distance)
            
            _fP, _fR, _fF = computePRF(_nOk, _nErr, _nMiss)
            
            traceln("simil:%.2f  P %5.1f  R %5.1f  F1 %5.1f   ok=%6d  err=%6d  miss=%6d  %s" %(
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
        traceln("ALL simil:%.2f  P %5.1f  R %5.1f  F1 %5.1f " % (fSimil, fP, fR, fF )
                , "        "
                ,"ok=%d  err=%d  miss=%d" %(nOk, nErr, nMiss))
        
    return (nOk, nErr, nMiss)
    

def eval_cluster(lsRunDir
                , bIgnoreHeader=False
                , bIgnoreOutOfTable=True
                , lfSimil=[i / 100.0 for i in [66, 80, 100]]
                , xpSelector=".//pc:TextLine"):
    """
    evaluate the cluster quality from a run folder
    
    We assume to have the groundtruth row and col in the files as well as the predicted clusters
    """
    assert lsRunDir
    dOkErrMiss = { fSimil:(0,0,0) for fSimil in lfSimil }

    for sRunDir in lsRunDir:
        lsFile = listFiles(sRunDir)
        traceln("-loaded %d files from %s" % (len(lsFile), sRunDir))

        for sFilename in lsFile:
            # cluster  -> [node_id]
            dGT  = defaultdict(list)
            dRun = defaultdict(list)
            
            doc = etree.parse(os.path.join(sRunDir, sFilename))
            rootNd = doc.getroot()
            assert len(PageXml.xpath(rootNd, "//pc:Page")) == 1, "NOT YET IMPLEMENTED: eval on multi-page files"
             
            for nd in PageXml.xpath(rootNd, xpSelector):
                if bIgnoreHeader and nd.get("DU_header") != "D": continue
                
                ndparent = nd.getparent() 
                ndid   = nd.get("id")

                val_gt = "%s__%s" % (ndparent.get("row"), ndparent.get("col"))
                if val_gt == 'None__None' and bIgnoreOutOfTable: 
                    continue

                dGT[val_gt].append(ndid)
 
                val_run = nd.get("DU_cluster")
                dRun[val_run].append(ndid)
                assert ndparent.tag.endswith("TableCell"), "expected TableCell got %s" % nd.getparent().tag
                
            for fSimil in lfSimil:
                _nOk, _nErr, _nMiss, _lFound, _lErr, _lMissed = evalPartitions(
                      list(dRun.values())
                    , list(dGT.values())
                    , fSimil
                    , jaccard_distance)
                
                _fP, _fR, _fF = computePRF(_nOk, _nErr, _nMiss)
                
                traceln("simil:%.2f  P %5.1f  R %5.1f  F1 %5.1f   ok=%6d  err=%6d  miss=%6d  %s" %(
                      fSimil
                    , _fP, _fR, _fF
                    , _nOk, _nErr, _nMiss
                    , os.path.basename(sFilename)))
                sFilename = "" # ;-)
                
                # global count
                nOk, nErr, nMiss = dOkErrMiss[fSimil]
                nOk   += _nOk
                nErr  += _nErr
                nMiss += _nMiss
                dOkErrMiss[fSimil] = (nOk, nErr, nMiss)
                
            traceln()
        
    for fSimil in lfSimil:
        nOk, nErr, nMiss = dOkErrMiss[fSimil]
        fP, fR, fF = computePRF(nOk, nErr, nMiss)
        traceln("ALL simil:%.2f  P %5.1f  R %5.1f  F1 %5.1f " % (fSimil, fP, fR, fF )
                , "        "
                ,"ok=%d  err=%d  miss=%d" %(nOk, nErr, nMiss))
        
    return (nOk, nErr, nMiss)


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

    eval_cluster      RUN_DIR+     eval the quality of the cluster using the GT (all in same file)
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

    parser.add_option("--ignore-header", dest='bIgnoreHeader',  action="store_true", default=False
                        , help="eval_row: ignore header text (and ignore empty cells, so ignore header cells!)")

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
        traceln("- evaluating cluster partitions (lfSimil=%s): " % options.lfSimil, lsRunDir)
        if options.lfSimil:
            eval_cluster(lsRunDir
                        , bIgnoreHeader=options.bIgnoreHeader
                        , bIgnoreOutOfTable=True
                        , lfSimil=options.lfSimil
                        )
        else:
            eval_cluster(lsRunDir
                        , bIgnoreHeader=options.bIgnoreHeader
                        , bIgnoreOutOfTable=True
                        )
        
        
    
    # --------------------------------------
    else:
        traceln(usage)
        



