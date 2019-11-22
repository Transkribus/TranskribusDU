# -*- coding: utf-8 -*-

"""
    Table Undertsanding
    
    - given a human-annotated table
    - find lines that reflect well the cell borders, for rows and columns
    
    This is done by linear interpolation of cell borders, bi row and by column.
    
    Copyright Naver Labs Europe 201
    JL Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""





import sys, os, math
import collections

from lxml import etree
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize._hungarian import linear_sum_assignment

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from common.trace import traceln
from xml_formats.PageXml import MultiPageXml , PageXml
from util.Polygon import Polygon
from util.partitionEvaluation import evalPartitions
from util.jaccard import jaccard_distance
from tasks.DU_Table.DU_ABPTableSkewed_CutAnnotator import SkewedCutAnnotator,\
    get_row_partition, _isBaselineInTable, computePRF
# from tasks.DU_ABPTableCutAnnotator import get_col_partition, CutAnnotator
import tasks.DU_Table.DU_ABPTableCutAnnotator



class DocSeparatorException(Exception): 
    pass


def getDocSeparators(sFilename):
    """
    return two dictionaries
    row -> list of (x1, y1, x2, y2)
    col -> list of (x1, y1, x2, y2)
    """
    parser = etree.XMLParser()
    doc = etree.parse(sFilename, parser)
    root = doc.getroot()
    lCell= MultiPageXml.getChildByName(root,'TableCell')
    if not lCell:
        raise DocSeparatorException("No TableCell element in %s" %sFilename)
    dRowSep, dColSep = getCellsSeparators(lCell)
    del doc
    return dRowSep, dColSep

def getCellsSeparators(lCell):
    """
    return two dictionaries
    row -> ((x1, y1), (x2, y2))    NOTE: top of row
    col -> ((x1, y1), (x2, y2))    NOTE: left of column
    """
    dRowSep = {}
    dColSep = {}
    
    # let's collect the segments forming the cell borders, by row, by col
    dRowSep_lSgmt = collections.defaultdict(list)
    dColSep_lSgmt = collections.defaultdict(list)
    for cell in lCell:
        row, col, rowSpan, colSpan = [int(cell.get(sProp)) for sProp \
                                      in ["row", "col", "rowSpan", "colSpan"] ]
        coord = cell.xpath("./a:%s" % ("Coords"),namespaces={"a":MultiPageXml.NS_PAGE_XML})[0]
        sPoints = coord.get('points')
        plgn = Polygon.parsePoints(sPoints)
        try:
            lT, lR, lB, lL = plgn.partitionSegmentTopRightBottomLeft()
        except ZeroDivisionError:
            traceln("ERROR: cell %s row=%d col=%d has empty area and is IGNORED"
                    % (cell.get("id"), row, col))
            continue
        #now the top segments contribute to row separator of index: row
        dRowSep_lSgmt[row].extend(lT)
        #now the bottom segments contribute to row separator of index: row+rowSpan
        dRowSep_lSgmt[row+rowSpan].extend(lB)
        
        dColSep_lSgmt[col].extend(lL)
        dColSep_lSgmt[col+colSpan].extend(lR)
        
    #now make linear regression to draw relevant separators
    def getX(lSegment):
        lX = list()
        for x1,_y1,x2,_y2 in lSegment:
            lX.append(x1)
            lX.append(x2)
        return lX

    def getY(lSegment):
        lY = list()
        for _x1,y1,_x2,y2 in lSegment:
            lY.append(y1)
            lY.append(y2)
        return lY

    for row, lSegment in dRowSep_lSgmt.items():
        X = getX(lSegment)
        Y = getY(lSegment)
        #sum(l,())
        lfNorm = [math.sqrt(np.linalg.norm((x2 - x1, y2 - y1))) for x1,y1,x2,y2 in lSegment]
        #duplicate each element 
        sumW = sum(lfNorm) * 2
        W = [fN/sumW for fN in lfNorm for _ in (0,1)]
        # a * x + b
        a, b = np.polynomial.polynomial.polyfit(X, Y, 1, w=W)

        xmin, xmax = min(X), max(X)
        y1 = a + b * xmin
        y2 = a + b * xmax
        
        dRowSep[row] = ((xmin, y1), (xmax, y2))
    
    for col, lSegment in dColSep_lSgmt.items():
        X = getX(lSegment)
        Y = getY(lSegment)
        #sum(l,())
        lfNorm = [math.sqrt(np.linalg.norm((x2 - x1, y2 - y1))) for x1,y1,x2,y2 in lSegment]
        #duplicate each element 
        sumW = sum(lfNorm) * 2
        W = [fN/sumW for fN in lfNorm for _ in (0,1)]
        a, b = np.polynomial.polynomial.polyfit(Y, X, 1, w=W)

        ymin, ymax = min(Y), max(Y)
        x1 = a + b * ymin
        x2 = a + b * ymax 
        dColSep[col] = ((x1, ymin), (x2, ymax))
        
    return dRowSep, dColSep


def op_eval_cell(sRowDir, sColDir, fRatio, bCutAbove=True, bVerbose=False):
    """
    Takes output DU files from 2 folders
    - one giving rows
    - on giving columns
    from the same input file!!
    
    Compute the quality partitioning in cells
    
    Show results 
    return (nOk, nErr, nMiss)
    """
    #lsRowFn, lsColFn = [sorted([_fn for _fn in os.listdir(_sDir) if _fn.lower().endswith("pxml")]) for _sDir in (sRowDir, sColDir)]
    lsRowFn = sorted([_fn for _fn in os.listdir(sRowDir) if _fn.lower().endswith("_du.mpxml")])
    lsColFn = sorted([_fn for _fn in os.listdir(sColDir) if _fn.lower().endswith(".mpxml")])   
    
    # checking coherence !!
    lsRowBFn = [_fn[2:-9] for _fn in lsRowFn]               # 'a_0001_S_Aldersbach_008-01_0064_du.mpxml'
    lsColBFn = [_fn[4:-6] for _fn in lsColFn]   # 'cut-0001_S_Aldersbach_008-01_0064.mpxml'"
    #lsColBFn = [os.path.basename(_fn) for _fn in lsColFn]   # 'cut-0001_S_Aldersbach_008-01_0064.mpxml'"
    if lsRowBFn != lsColBFn:
        setRowBFn = set(lsRowBFn)
        setColBFn = set(lsColBFn)
        traceln("WARNING: different filenames in each folder")
        setOnlyCol = setColBFn.difference(setRowBFn)
        if setOnlyCol:
            traceln("--- Only in cols:", sorted(setOnlyCol), "\t")
            traceln("--- Only in cols: %d files" % len(setOnlyCol), "\n\t", )
        setOnlyRow = setRowBFn.difference(setColBFn)
        if setOnlyRow:
            traceln("ERROR: different filenames in each folder")
            traceln("--- Only in rows:", sorted(setOnlyRow), "\t")
            traceln("--- Only in rows: %d files" % len(setOnlyRow), "\n\t", )
            sys.exit(1)
            
        if setOnlyCol:
            # ok, let's clean the col list... :-/
            lsColFn2 = []
            ibfn = 0
            for fn in lsColFn:
                if fn[4:-6] == lsRowBFn[ibfn]:
                    lsColFn2.append(fn)
                    ibfn += 1
            lsColFn = lsColFn2
            lsColBFn = [_fn[4:-6] for _fn in lsColFn]
            assert lsRowBFn == lsColBFn
            traceln("Reconciliated file lists...  %d files now"%len(lsRowBFn))
        del setOnlyCol
 
#     lfSimil = [ i / 100 for i in range(70, 101, 10)]
    lfSimil = [ i / 100.0 for i in [66, 80, 100]]
    
    dOkErrMissOnlyRow  = { fSimil:(0,0,0) for fSimil in lfSimil }
    dOkErrMissOnlyRow.update({'name':'OnlyCell'
                            , 'FilterFun':_isBaselineInTable})

    dOkErrMiss = dOkErrMissOnlyRow
    
    def cross_row_col(lsetRow, lsetCol):
        lsetCell = []
        for setRow in lsetRow:
            for setCol in lsetCol:
                setCell = setRow.intersection(setCol)
                lsetCell.append(setCell)
        return lsetCell
    
    def evalHungarian(lX,lY,th):
            """
            """
    
            cost_matrix=np.zeros((len(lX),len(lY)),dtype=float)
    
            for a,x in enumerate(lX):
                for b,y in enumerate(lY):
                    #print(x,y,    jaccard_distance(x,y))
                    cost_matrix[a,b]= jaccard_distance(x,y)
    
            r1,r2 = linear_sum_assignment(cost_matrix)
            ltobeDel=[]
            for a,i in enumerate(r2):
                # print (r1[a],ri)
                if 1 - cost_matrix[r1[a],i] < th :
                    ltobeDel.append(a)
                    # if bt:print(lX[r1[a]],lY[i],1- cost_matrix[r1[a],i])
                    # else: print(lX[i],lY[r1[a]],1-cost_matrix[r1[a],i])
                # else:
                    # if bt:print(lX[r1[a]],lY[i],1-cost_matrix[r1[a],i])
                    # else:print(lX[i],lY[r1[a]],1-cost_matrix[r1[a],i])
            r2 = np.delete(r2,ltobeDel)
            r1 = np.delete(r1,ltobeDel)
            # print("wwww",len(lX),len(lY),len(r1),len(r2))
    
            return len(r1), len(lX)-len(r1), len(lY)-len(r1)
    
    pnum = 1  # multi-page not supported 
    
    for n, (sBasefilename, sFilename_row, sFilename_col) in enumerate(zip(lsRowBFn
                                                           , lsRowFn, lsColFn)):
        assert sBasefilename in sFilename_row
        assert sBasefilename in sFilename_col
        dNS = {"pc":PageXml.NS_PAGE_XML}
        if bVerbose: traceln("-"*30)
        # Rows...
        sxpCut = './/pc:CutSeparator[@orient="0" and @DU_type="S"]' #how to find the cuts
        doer = SkewedCutAnnotator(bCutAbove)
        #traceln(" - Cut selector = ", sxpCut)
        
        lsetGT_row, llsetRun_row = get_row_partition(doer, sxpCut, dNS
                                             , os.path.join(sRowDir, sFilename_row)
                                             , [_isBaselineInTable]
                                             , bCutAbove=True, bVerbose=False
                                             , funIndex=lambda o: o._dom_id
                                             )
        #traceln("%d rows in GT"%len(lsetGT_row))
        
        [lsetRun_row] = llsetRun_row  # per page x filter function
        if bVerbose:
            for fSimil in lfSimil:
                _nOk, _nErr, _nMiss, _lFound, _lErr, _lMissed = evalPartitions(lsetRun_row, lsetGT_row, fSimil, jaccard_distance)
                _fP, _fR, _fF = computePRF(_nOk, _nErr, _nMiss)
                traceln("%4d %8s simil:%.2f  P %5.1f  R %5.1f  F1 %5.1f   ok=%6d  err=%6d  miss=%6d" %(
                      n+1, "row", fSimil
                    , _fP, _fR, _fF
                    , _nOk, _nErr, _nMiss))            
        
        # Columns...
        sxpCut = './/pc:CutSeparator[@orient="90"]' #how to find the cuts
        doer = tasks.DU_ABPTableCutAnnotator.CutAnnotator()
        #traceln(" - Cut selector = ", sxpCut)
        
        # load objects: Baseline and Cuts
        lsetGT_col, _lsetDataGT_col, llsetRun_col = tasks.DU_ABPTableCutAnnotator.get_col_partition(doer, sxpCut, dNS
                          , os.path.join(sColDir, sFilename_col)
                          , [_isBaselineInTable]
                          , fRatio
                          , bVerbose=False
                          , funIndex=lambda x: x._dom_id
                          )
        lsetGT_col = [set(_o) for _o in lsetGT_col]  # make it a list of set
        [lsetRun_col] = llsetRun_col  # per page x filter function
        lsetRun_col = [set(_o) for _o in lsetRun_col]  # make it a list of set
        if bVerbose:
            for fSimil in lfSimil:
                _nOk, _nErr, _nMiss, _lFound, _lErr, _lMissed = evalPartitions(lsetRun_col, lsetGT_col, fSimil, jaccard_distance)
                _fP, _fR, _fF = computePRF(_nOk, _nErr, _nMiss)
                traceln("%4d %8s simil:%.2f  P %5.1f  R %5.1f  F1 %5.1f   ok=%6d  err=%6d  miss=%6d" %(
                      n+1, "col", fSimil
                    , _fP, _fR, _fF
                    , _nOk, _nErr, _nMiss))            
        
        lsetGT_cell  = cross_row_col(lsetGT_row , lsetGT_col)
        lsetRun_cell = cross_row_col(lsetRun_row, lsetRun_col)
        
        #traceln("%d %d"%(len(lsetGT_cell), len(lsetRun_cell)))
        #lsetGT_cell  = [_s for _s in lsetGT_cell if _s]
        lsetRun_cell = [_s for _s in lsetRun_cell if _s]
        #traceln("%d %d"%(len(lsetGT_cell), len(lsetRun_cell)))
        
        
        # FIX
        lsetGT_cell = list()
        dNsSp = {"pc":"http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
        _parser = etree.XMLParser()
        _doc = etree.parse(os.path.join(sColDir, sFilename_col), _parser)
        for ndCell in _doc.getroot().xpath('//pc:TableCell', namespaces=dNsSp):
            setCell = set(_nd.get("id") for _nd in ndCell.xpath('.//pc:TextLine', namespaces=dNsSp))
            if setCell: lsetGT_cell.append(setCell)
        # traceln("%d non-empty cells in GT" % len(lsetGT_cell))
                
        for fSimil in lfSimil:
            nOk, nErr, nMiss = dOkErrMiss[fSimil]
            _nOk, _nErr, _nMiss = evalHungarian(lsetRun_cell, lsetGT_cell, fSimil)
            _fP, _fR, _fF = computePRF(_nOk, _nErr, _nMiss)
#             if bVerbose:
#                 traceln(" - - - simil = %.2f" % fSimil)
#                 traceln("----- RUN ----- ")
#                 for s in lsetRun_cell: traceln("  run ", sorted(s))
#                 traceln("----- REF ----- ")
#                 for s in lsetGT_cell: traceln("  ref ", sorted(s))
            nOk   += _nOk
            nErr  += _nErr
            nMiss += _nMiss
            traceln("%4d %8s simil:%.2f  P %5.1f  R %5.1f  F1 %5.1f   ok=%6d  err=%6d  miss=%6d  %s page=%d" %(
                          n+1, dOkErrMiss['name'], fSimil
                        , _fP, _fR, _fF
                        , _nOk, _nErr, _nMiss
                        , sBasefilename, pnum))
            dOkErrMiss[fSimil] = (nOk, nErr, nMiss)
    
#     for dOkErrMiss in [dOkErrMissOnlyRow, dOkErrMissTableRow]:
    
    traceln()
    name = dOkErrMiss['name']
    for fSimil in lfSimil:
        nOk, nErr, nMiss = dOkErrMiss[fSimil]
        fP, fR, fF = computePRF(nOk, nErr, nMiss)
        traceln("ALL %8s  simil:%.2f  P %5.1f  R %5.1f  F1 %5.1f " % (name, fSimil, fP, fR, fF )
                , "        "
                ,"ok=%d  err=%d  miss=%d" %(nOk, nErr, nMiss))
        
    return (nOk, nErr, nMiss)


# ------------------------------------------------------------------
if __name__ == "__main__":
    
    # list the input files
    lsFile = []
    for path in sys.argv:
        if os.path.isfile(path):
            lsFile.append(path)
        elif os.path.isdir(path):
            lsFilename = [os.path.join(path, "col", s) for s in os.listdir(os.path.join(path, "col")) if s.endswith(".mpxml") ]
            if not lsFilename:
                lsFilename = [os.path.join(path, "col", s) for s in os.listdir(os.path.join(path, "col")) if s.endswith(".pxml") and s[-7] in "0123456789"]
            lsFilename.sort()
            traceln(" folder %s --> %d files" % (path, len(lsFilename)))
            lsFile.extend(lsFilename)
    traceln("%d files to read" % len(lsFile))
             
    traceln(lsFilename)
    traceln("%d files to be processed" % len(lsFilename))

    # load the separators (estimated by linear regression)
    ldRowSep, ldColSep = [], []
    for sFilename in lsFilename:
        try:
            dRowSep, dColSep = getDocSeparators(sFilename)
        except DocSeparatorException as e:
            traceln("\t SKIPPING this file: " + str(e))
        ldRowSep.append(dRowSep)   
        ldColSep.append(dColSep)

    # Now look at the distribution of the separators' angles
    # horizontally, and vertically
    fig = plt.figure(1)
    fig.canvas.set_window_title("Distribution of separator angles (Obtained by linear regression)")
    for i, (bHorizontal, degavg, degmax, ldXYXY) in enumerate([ 
          (True ,  0, 5, ldRowSep)
        , (False, 90, 5, ldColSep)
        ]):
        C = collections.Counter()

        for dXYXY in ldXYXY:
            lAngle = []
            for (x1,y1),(x2,y2) in dXYXY.values():
                if bHorizontal:
                    angle = math.degrees(math.atan((y2-y1) / (x2-x1)))
                else:
                    angle = 90 - math.degrees(math.atan((x2-x1) / (y2 - y1)))
                angle = round(angle, 1)
                lAngle.append(angle)
            C.update(lAngle)

        plt.subplot(211+i)        
        ltV = list(C.items())
        ltV.sort()
        traceln(ltV)
        lX = [tV[0] for tV in ltV if abs(tV[0]-degavg) <= degmax]
        lY = [tV[1] for tV in ltV if abs(tV[0]-degavg) <= degmax]
        if len(lX) < len(ltV):
            traceln("WARNING: excluded %d bins (%d values in total) outside of [%.1f, %.1f]" 
                    % (len(ltV)-len(lX)
                       , sum((tV[1] for tV in ltV if abs(tV[0]-degavg) > degmax))
                       , degavg-degmax, degavg+degmax))
        #plt.plot(lX, lY)
        plt.scatter(lX, lY)
        plt.ylabel("Count")
        plt.grid(which='both', axis='both')
        plt.xticks(lX)
    plt.xlabel("Degrees")
    plt.show()
    
