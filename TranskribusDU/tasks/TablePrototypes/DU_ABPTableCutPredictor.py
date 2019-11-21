# -*- coding: utf-8 -*-

"""
    Find cuts of a page and generates BIESO annotation
    
    Do BIESO in each cell, taking into account x-overlap
    
    Compute usual metrics is MPXML is annotated
    
    
It looks for horizontal or vertical cuts, tolerating to cut accross some small object along its path (hyperparameter: currently 5%, i.e. object length(s) must be (cumulatively) less than 5% of the page width or height to be ignored)

Then, the blocks in each cell are BIESO-labelled by sorting them vertically. To get better results, the sorting is done considering only blocks having an horizontal overlap. So if several columns were merged (under segmentation), the BIESO labels in each (merged) cell generally good.

On the 45 test file of fold_1, it gets 82 in accuracy while the CRF-based method reported in DAS does 93.

Tuning the hyperparameter is not doable because as usually, 1 value does not fit all. The 0.05 value is already produced by careful (manual) tuning.

We could do a bit better for the O but O represent 3% of the blocks.

####################  CUT RESULTS ####################

--- Wed Apr 11 08:43:36 2018---------------------------------
TEST REPORT FOR: HCutter

  Line=True class, column=Prediction
B  [[2269  100   16  287  117]
I   [  91  987  154   28   21]
E   [  13   92 2256  371   57]
S   [  38   18   55 2113  160]
O   [   6    1    3  112  195]]

(unweighted) Accuracy score = 0.82     trace=7820 sum=9560

             precision    recall  f1-score   support

          B      0.939     0.814     0.872      2789
          I      0.824     0.770     0.796      1281
          E      0.908     0.809     0.856      2789
          S      0.726     0.886     0.798      2384
          O      0.355     0.615     0.450       317

avg / total      0.842     0.818     0.825      9560

####################   CRF ####################

--- Mon Mar 26 13:03:40 2018---------------------------------
TEST REPORT FOR: CV4_row_UW_fold_1

  Line=True class, column=Prediction
row_B  [[2621   48   10   94   16]
row_I   [  52 1164   60    5    0]
row_E   [  10   18 2677   81    3]
row_S   [  27    2  121 2199   35]
row_O   [  21   11   13   17  255]]

(unweighted) Accuracy score = 0.93     trace=8916 sum=9560

             precision    recall  f1-score   support

      row_B      0.960     0.940     0.950      2789
      row_I      0.936     0.909     0.922      1281
      row_E      0.929     0.960     0.944      2789
      row_S      0.918     0.922     0.920      2384
      row_O      0.825     0.804     0.815       317

avg / total      0.933     0.933     0.933      9560
    
    
    Copyright Naver Labs Europe 2018
    JL Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""




import sys, os
from lxml import etree

import numpy as np
import collections

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from xml_formats.PageXml import MultiPageXml 
from util.Polygon import Polygon

from common.TestReport import TestReport, TestReportConfusion

from tasks.DU_ABPTableCutAnnotator import CutAnnotator as CutAnnotator

class CutPredictor(CutAnnotator):
    """
    Cutting the page horizontally
    """
    lsClassName = ['B', 'I', 'E', 'S', 'O']
    dLblIndex = {s:i for i,s in enumerate(['B', 'I', 'E', 'S', 'O'])}
    n = 5 
    
    def __init__(self, lsClassName, fMinHorizProjection=0.05, fMinVertiProjection=0.05):
        self.fMinHorizProjection=fMinHorizProjection
        self.fMinVertiProjection=fMinVertiProjection
        self.parser = etree.XMLParser(remove_blank_text=True)


    def predict(self, sFilename):
        """
        
        return Y_pred, YGT   (shape=(nb_node,) dtype=np.int)
        """         
        
        #for the pretty printer to format better...
        assert os.path.isfile(sFilename), sFilename
        doc  = etree.parse(sFilename, self.parser)
        #doc  = etree.parse(sFilename)
        root = doc.getroot()
        
        # Find cuts
        lY, lX = self.add_cut_to_DOM(root,
                            fMinHorizProjection=self.fMinHorizProjection,
                            fMinVertiProjection=self.fMinVertiProjection)
        
        # ################################################################
        # NOTE : we will assumes first and last row/column contain Other
        # ################################################################
        
        lyy = list(zip(lY[:-1], lY[1:])) # list of intervals
        lxx = list(zip(lX[:-1], lX[1:])) # list of intervals
        
        dTable = collections.defaultdict(lambda : collections.defaultdict(list) )
        # dTable[i][j] --> list of TExLine in that cell
        
        def getTableIndex(v, lvv):
            """
            index in the table row or columns. The 1st border is at 0
            """
            for i, (v1, v2) in enumerate(lvv):
                if v1 <= v and v <= v2:
                    return i+1
            if v < lvv[0][0]:
                return 0
            else:
                return len(lvv)+1
        
        
        #place each TextLine in the table rows and columns
        ndPage = MultiPageXml.getChildByName(root, 'Page')[0]
        w, h = int(ndPage.get("imageWidth")), int(ndPage.get("imageHeight"))
        
        lndTexLine = MultiPageXml.getChildByName(ndPage, 'TextLine') 
        
        imax, jmax = -1, -1
        for nd in lndTexLine:
            sPoints=MultiPageXml.getChildByName(nd,'Coords')[0].get('points')
            #x1,y1,x2,y2 = Polygon.parsePoints(sPoints).fitRectangle()
            x1, y1, x2, y2 = Polygon.parsePoints(sPoints).fitRectangle()
            
            i = getTableIndex((y1+y2)/2.0, lyy)
            j = getTableIndex((x1+x2)/2.0, lxx)
            
            dTable[i][j].append( (y1, y2, x1, x2, nd) )
            imax = max(i, imax)
            jmax = max(j, jmax)

        def getGT(nd):
            try:
                return CutPredictor.dLblIndex[nd.get('DU_row')]
            except:
                return 0

        def overlap(a, b):
            _y1, _y2, ax1, ax2, _nd = a
            _y1, _y2, bx1, bx2, _nd = b
            return min(ax2, bx2) - max(ax1, bx1)
        
        def label(lt):
            """
            set the attribute cutDU_row to each node for BIESO labelling
            """
            lt.sort()
            
            #the 'O'
            newlt = []
            for (y1, y2, x1, x2, nd) in lt:
                bInMargin = x2 < 350 or x1 > 4600
                
                if bInMargin:
                    nd.set('cutDU_row',  'O')
                else:
                    newlt.append((y1, y2, x1, x2, nd))
            
            for i, t in enumerate(newlt):
                (y1, y2, x1, x2, nd) = t
                
                #is there someone above?? if yes get the closest node above
                nd_just_above = None
                for j in range(i-1, -1, -1):
                    tt = newlt[j]
                    if overlap(tt, t) > 0:
                        nd_just_above = tt[4]
                        if nd_just_above.get('cutDU_row') != 'O': break
                
                if nd_just_above is None:
                    #S by default
                    nd.set('cutDU_row',  'S')
                else:
                    if nd_just_above.get('cutDU_row') == 'S':
                        nd_just_above.set('cutDU_row', 'B')
                        nd           .set('cutDU_row', 'E')
                    elif nd_just_above.get('cutDU_row') == 'E':
                        nd_just_above.set('cutDU_row', 'I')
                        nd           .set('cutDU_row', 'E')
                    elif nd_just_above.get('cutDU_row') == 'I':
                        nd.set('cutDU_row',  'E')
                    elif nd_just_above.get('cutDU_row') == 'B':
                        #bad luck, we do not see the intermediary node
                        nd           .set('cutDU_row', 'E')
                    elif nd_just_above.get('cutDU_row') == 'O':
                        raise Exception('Internal error')
                
        #now set the BIESO labels... (only vertically for now)
        liGTLabel = list()
        liLabel = list()
        for i in range(0, imax+1):
            dRow = dTable[i]
            for j in range(0, jmax+1):
                lt = dRow[j]
                if not lt: continue
                
                label(lt)

                liGTLabel.extend([getGT(nd) for y1, y2, x1, x2, nd in lt])
                liLabel.extend([self.dLblIndex[nd.get('cutDU_row')] for y1, y2, x1, x2, nd in lt])

        Y_pred = np.array(liLabel  , dtype=np.int)
        YGT    = np.array(liGTLabel, dtype=np.int)
                     
        sOutFilename = sFilename[:-6] + "_cut.mpxml"
        doc.write(sOutFilename, encoding='utf-8',pretty_print=True,xml_declaration=True)
        print('Annotated cut separators in %s'%sOutFilename)   
        
        del doc
        
        return Y_pred, YGT
                     
# ------------------------------------------------------------------
def main(lsFilename, fMinHorizProjection=0.05, fMinVertiProjection=0.05):


    doer = CutPredictor(fMinHorizProjection, fMinVertiProjection)
    
    l_Y_pred, l_YGT = [], []
    
    for sFilename in lsFilename:
        print(" - processing ", sFilename)
        Y_pred, YGT = doer.predict(sFilename)
        l_Y_pred.append(Y_pred)
        l_YGT.append(YGT)
        
    oTstRpt = TestReport("HCutter", l_Y_pred, l_YGT, CutPredictor.lsClassName, lsFilename)

    return oTstRpt
    
# ------------------------------------------------------------------
if __name__ == "__main__":
    #load mpxml 
    sInput = sys.argv[1]
    if os.path.isdir(sInput):
        lsFilename = [os.path.join(sInput, "col", s) for s in os.listdir(os.path.join(sInput, "col")) if s.endswith(".mpxml") and s[-7] in "0123456789"]
        lsFilename.sort()
        print(lsFilename)
    else:
        lsFilename = [sInput]

    try:
        fMinH = float(sys.argv[2])
    except:
        fMinH = None
        fMinV = None
        
    if not fMinH is None:
        fMinV = float(sys.argv[4])  # specify none or both

    if fMinV == None:
        oTstRpt = main(lsFilename)
    else:
        oTstRpt = main(lsFilename, fMinH, fMinV)

    print(oTstRpt.toString(False, False))
    
    
    


