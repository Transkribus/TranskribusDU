# -*- coding: utf-8 -*-
"""


    Build Rows for a BIESO model

     H. Déjean
    

    copyright Xerox 2017, Naver 2017, 2018
    READ project 

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
from __future__ import absolute_import
from __future__ import  print_function
from __future__ import unicode_literals

import sys, os.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))

import collections
from lxml import etree
from sklearn.metrics import  adjusted_rand_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score
import common.Component as Component
from common.trace import traceln
import config.ds_xml_def as ds_xml
from ObjectModel.xmlDSDocumentClass import XMLDSDocument
from ObjectModel.XMLDSTEXTClass  import XMLDSTEXTClass
from ObjectModel.XMLDSTABLEClass import XMLDSTABLEClass
from ObjectModel.XMLDSCELLClass import XMLDSTABLECELLClass
from ObjectModel.XMLDSTableRowClass import XMLDSTABLEROWClass
from ObjectModel.XMLDSTableColumnClass import XMLDSTABLECOLUMNClass
from spm.spmTableRow import tableRowMiner
from xml_formats.Page2DS import primaAnalysis
from util.partitionEvaluation import evalPartitions, jaccard, iuo
from util.geoTools import sPoints2tuplePoints
from shapely.geometry import Polygon
from shapely import affinity
from shapely.ops import cascaded_union


BTAG='S'
STAG= 'SS'
class RowDetection(Component.Component):
    """
        row detection
        @precondition: column detection done, BIES tagging done for text elements
                
        11/9/2018: last idea: suppose the cell segmentation good enough:  group cells which are unambiguous 
        with the cell in the (none empty) next column .     
            12/11/2018: already done in mergehorinzontalCells !!
        12/11/2018:  assume perfect cells: build simple: take next lright as same row
                      then look for elements belonging  to several rows
    """
    usage = "" 
    version = "v.1.0"
    description = "description: rowDetection"

    #--- INIT -------------------------------------------------------------------------------------------------------------    
    def __init__(self):
        """
        Always call first the Component constructor.
        """
        Component.Component.__init__(self, "RowDetection", self.usage, self.version, self.description) 
        
        self.colname = None
        self.docid= None

        self.do2DS= False
        
        self.THHighSupport = 0.20
        self.bYCut = False
        self.bCellOnly = False
        # for --test
        self.bCreateRef = False
        self.bCreateRefCluster = False
        
        self.bNoTable = False
        self.bEvalCluster=False
        self.evalData = None
        
    def setParams(self, dParams):
        """
        Always call first the Component setParams
        Here, we set our internal attribute according to a possibly specified value (otherwise it stays at its default value)
        """
        Component.Component.setParams(self, dParams)
#         if dParams.has_key("coldir"): 
#             self.colname = dParams["coldir"]
        if "docid" in dParams:         
            self.docid = dParams["docid"]
        if "dsconv" in dParams:         
            self.do2DS = dParams["dsconv"]
                        
        if "createref" in dParams:         
            self.bCreateRef = dParams["createref"]


        if "bNoColumn" in dParams:         
            self.bNoTable = dParams["bNoColumn"]
            
        if "createrefCluster" in dParams:         
            self.bCreateRefCluster = dParams["createrefCluster"]

        if "evalCluster" in dParams:         
            self.bEvalCluster = dParams["evalCluster"]
                        
        if "thhighsupport" in dParams:
            self.THHighSupport = dParams["thhighsupport"] * 0.01
         
        if 'YCut' in dParams: self.bYCut =  dParams["YCut"]
        if 'bCellOnly' in dParams: self.bCellOnly =  dParams["bCellOnly"]
          
    def createCells(self, table):
        """
            create new cells using BIESO tags
            @input: tableObeject with old cells
            @return: tableObject with BIES cells
            @precondition: requires columns
            
            if DU_col = M : ignore
            
        """
#         print ('nbcells:',len(table.getAllNamedObjects(XMLDSTABLECELLClass)))
        table._lObjects = []
        lSkipped =[]
        for col in table.getColumns():
#             print (col)

            lNewCells=[]
            # keep original positions
            try:col.resizeMe(XMLDSTABLECELLClass)
            except: pass
            # in order to ignore existing cells from GT: collect all objects from cells
            lObjects = [txt for cell in col.getCells() for txt in cell.getObjects() ]
            lObjects.sort(key=lambda x:x.getY())
            
            curChunk=[]
            lChunks = []
            for txt in lObjects:
                if  txt.getAttribute("DU_col") == 'Mx':
                    lSkipped.append(txt)
                elif txt.getAttribute("DU_row") == STAG:
                    if curChunk != []:
                        lChunks.append(curChunk)
                        curChunk=[]
                    lChunks.append([txt])
                elif txt.getAttribute("DU_row") in ['I', 'E']:
                    curChunk.append(txt)
                elif txt.getAttribute("DU_row") == BTAG:
                    if curChunk != []:
                        lChunks.append(curChunk)
                    curChunk=[txt]
                elif txt.getAttribute("DU_row") == 'O':
                    ## add Other as well??? no
                    curChunk.append(txt)
#                     pass
                        
            if curChunk != []:
                lChunks.append(curChunk)
                
            if lChunks != []:
                # create new cells
#                 table.delCell(cell)
                irow= txt.getParent().getIndex()[0]
                for i,c in enumerate(lChunks):
                    #create a new cell per chunk and replace 'cell'
                    newCell = XMLDSTABLECELLClass()
                    newCell.setPage(txt.getParent().getPage())
                    newCell.setParent(table)
                    newCell.setName(ds_xml.sCELL)
#                     newCell.setIndex(irow+i,txt.getParent().getIndex()[1])
                    newCell.setIndex(i,txt.getParent().getIndex()[1])
                    newCell.setObjectsList(c)
#                     newCell.addAttribute('type','new')
                    newCell.resizeMe(XMLDSTEXTClass)
                    newCell.tagMe2()
                    for o in newCell.getObjects():
                        o.setParent(newCell)
                        o.tagMe()
#                     contour = self.createContourFromListOfElements(newCell.getObjects())
#                     if contour is not None:
# #                         newCell.addAttribute('points',','.join("%s,%s"%(x[0],x[1]) for x in contour.lXY))
#                         newCell.addAttribute('points',','.join("%s,%s"%(x[0],x[1]) for x in contour))
#                     newCell.tagMe2()
                        
#                         table.addCell(newCell)
                    lNewCells.append(newCell)
#                 if txt.getParent().getNode().getparent() is not None: txt.getParent().getNode().getparent().remove(txt.getParent().getNode())
#                 del(txt.getParent())
            #delete all cells
            
            for cell in col.getCells():
#                 print (cell)
                try: 
                    if cell.getNode().getparent() is not None: cell.getNode().getparent().remove(cell.getNode())
                except: pass
            
            [table.delCell(cell) for cell in col.getCells() ]
#             print ('\t nbcells 2:',len(table.getAllNamedObjects(XMLDSTABLECELLClass)))
            col._lcells= []
            col._lObjects=[]
#             print (col.getAllNamedObjects(XMLDSTABLECELLClass))
            [table.addCell(c) for c in lNewCells]        
            [col.addCell(c) for c in lNewCells]
#             print ('\t nbcells 3:',len(table.getAllNamedObjects(XMLDSTABLECELLClass)))

#         print ('\tnbcells:',len(table.getAllNamedObjects(XMLDSTABLECELLClass)))

            
    
    def matchCells(self,table):
        """
            use lcs (dtw?) for matching
                dtw: detect merging situation
            for each col: match with next col
            
            series 1 : col1 set of cells
            series 2 : col2 set of cells
            distance = Yoverlap
               
        """
        dBest = {}
        #in(self.y2, tb.y2) - max(self.y1, tb.y1)
        def distY(c1,c2): 
            o  = min(c1.getY2() , c2.getY2()) - max(c1.getY() , c2.getY())
            if o < 0:
                return 1
#             d = (2* (min(c1.getY2() , c2.getY2()) - max(c1.getY() , c2.getY()))) / (c1.getHeight() + c2.getHeight())
#             print(c1,c1.getY(),c1.getY2(), c2,c2.getY(),c2.getY2(),o,d) 
            return 1 - (1 * (min(c1.getY2() , c2.getY2()) - max(c1.getY() , c2.getY()))) / min(c1.getHeight() , c2.getHeight())
        
        laErr=[]
        for icol, col in enumerate(table.getColumns()):
            lc = col.getCells() + laErr
            lc.sort(key=lambda x:x.getY())
            if icol+1 < table.getNbColumns():
                col2 = table.getColumns()[icol+1]
                if col2.getCells() != []:
                    cntOk,cntErr,cntMissed, lFound,lErr,lMissed = evalPartitions(lc,col2.getCells(), .25,distY)
                    [laErr.append(x) for x in lErr if x not in laErr]
                    [laErr.remove(x) for x,y in lFound if x in laErr]
                    # lErr: cell not matched in col1
                    # lMissed: cell not matched in col2
                    print (col,col2,cntOk,cntErr,cntMissed,lErr) #, lFound,lErr,lMissed)
                    for x,y in lFound:
                        dBest[x]=y
                else:
                    [laErr.append(x) for x in lc if x not in laErr]
        # create row
        #sort keys by x
        skeys = sorted(dBest.keys(),key=lambda x:x.getX())
        lcovered=[]
        llR=[]
        for key in skeys:
#             print (key,lcovered)
            if key not in lcovered:
                lcovered.append(key)
                nextC = dBest[key]
#                 print ("\t",key,nextC,lcovered)
                lrow = [key]
                while nextC:
                    lrow.append(nextC)
                    lcovered.append(nextC)
                    try:
                        nextC=dBest[nextC]
                    except KeyError:
                        print ('\txx\t',lrow)
                        llR.append(lrow)
                        nextC=None
            
        for lrow in llR:
            contour = self.createContourFromListOfElements(lrow)
            if contour is not None:
                spoints = ','.join("%s,%s"%(x[0],x[1]) for x in contour)
                r = XMLDSTABLEROWClass(1)
                r.setParent(table)
                r.addAttribute('points',spoints)
                r.tagMe('VV')
    
    def assessCuts(self,table,lYCuts):
        """
            input: table, ycuts
            output: 
        """
        # features or values ?
        try:lYCuts = map(lambda x:x.getValue(),lYCuts)
        except:pass
        
        lCells = table.getCells()
        prevCut = table.getY()
        irowIndex = 0
        lRows= []
        dCellOverlap = {}
        for _,cut in enumerate(lYCuts):
            row=[]
            if cut - prevCut > 0:
                [b1,b2] = prevCut, cut                   
                for c in lCells:
                    [a1, a2] = c.getY(),c.getY() + c.getHeight()
                    if min(a2, b2) >=  max(a1, b1):  
                        row.append(c)
                lRows.append(row)
                irowIndex += 1                
            prevCut = cut        

        ## BIO coherence
        
    
    
    def buildLineCandidates(self,table):
        """
            return a lits of lines corresponding to top row line candidates
        """
        
    
    def mineTableRowPattern(self,table):
        """
            find rows and columns patterns in terms of typographical position // mandatory cells,...
            input: a set of rows  (table)
            action: seq mining of rows 
            output: pattern
            
        Mining at table/page level
        # of cells per row
        # of cells per colmun
        # cell with content (j: freq  ; i: freq)
        
        Sequential pattern:(itemset: setofrows; item cells?)
        
        """
        # which col is mandatory
        # text alignment  in cells (per col) 
        for row in table.getRows(): 
#             self.mineTypography()
            a = row.computeSkewing()


    """
        skewing detection: use synthetic data !!
        simply scan row by row with previous row and adjust with coherence
        
    """
    
    
    def getSkewingRepresentation(self,lcuts):
        """
            input: list of featureObject
            output: skewed cut (a,b)
            alog: for each feature: get the text nodes baselines and create a skewed line (a,b)
        """
            
            
    
    def miningSeparatorShape(self,table,lCuts):
#         import numpy as np
        from shapely.geometry import MultiLineString
        for cut in lCuts:
            xordered=  list(cut.getNodes())
            print(cut,[x.getX() for x in xordered])
            xordered.sort(key = lambda x:x.getX())
            lSeparators = [ (x.getX(),x.getY()) for x in [xordered[0],xordered[-1]]]
            print( lSeparators)
            ml = MultiLineString(lSeparators)
            print (ml.wkt)
#             X = [x[0] for x in lSeparators]
#             Y = [x[1] for x in lSeparators]
#             print(X,Y)
#             a, b = np.polynomial.polynomial.polyfit(X, Y, 1)
#             xmin, xmax = table.getX(), table.getX2()
#             y1 = a + b * xmin
#             y2 = a + b * xmax
#             print (y1,y2)
#             print ([ (x.getObjects()[0].getBaseline().getY(),x.getObjects()[0].getBaseline().getAngle(),x.getY()) for x in xordered])

    def processRows(self, table, predefinedCuts=[]):
        """
            Apply mining to get Y cuts for rows
            
            If everything is centered?
            Try thnum= [5,10,20,30,40,50] and keep better coherence!
            
            Then adjust skewing ? using features values: for c in lYcuts:  print (c, [x.getY() for x in c.getNodes()])
            
            replace columnMining by cell matching from col to col!!
                simply best match (max overlap) between two cells    NONONO
                
            perform chk of cells (tagging is now very good!) and use it for column mining (chk + remaining cells)
        
        """
#         self.matchCells(table)
#         return
        fMaxCoherence = 0.0
        rowMiner= tableRowMiner()
        # % of columns needed 
        lTHSUP= [0.2,0.3,0.4]
#         lTHSUP= [0.2]
        bestTHSUP =None
        bestthnum= None
        bestYcuts = None
        for thnum in [10,20,30]:  # must be correlated with leading/text height? 
#         for thnum in [30]:  # must be correlated with leading/text height? 

#         for thnum in [50]:  # must be correlated with leading/text height? 

            """
                07/1/2018: to be replace by HChunks
                    for each hchunks: % of cuts(beginning)  = validate the top line as segmentor
                    ## hchunk at cell level  : if yes select hchunks at textline level as well?
            """
            lLYcuts = rowMiner.columnMining(table,thnum,lTHSUP,predefinedCuts)
#             print (lLYcuts)
            # get skewing represenation
#             [ x.setValue(x.getValue()-0) for x in lYcuts ]
            for iy,lYcuts in enumerate(lLYcuts):
#                 print ("%s %s " %(thnum, lTHSUP[iy]))
#                 lYcuts.sort(key= lambda x:x.getValue())
#                 self.miningSeparatorShape(table,lYcuts)
#                 self.assessCuts(table, lYcuts)
#                 self.createRowsWithCuts2(table,lYcuts)
                table.createRowsWithCuts(lYcuts)
                table.reintegrateCellsInColRow()
                coherence = self.computeCoherenceScore(table)
                if coherence > fMaxCoherence:
                    fMaxCoherence = coherence
                    bestYcuts= lYcuts[:]
                    bestTHSUP = lTHSUP[iy]
                    bestthnum= thnum
#                 else: break
#                 print ('coherence Score for (%s,%s): %f\t%s'%(thnum,lTHSUP[iy],coherence,bestYcuts))
        if bestYcuts is not None:
            ### create the separation with the hullcontour : row as polygon!!
            ## if no intersection with previous row : OK
            ## if intersection 
#             print (bestYcuts)
#             for y in bestYcuts:
#                 ## get top elements of the cells to build the boundary ??
#                 print ('%s %s'%(y.getValue(),[(c.getX(),c.getY()) for c in sorted(y.getNodes(),key=lambda x:x.getX())]))
                ## what about elements outside the cut (beforeà)
                ##  try "skew option and evaluate""!!
                ## take max -H 
                ##  take skew 
            table.createRowsWithCuts(bestYcuts)
            table.reintegrateCellsInColRow()
            for row in table.getRows(): 
                row.addAttribute('points',"0,0")
                contour = self.createContourFromListOfElements([x for c in row.getCells() for x in c.getObjects()])
                if contour is not None:
                    spoints = ','.join("%s,%s"%(x[0],x[1]) for x in contour)
                    row.addAttribute('points',spoints)
#         print (len(table.getPage().getAllNamedObjects(XMLDSTABLECELLClass)))
        table.buildNDARRAY()
#         self.mineTableRowPattern(table)
   
#    def defineRowTopBoundary(self,row,ycut):
#        """
#            define a top row boundary
#        """
   
    def  findBoundaryLinesFromChunks(self,table,lhckh):
        """
            create lines from chunks (create with cells)
            
            take each chunk and create (a,b) with top contour
        """
         
        from util.Polygon import  Polygon as dspp
        import numpy as np
        
        dTop_lSgmt = collections.defaultdict(list)
        for chk in lhckh:
            sPoints = chk.getAttribute('points') #.replace(',',' ')
            spoints = ' '.join("%s,%s"%((x,y)) for x,y in zip(*[iter(sPoints.split(','))]*2))
            it_sXsY = (sPair.split(',') for sPair in spoints.split(' '))
            plgn = dspp((float(sx), float(sy)) for sx, sy in it_sXsY)
            try:
                lT, lR, lB, lL = plgn.partitionSegmentTopRightBottomLeft()
                dTop_lSgmt[chk].extend(lT)
            except ValueError: pass       
        #now make linear regression to draw relevant separators
        def getX(lSegment):
            lX = list()
            for x1,y1,x2,y2 in lSegment:
                lX.append(x1)
                lX.append(x2)
            return lX
    
        def getY(lSegment):
            lY = list()
            for x1,y1,x2,y2 in lSegment:
                lY.append(y1)
                lY.append(y2)
            return lY
    
    
        dAB = collections.defaultdict(list)
        icmpt=0
        for icol, lSegment in dTop_lSgmt.items(): #sorted(dTop_lSgmt.items()):
            print (icol,lSegment)
            X = getX(lSegment)
            Y = getY(lSegment)
            #sum(l,())
            lfNorm = [np.linalg.norm([[x1,y1], [x2,y2]]) for x1,y1,x2,y2 in lSegment]
            #duplicate each element 
            W = [fN for fN in lfNorm for _ in (0,1)]
    
            # a * x + b
            a, b = np.polynomial.polynomial.polyfit(X, Y, 1, w=W)
            xmin, xmax = min(X), max(X)
            y1 = a + b * (0)
            y2 = a + b * table.getX2()
            dAB[b].append((a,b))
            rowline = XMLDSTABLEROWClass(icmpt)
            rowline.setPage(table.getPage())
            rowline.setParent(table)
            icmpt+=1
#             table.addColumn(rowline)                                # prevx1, prevymin,x1, ymin, x2, ymax, prevx2, prevymax))
            rowline.addAttribute('points',"%s,%s %s,%s"%(0, y1,  table.getX2(),y2))
#             rowline.setX(prevxmin)
#             rowline.setY(prevy1)
#             rowline.setHeight(y2 - prevy1)
#             rowline.setWidth(xmax- xmin)
            rowline.tagMe('SeparatorRegion')                  
            
#             print (a,b)
            
        
#         for b in sorted(dAB.keys()):
#             print (b,dAB[b])
        
        
        
    def processRows3(self,table,predefinedCuts=[] ):
        """
            build rows:
            for a given cell: if One single Y overlapping cell in the next column: integrate it in the row 
            
        """
        from tasks.TwoDChunking import TwoDChunking
        hchk = TwoDChunking()
        lElts=[]
        [lElts.append(x) for col in table.getColumns() for x in col.getCells()]
        lhchk = hchk.HorizonalChunk(table.getPage(),lElts=lElts,bStrict=False)
          
#         lRows = []
#         curRow = []
#         for col in table.getColumns():
#             lcells = col.getCells()
        
    def processRows2(self,table,predefinedCuts=[]):
        """
            Apply mining to get Y cuts for rows
        
        """
        from tasks.TwoDChunking import TwoDChunking

        hchk = TwoDChunking()
        lhchk = hchk.HorizonalChunk(table.getPage(),lElts=table.getCells())  

        # create bounday lines from lhckh
#         lYcuts = self.findBoundaryLinesFromChunks(table,lhchk)
        
#         lYcuts.sort(key= lambda x:x.getValue())
#                 self.getSkewingRepresentation(lYcuts)
#                 self.assessCuts(table, lYcuts)
#                 self.createRowsWithCuts2(table,lYcuts)
#         table.createRowsWithCuts(lYcuts)
#         table.reintegrateCellsInColRow()
#             
#         table.buildNDARRAY()
        
    def checkInputFormat(self,lPages):
        """
            delete regions : copy regions elements at page object
            unlink subnodes
        """
        for page in lPages:
            lTables = page.getAllNamedObjects(XMLDSTABLEClass)
            for table in lTables:
                lRegions = table.getAllNamedObjects("CELL")
                lElts=[]
                [lElts.extend(x.getObjects()) for x in lRegions]
                [table.addObject(x,bDom=True) for x in lElts]
                [table.removeObject(x,bDom=True) for x in lRegions]
                    
    def processYCuts(self,ODoc):
        from util.XYcut import mergeSegments
        
        self.checkInputFormat(ODoc.getPages())
        for page in ODoc.getPages():
            traceln("page: %d" %  page.getNumber())        
            lTables = page.getAllNamedObjects(XMLDSTABLEClass)
            for table in lTables:
                print ('nb Y: %s'% len(set([round(x.getY()) for x in page.getAllNamedObjects(XMLDSTEXTClass)])),len(page.getAllNamedObjects(XMLDSTEXTClass)))
#                 lCuts, _, _ = mergeSegments([(x.getY(),x.getY() + x.getHeight(),x) for x in page.getAllNamedObjects(XMLDSTEXTClass)],0)
#                 for i, (y,_,cut) in enumerate(lCuts):
#                     ll =list(cut)
#                     ll.sort(key=lambda x:x.getY())
#                     #add column
#                     myRow= XMLDSTABLEROWClass(i)
#                     myRow.setPage(page)
#                     myRow.setParent(table)
#                     table.addObject(myRow)
#                     myRow.setY(y)
#                     myRow.setX(table.getX())
#                     myRow.setWidth(table.getWidth())
#                     if i +1  < len(lCuts):
#                         myRow.setHeight(lCuts[i+1][0]-y)
#                     else: # use table 
#                         myRow.setHeight(table.getY2()-y)
#                     table.addRow(myRow)
#                     print (myRow)
#                     myRow.tagMe(ds_xml.sROW)

        
    def mergeHorizontalCells(self,table):
        """
            merge cell a to b|next col iff b overlap horizontally with a (using right border from points)
            input: a table, with candidate cells
            output: cluster of cells as row candidates
             
             
            simply ignore cells which overlap several cells in the next column
            then: extend row candidates if needed
            
            
            if no column known: simply take the first cell in lright if cells in  lright do ot X overlap (the first nearest w/o issue)
        """
        # firtst create an index for hor neighbours
        lNBNeighboursNextCol=collections.defaultdict(list)
        lNBNeighboursPrevCol=collections.defaultdict(list)
        for cell in table.getCells():
            # get next col
            icol = cell.getIndex()[1]
            if icol < table.getNbColumns()-1:
                nextColCells=table.getColumns()[icol+1].getCells()
                sorted(nextColCells,key=lambda x:x.getY())
                lHOverlap= []
                [lHOverlap.append(c) for c in nextColCells if cell.signedRatioOverlapY(c)> 1]
                # if no overlap: take icol + 2
                lNBNeighboursNextCol[cell].extend(lHOverlap)
            if icol > 1:
                prevColCells=table.getColumns()[icol-1].getCells()
                sorted(prevColCells,key=lambda x:x.getY())
                lHOverlap= []
                [lHOverlap.append(c) for c in prevColCells if cell.signedRatioOverlapY(c)> 1]
                # if not overlap take icol-2
                lNBNeighboursPrevCol[cell].extend(lHOverlap)
 
             
        lcovered=[]        
        for icol,col in enumerate(table.getColumns()):
            sortedC = sorted(col.getCells(),key=lambda x:x.getY())
            for cell in sortedC:
                if len(lNBNeighboursNextCol[cell]) < 2 and len(lNBNeighboursPrevCol[cell]) < 2:
                    if cell not in lcovered:
                        print(type(cell.getContent()))
                        print ('START :', icol,cell, cell.getContent(),cell.getY(),cell.getY2())
                        lcovered.append(cell)
                        lcurRow = [cell]
                        iicol=icol
                        curCell = cell
                        while iicol < table.getNbColumns()-1:
                            nextColCells=table.getColumns()[iicol+1].getCells()
                            sorted(nextColCells,key=lambda x:x.getY())
                            for c in nextColCells: 
                                if len(lNBNeighboursNextCol[c]) < 2 and len(lNBNeighboursPrevCol[c]) < 2:
                                    if curCell.signedRatioOverlapY(c) > 0.25 * curCell.getHeight():
                                        lcovered.append(c)
                                        lcurRow.append(c)
                                        print (curCell, curCell.getY(),curCell.getHeight(),c, curCell.signedRatioOverlapY(c),c.getY(), c.getHeight(),list(map(lambda x:x.getContent(),lcurRow)))
                                        curCell = c
                            iicol +=1
                        print ("FINAL", list(map(lambda x:(x,x.getContent()),lcurRow)) )
                        print ("\t", list(map(lambda x:x.getIndex(),lcurRow)) )
                        if len(lcurRow)>1:
                            # create a contour for visualization
                            # order by col: get top and  bottom polylines for them
                            contour = self.createContourFromListOfElements(lcurRow)
                            spoints = ','.join("%s,%s"%(x[0],x[1]) for x in contour)
                            r = XMLDSTABLEROWClass(1)
                            r.setParent(table)
                            r.addAttribute('points',spoints)
                            r.tagMe('HH')
                    

#     def mergeHorizontalTextLines(self,table):
#         """
#             merge text lines which are aligned
#             input: a table, with candidate textlines
#             output: cluster of textlines as row candidates
#             
#         """
#         from shapely.geometry import Polygon as pp
#         from rtree import index
#         
#         cellidx = index.Index()
#         lTexts = []
#         lPText=[]
#         lReverseIndex  = {}
#         # Populate R-tree index with bounds of grid cells
#         it=0
#         for cell in table.getCells():
#             for text in cell.getObjects():
#                 tt  = pp( [(text.getX(),text.getY()),(text.getX2(),text.getY()),(text.getX2(),text.getY2()), ((text.getX(),text.getY2()))] )
#                 lTexts.append(text)
#                 lPText.append(tt)
#                 cellidx.insert(it, tt.bounds)
#                 it += 1
#                 lReverseIndex[tt.bounds] = text
#         
#         lcovered=[]
#         lfulleval= []
#         for text in lTexts:
#             if text not in lcovered:
# #                 print ('START :', text, text.getContent())
#                 lcovered.append(text)
#                 lcurRow = [text]
#                 curText= text
#                 while curText is not None:
# #                     print (curText, lcurRow)
#     #                 sPoints = text.getAttribute('points') 
#                     sPoints = curText.getAttribute('blpoints')
# #                     print (sPoints) 
#                     # modify for creating aline to the right
#                     # take the most right X
#                     lastx,lasty = list([(float(x),float(y)) for x,y in zip(*[iter(sPoints.split(','))]*2)])[-1]
#     #                 polytext = pp([(float(x),float(y)) for x,y in zip(*[iter(sPoints.split(','))]*2)])
#                     polytext = pp([(lastx,lasty-10),(lastx+1000,lasty-10),(lastx+1000,lasty),(lastx,lasty)])
# #                     print([(lastx,lasty-10),(lastx+1000,lasty-10),(lastx+1000,lasty),(lastx,lasty)])
#                     ltover = [lPText[pos] for pos in cellidx.intersection(polytext.bounds)]
#                     ltover.sort(key=lambda x:x.centroid.coords[0])
#                     lnextStep=[]
# #                     print ('\tnext\t',list(map(lambda x:lReverseIndex[x.bounds].getContent(),ltover)))
# 
#                     for t1 in ltover: 
#                         # here conditions: vertical porjection and Y overlap ; not area!
#                         if polytext.intersection(t1).area > 0.1: #t1.area*0.5:
#                             if t1 not in lnextStep and lReverseIndex[t1.bounds] not in lcovered:
#                                 lnextStep.append(t1)
#                     if lnextStep != []:
#                         lnextStep.sort(key=lambda x:x.centroid.coords[0])
# #                         print ('\t',list(map(lambda x:(lReverseIndex[x.bounds].getX(),lReverseIndex[x.bounds].getContent()),lnextStep)))
#                         nextt = lnextStep[0]
#                         lcurRow.append(lReverseIndex[nextt.bounds])
#                         lcovered.append(lReverseIndex[nextt.bounds])
#                         curText = lReverseIndex[nextt.bounds]
#                     else:curText = None
#                         
# #                 print ("FINAL", list(map(lambda x:(x,x.getContent()),lcurRow)) )
# #                 print ("FINAL", list(map(lambda x:(x,x.getParent()),lcurRow)) )
#                 lfulleval.append(self.comptureClusterHomogeneity(lcurRow,0))
#                 
#                 if len(lcurRow)>1:
#                     # create a contour for visualization
#                     # order by col: get top and  bottom polylines for them
#                     contour = self.createContourFromListOfElements(lcurRow)
#                     spoints = ','.join("%s,%s"%(x[0],x[1]) for x in contour)
#                     r = XMLDSTABLEROWClass(1)
#                     r.setParent(table)
#                     r.addAttribute('points',spoints)
#                     r.tagMe('VV')
#                     r.tagMe()
#         
#         print (sum(lfulleval)/len(lfulleval))
        

    def mergeHorVerTextLines(self,table):
        """
            build HV lines 
        """
        from util import TwoDNeighbourhood as  TwoDRel
        lTexts = []
        if self.bNoTable:
            lTexts = table.getAllNamedObjects(XMLDSTEXTClass)
        else: 
            for cell in table.getCells():
                # bug to be fixed!!
                if cell.getRowSpan() == 1 and cell.getColSpan() == 1:
                    lTexts.extend(set(cell.getObjects()))
            
        for e in lTexts:
            e.lright=[]
            e.lleft=[]
            e.ltop=[]
            e.lbottom=[]
        lVEdge = TwoDRel.findVerticalNeighborEdges(lTexts)         
        for  a,b in lVEdge:
            a.lbottom.append( b )
            b.ltop.append(a)                 
        for elt in lTexts: 
            # dirty!
            elt.setHeight(max(5,elt.getHeight()-3))
            elt.setWidth(max(5,elt.getWidth()-3))
            TwoDRel.rotateMinus90degOLD(elt)              
        lHEdge = TwoDRel.findVerticalNeighborEdges(lTexts)
        for elt in lTexts:
#             elt.tagMe()
            TwoDRel.rotatePlus90degOLD(elt)
#         return     
        for  a,b in lHEdge:
            a.lright.append( b )
            b.lleft.append(a)             
#         ss
        for elt in lTexts:
            elt.lleft.sort(key = lambda x:x.getX(),reverse=True)
#             elt.lright.sort(key = lambda x:x.getX())
            if len(elt.lright) > 1:
                elt.lright = []
            elt.lright.sort(key = lambda x:elt.signedRatioOverlapY(x),reverse=True)
#             print (elt, elt.getY(), elt.lright)
            elt.ltop.sort(key = lambda x:x.getY())
            if len(elt.lbottom) >1:
                elt.lbottom = []
            elt.lbottom.sort(key = lambda x:elt.signedRatioOverlapX(x),reverse=True)
            


        # Horizontal  
        lTexts.sort(key = lambda x:x.getX())
        lcovered=[]
        lfulleval = []
        for text in lTexts:
            if text not in lcovered:
#                 print ('START :', text, text.getContent())
                lcovered.append(text)
                lcurRow = [text]
                curText= text
                while curText is not None:
                    try:
                        nextT = curText.lright[0]
#                         print ('\t',[(x,curText.signedRatioOverlapY(x)) for x in curText.lright])
                        if nextT not in lcovered:
                            lcurRow.append(nextT)
                            lcovered.append(nextT)
                        curText = nextT
                    except IndexError:curText = None
                         
#                 print ("FINAL", list(map(lambda x:(x,x.getContent()),lcurRow)) )
#                 lfulleval.append(self.comptureClusterHomogeneity(lcurRow,0))
                if len(lcurRow) > 1:
                    # create a contour for visualization
                    # order by col: get top and  bottom polylines for them
                    contour = self.createContourFromListOfElements(lcurRow)
                    if contour is not None:
                        spoints = ','.join("%s,%s"%(x[0],x[1]) for x in contour)
                        r = XMLDSTABLEROWClass(1)
                        r.setParent(table)
                        r.addAttribute('points',spoints)
                        r.tagMe('HH')
#         print (sum(lfulleval)/len(lfulleval))


        # Vertical 
        lTexts.sort(key = lambda x:x.getY())
        lcovered=[]
        lfulleval = []
        for text in lTexts:
            if text not in lcovered:
#                 print ('START :', text, text.getContent())
                lcovered.append(text)
                lcurCol = [text]
                curText= text
                while curText is not None:
                    try:
                        nextT = curText.lbottom[0]
#                         print ('\t',[(x,curText.signedRatioOverlapY(x)) for x in curText.lright])
                        if nextT not in lcovered and len(nextT.lbottom) == 1:
                            lcurCol.append(nextT)
                            lcovered.append(nextT)
                        curText = nextT
                    except IndexError:curText = None
                         
#                 print ("FINAL", list(map(lambda x:(x,x.getContent()),lcurCol)) )
#                 lfulleval.append(self.comptureClusterHomogeneity(lcurCol,1))
                if len(lcurCol)>1:
                    # create a contour for visualization
                    # order by col: get top and  bottom polylines for them
                    contour = self.createContourFromListOfElements(lcurCol)
                    if contour is not None:
                        spoints = ','.join("%s,%s"%(x[0],x[1]) for x in contour)
                        r = XMLDSTABLEROWClass(1)
                        r.setParent(table)
                        r.addAttribute('points',spoints)
#                         r.setDimensions(...)
                        r.tagMe('VV')
#         print (sum(lfulleval)/len(lfulleval))
    
        
    def mergeHorVerCells(self,table):
        """
            build HV chunks cells 
        """
        from util import TwoDNeighbourhood as  TwoDRel
        lTexts = []
        for cell in table.getCells():
            # bug to be fixed!!
            if cell.getRowSpan() == 1 and cell.getColSpan() == 1:
#                 lTexts.extend(set(cell.getObjects()))
                lTexts.append(cell)
            
        for e in lTexts:
            e.lright=[]
            e.lleft=[]
            e.ltop=[]
            e.lbottom=[]
        lVEdge = TwoDRel.findVerticalNeighborEdges(lTexts)         
        for  a,b in lVEdge:
            a.lbottom.append( b )
            b.ltop.append(a)                 
        for elt in lTexts: 
            # dirty!
            elt.setHeight(max(5,elt.getHeight()-3))
            elt.setWidth(max(5,elt.getWidth()-3))
            TwoDRel.rotateMinus90degOLD(elt)              
        lHEdge = TwoDRel.findVerticalNeighborEdges(lTexts)
        for elt in lTexts:
#             elt.tagMe()
            TwoDRel.rotatePlus90degOLD(elt)
#         return     
        for  a,b in lHEdge:
            a.lright.append( b )
            b.lleft.append(a)             
#         ss
        for elt in lTexts:
            elt.lleft.sort(key = lambda x:x.getX(),reverse=True)
#             elt.lright.sort(key = lambda x:x.getX())
            elt.lright.sort(key = lambda x:elt.signedRatioOverlapY(x),reverse=True)
            if len(elt.lright) >1:
                elt.lright = []
#             print (elt, elt.getY(), elt.lright)
            elt.ltop.sort(key = lambda x:x.getY())
            elt.lbottom.sort(key = lambda x:elt.signedRatioOverlapX(x),reverse=True)


        # Horizontal  
        lTexts.sort(key = lambda x:x.getX())
        lcovered=[]
        lfulleval = []
        for text in lTexts:
            if text not in lcovered:
#                 print ('START :', text, text.getContent())
                lcovered.append(text)
                lcurRow = [text]
                curText= text
                while curText is not None:
                    try:
                        nextT = curText.lright[0]
#                         print ('\t',[(x,curText.signedRatioOverlapY(x)) for x in curText.lright])
                        if nextT not in lcovered:
                            lcurRow.append(nextT)
                            lcovered.append(nextT)
                        curText = nextT
                    except IndexError:curText = None
                         
                print ("FINAL", list(map(lambda x:(x,x.getContent()),lcurRow)) )
#                 lfulleval.append(self.comptureClusterHomogeneity(lcurRow,0))
                if len(lcurRow) > 1:
                    # create a contour for visualization
                    # order by col: get top and  bottom polylines for them
                    contour = self.createContourFromListOfElements(lcurRow)
                    if contour is not None:
                        spoints = ','.join("%s,%s"%(x[0],x[1]) for x in contour)
                        r = XMLDSTABLEROWClass(1)
                        r.setParent(table)
                        r.addAttribute('points',spoints)
                        r.tagMe('HH')
#         print (sum(lfulleval)/len(lfulleval))


#         # Vertical 
#         lTexts.sort(key = lambda x:x.getY())
#         lcovered=[]
#         lfulleval = []
#         for text in lTexts:
#             if text not in lcovered:
# #                 print ('START :', text, text.getContent())
#                 lcovered.append(text)
#                 lcurCol = [text]
#                 curText= text
#                 while curText is not None:
#                     try:
#                         nextT = curText.lbottom[0]
# #                         print ('\t',[(x,curText.signedRatioOverlapY(x)) for x in curText.lright])
#                         if nextT not in lcovered:
#                             lcurCol.append(nextT)
#                             lcovered.append(nextT)
#                         curText = nextT
#                     except IndexError:curText = None
#                          
# #                 print ("FINAL", list(map(lambda x:(x,x.getContent()),lcurRow)) )
#                 lfulleval.append(self.comptureClusterHomogeneity(lcurCol,1))
#                 if len(lcurCol)>1:
#                     # create a contour for visualization
#                     # order by col: get top and  bottom polylines for them
#                     contour = self.createContourFromListOfElements(lcurCol)
#                     if contour is not None:
#                         spoints = ','.join("%s,%s"%(x[0],x[1]) for x in contour)
#                         r = XMLDSTABLEROWClass(1)
#                         r.setParent(table)
#                         r.addAttribute('points',spoints)
#                         r.tagMe('VV')
#         print (sum(lfulleval)/len(lfulleval))        
        
    def createContourFromListOfElements(self, lElts):
        """
            create a polyline from a list of elements
            input : list of elements
            output: Polygon object
        """
        from shapely.geometry import Polygon as pp
        from shapely.ops import cascaded_union
        lP = []
        for elt in lElts:
            
            sPoints = elt.getAttribute('points')
            if sPoints is None:
                lP.append(pp([(elt.getX(),elt.getY()),(elt.getX(),elt.getY2()), (elt.getX2(),elt.getY2()),(elt.getX2(),elt.getY())] ))
            else:    
                lP.append(pp([(float(x),float(y)) for x,y in zip(*[iter(sPoints.split(','))]*2)]))
        try:ss = cascaded_union(lP)
        except ValueError: 
#             print(lElts,lP)
            return None
        if not ss.is_empty:
            return list(ss.convex_hull.exterior.coords)
        else: return None
    
    
    def comptureClusterHomogeneity(self,c,dir):
        """
            % of elements belonging to the same structre
            dir: 0 : row, 1 column
        """
        
        ldict = collections.defaultdict(list)
        [ ldict[elt.getParent().getIndex()[dir]].append(elt) for elt in c]
        lstat =  ([(k,len(ldict[k])) for k in ldict])
        total = sum([x[1] for x in lstat])
        leval = (max(([len(ldict[x])/total for x in ldict])))
        return leval
        
    def findRowsInDoc(self,ODoc):
        """
            find rows for each table in document
            input: a document
            output: a document where tables have rows
        """
        from tasks.TwoDChunking import TwoDChunking
        
        self.lPages = ODoc.getPages()   
#         hchk = TwoDChunking()
        # not always?
#         self.mergeLineAndCells(self.lPages)
     
        for page in self.lPages:
            traceln("page: %d" %  page.getNumber())
#             print (len(page.getAllNamedObjects(XMLDSTABLECELLClass)))
            lTables = page.getAllNamedObjects(XMLDSTABLEClass)
            for table in lTables:
                # col as polygon
                self.getPolylinesForRowsColumns(table)
#                 self.getPolylinesForRows(table)
#                 rowscuts = list(map(lambda r:r.getY(),table.getRows()))
                rowscuts=[]
#                 traceln ('initial cuts:',rowscuts)
                self.createCells(table)
#                 lhchk = hchk.HorizonalChunk(page,lElts=table.getCells())
#                 hchk.VerticalChunk(page,tag=XMLDSTEXTClass)
#                 self.mergeHorizontalCells(table)
                #   then merge overlaping then sort Y and index  : then  insert ambiguous textlines 
#                 self.mergeHorizontalTextLines(table)
#                 self.mergeHorVerTextLines(table)
#                 self.processRows3(table)
                if self.bCellOnly:
                    continue
#                 self.mergeHorizontalCells(table)
#                 self.mergeHorVerCells(table)
                self.processRows(table,rowscuts)      
#                 self.mineTableRowPattern(table)        
                table.tagMe()
            if self.bNoTable:
                self.mergeHorVerTextLines(page)
                
    
    
#     def extendLines(self,table):
#         """
#             Extend textlines up to table width using baseline 
#             input:table
#             output: table with extended baselines 
#         """
#         for col in table.getColumns():
#             for cell in col.getCells():
#                 for elt in cell.getObjects():
#                     if elt.getWidth()> 100: 
#                         #print ([ (x.getObjects()[0].getBaseline().getY(),x.getObjects()[0].getBaseline().getAngle(),x.getY()) for x in xordered])
#                         print (elt,elt.getBaseline().getAngle(), elt.getBaseline().getBx(),elt.getBaseline().getPoints())
#                         newBl  = [(table.getX(),elt.getBaseline().getAngle()* table.getX() + elt.getBaseline().getBx()),
#                                   (table.getX2(),elt.getBaseline().getAngle()* table.getX2() + elt.getBaseline().getBx())
#                                   ]
#                         elt.getBaseline().setPoints(newBl)
#                         myPoints = '%f,%f,%f,%f'%(newBl[0][0],newBl[0][1],newBl[1][0],newBl[1][1])      
#                         elt.addAttribute('blpoints',myPoints)
        
            
#         sys.exit(0)
        
    def getPolylinesForRowsColumns(self,table):
        """
            input:  list of  cells (=table)
            output: columns defined by polylines (not Bounding box) 
        """
        import numpy as np
        from util.Polygon import  Polygon
        from shapely.geometry import Polygon as pp
#         from shapely.ops import cascaded_union
        from rtree import index
        
        cellidx = index.Index()
        lCells = []
        lReverseIndex  = {}
        # Populate R-tree index with bounds of grid cells
        for pos, cell in enumerate(table.getCells()):
            # assuming cell is a shapely object
            cc  = pp( [(cell.getX(),cell.getY()),(cell.getX2(),cell.getY()),(cell.getX2(),cell.getY2()), ((cell.getX(),cell.getY2()))] )
            lCells.append(cc)
            cellidx.insert(pos, cc.bounds)
            lReverseIndex[cc.bounds] = cell
        
        
        dColSep_lSgmt = collections.defaultdict(list)
        dRowSep_lSgmt = collections.defaultdict(list)
        for cell in table.getCells():
            row, col, rowSpan, colSpan = [int(cell.getAttribute(sProp)) for sProp \
                                          in ["row", "col", "rowSpan", "colSpan"] ]
            sPoints = cell.getAttribute('points') #.replace(',',' ')
#             print (cell,sPoints)
            spoints = ' '.join("%s,%s"%((x,y)) for x,y in zip(*[iter(sPoints.split(','))]*2))
            it_sXsY = (sPair.split(',') for sPair in spoints.split(' '))
            plgn = Polygon((float(sx), float(sy)) for sx, sy in it_sXsY)
#             print (plgn.getBoundingBox(),spoints)
            try:
                lT, lR, lB, lL = plgn.partitionSegmentTopRightBottomLeft()
                #now the top segments contribute to row separator of index: row
                dRowSep_lSgmt[row].extend(lT)
                dRowSep_lSgmt[row+rowSpan].extend(lB)                
                dColSep_lSgmt[col].extend(lL)
                dColSep_lSgmt[col+colSpan].extend(lR)
            except ValueError: pass

        #now make linear regression to draw relevant separators
        def getX(lSegment):
            lX = list()
            for x1,y1,x2,y2 in lSegment:
                lX.append(x1)
                lX.append(x2)
            return lX
    
        def getY(lSegment):
            lY = list()
            for x1,y1,x2,y2 in lSegment:
                lY.append(y1)
                lY.append(y2)
            return lY
    
        prevx1 , prevx2 , prevymin , prevymax = None,None,None,None #table.getX(),table.getX(),table.getY(),table.getY2()
        
        
        # erase columns:
        table.eraseColumns()
        icmpt=0
        for icol, lSegment in sorted(dColSep_lSgmt.items()):
            X = getX(lSegment)
            Y = getY(lSegment)
            #sum(l,())
            lfNorm = [np.linalg.norm([[x1,y1], [x2,y2]]) for x1,y1,x2,y2 in lSegment]
            #duplicate each element 
            W = [fN for fN in lfNorm for _ in (0,1)]
    
            # a * x + b
            a, b = np.polynomial.polynomial.polyfit(Y, X, 1, w=W)
    
            ymin, ymax = min(Y), max(Y)
            x1 = a + b * ymin
            x2 = a + b * ymax 
            if prevx1:
                col = XMLDSTABLECOLUMNClass()
                col.setPage(table.getPage())
                col.setParent(table)
                col.setIndex(icmpt)
                icmpt+=1
                table.addColumn(col)                
                col.addAttribute('points',"%s,%s %s,%s,%s,%s %s,%s"%(prevx1, prevymin,x1, ymin, x2, ymax, prevx2, prevymax))
                col.setX(prevx1)
                col.setY(prevymin)
                col.setHeight(ymax- ymin)
                col.setWidth(x2-prevx1)
                col.tagMe()      
#                 from shapely.geometry import Polygon as pp
                polycol = pp([(prevx1, prevymin),(x1, ymin), (x2, ymax), (prevx2, prevymax)] )
#                 print ((prevx1, prevymin),(x1, ymin), (x2, ymax), (prevx2, prevymax))
#                 colCells = cascaded_union([cells[pos] for pos in cellidx.intersection(polycol.bounds)])
                colCells = [lCells[pos] for pos in cellidx.intersection(polycol.bounds)]
                for cell in colCells:
                    try: 
                        if polycol.intersection(cell).area > cell.area*0.5:
                            col.addCell(lReverseIndex[cell.bounds])
                    except:
                        pass
                
            prevx1 , prevx2 , prevymin , prevymax = x1, x2, ymin, ymax         
            
            
    def getPolylinesForRows(self,table):
        """
            input:  list of  candidate cells (=table)
            output: "rows" defined by top polylines
        """
        import numpy as np
        from util.Polygon import  Polygon
        from shapely.geometry import Polygon as pp
#         from shapely.ops import cascaded_union
        from rtree import index
        
        cellidx = index.Index()
        lCells = []
        lReverseIndex  = {}
        # Populate R-tree index with bounds of grid cells
        for pos, cell in enumerate(table.getCells()):
            # assuming cell is a shapely object
            cc  = pp( [(cell.getX(),cell.getY()),(cell.getX2(),cell.getY()),(cell.getX2(),cell.getY2()), ((cell.getX(),cell.getY2()))] )
            lCells.append(cc)
            cellidx.insert(pos, cc.bounds)
            lReverseIndex[cc.bounds] = cell
        
        
        dColSep_lSgmt = collections.defaultdict(list)
        dRowSep_lSgmt = collections.defaultdict(list)
        for cell in table.getCells():
            row, col, rowSpan, colSpan = [int(cell.getAttribute(sProp)) for sProp \
                                          in ["row", "col", "rowSpan", "colSpan"] ]
            sPoints = cell.getAttribute('points') #.replace(',',' ')
#             print (cell,sPoints)
            spoints = ' '.join("%s,%s"%((x,y)) for x,y in zip(*[iter(sPoints.split(','))]*2))
            it_sXsY = (sPair.split(',') for sPair in spoints.split(' '))
            plgn = Polygon((float(sx), float(sy)) for sx, sy in it_sXsY)
#             print (plgn.getBoundingBox(),spoints)
            try:
                lT, lR, lB, lL = plgn.partitionSegmentTopRightBottomLeft()
                #now the top segments contribute to row separator of index: row
                dRowSep_lSgmt[row].extend(lT)
                dRowSep_lSgmt[row+rowSpan].extend(lB)                
                dColSep_lSgmt[col].extend(lL)
                dColSep_lSgmt[col+colSpan].extend(lR)
            except ValueError: pass

        #now make linear regression to draw relevant separators
        def getX(lSegment):
            lX = list()
            for x1,y1,x2,y2 in lSegment:
                lX.append(x1)
                lX.append(x2)
            return lX
    
        def getY(lSegment):
            lY = list()
            for x1,y1,x2,y2 in lSegment:
                lY.append(y1)
                lY.append(y2)
            return lY
    
        prevxmin , prevxmax , prevy1 , prevy2 = None,None,None,None #table.getX(),table.getX(),table.getY(),table.getY2()
        
        
        # erase columns:
        table.eraseColumns()
        icmpt=0
        for _, lSegment in sorted(dRowSep_lSgmt.items()):
            X = getX(lSegment)
            Y = getY(lSegment)
            #sum(l,())
            lfNorm = [np.linalg.norm([[x1,y1], [x2,y2]]) for x1,y1,x2,y2 in lSegment]
            #duplicate each element 
            W = [fN for fN in lfNorm for _ in (0,1)]
    
            # a * x + b
            a, b = np.polynomial.polynomial.polyfit(X, Y, 1, w=W)
            xmin, xmax = min(X), max(X)
            y1 = a + b * xmin
            y2 = a + b * xmax
    
            if prevy1:
                col = XMLDSTABLEROWClass(icmpt)
                col.setPage(table.getPage())
                col.setParent(table)
                icmpt+=1
                table.addColumn(col)                                # prevx1, prevymin,x1, ymin, x2, ymax, prevx2, prevymax))
                col.addAttribute('points',"%s,%s %s,%s,%s,%s %s,%s"%(prevxmin, prevy1,  prevxmax,prevy2, xmax,y2, prevxmax,y1))
                col.setX(prevxmin)
                col.setY(prevy1)
                col.setHeight(y2 - prevy1)
                col.setWidth(xmax- xmin)
                col.tagMe()      
#                 from shapely.geometry import Polygon as pp
#                 polycol = pp([(prevx1, prevymin),(x1, ymin), (x2, ymax), (prevx2, prevymax)] )
# #                 print ((prevx1, prevymin),(x1, ymin), (x2, ymax), (prevx2, prevymax))
# #                 colCells = cascaded_union([cells[pos] for pos in cellidx.intersection(polycol.bounds)])
#                 colCells = [lCells[pos] for pos in cellidx.intersection(polycol.bounds)]
#                 for cell in colCells: 
#                     if polycol.intersection(cell).area > cell.area*0.5:
#                         col.addCell(lReverseIndex[cell.bounds])
                
                
            prevy1 , prevy2 , prevxmin , prevxmax = y1, y2, xmin, xmax              
            
        for cell in table.getCells():
            del cell._lAttributes['points']
            
    
    def testscale(self,ltexts):        
        return 
        for t in ltexts:
            if True or t.getAttribute('id')[-4:] == '1721':
#                 print (t)
    #             print (etree.tostring(t.getNode()))
                shrinked = affinity.scale(t.toPolygon(),3,-0.8)
    #             print (list(t.toPolygon().exterior.coords), list(shrinked.exterior.coords))
                ss = ",".join(["%s,%s"%(x,y) for x,y in shrinked.exterior.coords])
    #             print (ss)
                t.getNode().set("points",ss)
    #             print (etree.tostring(t.getNode()))
                
            
            
            
    def testshapely(self,Odoc):
        for page in Odoc.lPages:
            self.testscale(page.getAllNamedObjects(XMLDSTEXTClass))
            traceln("page: %d" %  page.getNumber())
#             lTables = page.getAllNamedObjects(XMLDSTABLEClass)
#             for table in lTables:
#                 table.testPopulate()
            
    def run(self,doc):
        """
           load dom and find rows 
        """
        # conver to DS if needed
        if self.bCreateRef:
            if self.do2DS:
                dsconv = primaAnalysis()
                doc = dsconv.convert2DS(doc,self.docid)
            
            refdoc = self.createRef(doc)
            return refdoc
            # single ref per page
#             refdoc= self.createRefPerPage(doc)
#             return None
        
        if self.bCreateRefCluster:
            if self.do2DS:
                dsconv = primaAnalysis()
                doc = dsconv.convert2DS(doc,self.docid)
            
#             refdoc = self.createRefCluster(doc)            
            refdoc = self.createRefPartition(doc)            

            return refdoc
        
        if self.do2DS:
            dsconv = primaAnalysis()
            self.doc = dsconv.convert2DS(doc,self.docid)
        else:
            self.doc= doc
        self.ODoc = XMLDSDocument()
        self.ODoc.loadFromDom(self.doc,listPages = range(self.firstPage,self.lastPage+1))
#         self.testshapely(self.ODoc)
# #         self.ODoc.loadFromDom(self.doc,listPages = range(30,31))        
        if self.bYCut:
            self.processYCuts(self.ODoc)
        else:
            self.findRowsInDoc(self.ODoc)
        return self.doc
        
        

    def computeCoherenceScore(self,table):
        """
            input: table with rows, BIEOS tagged textlines
                                    BIO now !
            output: coherence score
            
            coherence score: float
                percentage of textlines those BIESO tagged is 'coherent with the row segmentation'
            
        """
        coherenceScore = 0
        nbTotalTextLines = 0
        for row in table.getRows():
            for cell in row.getCells():
                nbTextLines = len(cell.getObjects())
                nbTotalTextLines += nbTextLines
                if nbTextLines == 1 and cell.getObjects()[0].getAttribute("DU_row") == STAG: coherenceScore+=1
                else: 
                    for ipos, textline in enumerate(cell.getObjects()):
                        if ipos == 0:
                            if textline.getAttribute("DU_row") in [BTAG]: coherenceScore += 1
                        else:
                            if textline.getAttribute("DU_row") in ['I']: coherenceScore += 1                            
#                         if ipos == nbTextLines-1:
#                             if textline.getAttribute("DU_row") in ['E']: coherenceScore += 1
#                         if ipos not in [0, nbTextLines-1]:
#                             if textline.getAttribute("DU_row") in ['I']: coherenceScore += 1
                        
        if nbTotalTextLines == 0: return 0
        else : return  coherenceScore /nbTotalTextLines
         
    ################ TEST ##################
    

    def testRun(self, filename, outFile=None):
        """
        evaluate using ABP new table dataset with tablecell
        """
        
        self.evalData=None
        doc = self.loadDom(filename)
        doc =self.run(doc)
        if self.bEvalCluster:
            self._evalData = self.createRunPartition( self.ODoc)
#             self.evalData = self.createRefCluster(doc)
        else:
            self.evalData = self.createRef(doc)
        if outFile: self.writeDom(doc)
        return etree.tostring(self._evalData,encoding='unicode',pretty_print=True)
    
    
    
    
    
    def testCluster(self, srefData, srunData, bVisual=False):
        """
        <DOCUMENT>
          <PAGE number="1" imageFilename="g" width="1457.52" height="1085.04">
            <TABLE x="120.72" y="90.72" width="1240.08" height="923.28">
              <ROW>
                <TEXT id="line_1502076498510_2209"/>
                <TEXT id="line_1502076500291_2210"/>
                <TEXT id="line_1502076502635_2211"/>
                <TEXT id="line_1502076505260_2212"/>
        
            
            
            NEED to work at page level !!??
            then average?
        """
        cntOk = cntErr = cntMissed = 0
        
        RefData = etree.XML(srefData.strip("\n").encode('utf-8'))
        RunData = etree.XML(srunData.strip("\n").encode('utf-8'))

        lPages = RefData.xpath('//%s' % ('PAGE[@number]'))
        lRefKeys={}
        dY = {}
        lY={}
        dIDMap={}
        for page in lPages:
            pnum=page.get('number')
            key=page.get('pagekey') 
            dIDMap[key]={}
            lY[key]=[]
            dY[key]={}
            xpath = ".//%s" % ("R")
            lrows = page.xpath(xpath)
            if len(lrows) > 0:
                for i,row in enumerate(lrows):
                    xpath = ".//@id" 
                    lids = row.xpath(xpath)
                    for id in lids:
                        # with spanning an element can belong to several rows?
                        if id not in dY[key]: 
                            dY[key][id]=i 
                            lY[key].append(i)
                            dIDMap[key][id]=len(lY[key])-1
                    try:lRefKeys[key].append((pnum,key,lids))
                    except KeyError:lRefKeys[key] = [(pnum,key,lids)]
        rand_score = completeness = homogen_score = 0
        if RunData is not None:
            lpages = RunData.xpath('//%s' % ('PAGE[@number]'))
            for page in lpages:
                pnum=page.get('number')
                key=page.get('pagekey')
                if key in lRefKeys:
                    lX=[-1 for i in range(len(dIDMap[key]))]
                    xpath = ".//%s" % ("ROW")
                    lrows = page.xpath(xpath)
                    if len(lrows) > 0:
                        for i,row in enumerate(lrows):
                            xpath = ".//@id" 
                            lids = row.xpath(xpath)
                            for id in lids: 
                                lX[ dIDMap[key][id]] = i

                    #adjusted_rand_score(ref,run)
                    rand_score += adjusted_rand_score(lY[key],lX)
                    completeness += completeness_score(lY[key], lX)
                    homogen_score += homogeneity_score(lY[key], lX) 

        ltisRefsRunbErrbMiss= list()
        return (rand_score/len(lRefKeys), completeness/len(lRefKeys), homogen_score/len(lRefKeys),ltisRefsRunbErrbMiss)  
       
       
    def testGeometry(self, th, srefData, srunData, bVisual=False):
        """
            compare geometrical zones (dtw + iou)
            :param
            
            returns tuple (cntOk, cntErr, cntMissed,ltisRefsRunbErrbMiss
            
        """

        cntOk = cntErr = cntMissed = 0
        ltisRefsRunbErrbMiss = list()
        RefData = etree.XML(srefData.strip("\n").encode('utf-8'))
        RunData = etree.XML(srunData.strip("\n").encode('utf-8'))
        
        lPages = RefData.xpath('//%s' % ('PAGE[@number]'))
        
        for ip,page in enumerate(lPages):
            lY=[]
            key=page.get('pagekey') 
            xpath = ".//%s" % ("ROW")
            lrows = page.xpath(xpath)
            if len(lrows) > 0:
                for col in lrows:
                    xpath = ".//@points"
                    lpoints  = col.xpath(xpath) 
                    colgeo = cascaded_union([ Polygon(sPoints2tuplePoints(p)) for p in lpoints])
                    if lpoints != []:
                        lY.append(colgeo)
        
            if RunData is not None:
                lpages = RunData.xpath('//%s' % ('PAGE[@pagekey="%s"]' % key))
                lX=[]
                if lpages != []:
                    for page in lpages[0]:
                        xpath = ".//%s" % ("ROW")
                        lrows = page.xpath(xpath)
                        if len(lrows) > 0:
                            for col in lrows:
                                xpath = ".//@points"
                                lpoints =  col.xpath(xpath)
                                if lpoints != []:
                                    lX.append(  Polygon(sPoints2tuplePoints(lpoints[0])))
                    lX = list(filter(lambda x:x.is_valid,lX))
                    ok , err , missed,lfound,lerr,lmissed = evalPartitions(lX, lY, th,iuo)
                    cntOk += ok 
                    cntErr += err
                    cntMissed +=missed
                    [ltisRefsRunbErrbMiss.append((ip, y1.bounds, x1.bounds,False, False)) for (x1,y1) in lfound]
                    [ltisRefsRunbErrbMiss.append((ip, y1.bounds, None,False, True)) for y1 in lmissed]
                    [ltisRefsRunbErrbMiss.append((ip, None, x1.bounds,True, False)) for x1 in lerr]

#                     ltisRefsRunbErrbMiss.append(( lfound, ip, ok,err, missed))
#                     print (key, cntOk , cntErr , cntMissed)
        return (cntOk , cntErr , cntMissed,ltisRefsRunbErrbMiss)   
            
    def testCluster2(self, th, srefData, srunData, bVisual=False):
        """
        <DOCUMENT>
          <PAGE number="1" imageFilename="g" width="1457.52" height="1085.04">
            <TABLE x="120.72" y="90.72" width="1240.08" height="923.28">
              <ROW>
                <TEXT id="line_1502076498510_2209"/>
                <TEXT id="line_1502076500291_2210"/>
                <TEXT id="line_1502076502635_2211"/>
                <TEXT id="line_1502076505260_2212"/>
        
            
            
            NEED to work at page level !!??
            then average?
        """
        RefData = etree.XML(srefData.strip("\n").encode('utf-8'))
        RunData = etree.XML(srunData.strip("\n").encode('utf-8'))

        lPages = RefData.xpath('//%s' % ('PAGE[@number]'))
        for page in lPages:
            lY=[]
            key=page.get('pagekey') 
            xpath = ".//%s" % ("ROW")
            lrows = page.xpath(xpath)
            if len(lrows) > 0:
                for row in lrows:
                    xpath = ".//@id"
                    lid  = row.xpath(xpath) 
                    if lid != []:
                        lY.append(lid)
#                     print (row.xpath(xpath))
        
            if RunData is not None:
                lpages = RunData.xpath('//%s' % ('PAGE[@pagekey="%s"]' % key))
                lX=[]
                for page in lpages[:1]:
                    xpath = ".//%s" % ("ROW")
                    lrows = page.xpath(xpath)
                    if len(lrows) > 0:
                        for row in lrows:
                            xpath = ".//@id"
                            lid =  row.xpath(xpath)
                            if lid != []:
                                lX.append( lid)
                cntOk , cntErr , cntMissed,lf,le,lm = evalPartitions(lX, lY, th,jaccard)
#                 print ( cntOk , cntErr , cntMissed)
        ltisRefsRunbErrbMiss= list()
        return (cntOk , cntErr , cntMissed,ltisRefsRunbErrbMiss)  
        
        
    def overlapX(self,zone):
        
    
        [a1,a2] = self.getX(),self.getX()+ self.getWidth()
        [b1,b2] = zone.getX(),zone.getX()+ zone.getWidth()
        return min(a2, b2) >=   max(a1, b1) 
        
    def overlapY(self,zone):
        [a1,a2] = self.getY(),self.getY() + self.getHeight()
        [b1,b2] = zone.getY(),zone.getY() + zone.getHeight()
        return min(a2, b2) >=  max(a1, b1)      
    def signedRatioOverlap(self,z1,z2):
        """
         overlap self and zone
         return surface of self in zone 
        """
        [x1,y1,h1,w1] = z1.getX(),z1.getY(),z1.getHeight(),z1.getWidth()
        [x2,y2,h2,w2] = z2.getX(),z2.getY(),z2.getHeight(),z2.getWidth()
        
        fOverlap = 0.0
        
        if self.overlapX(z2) and self.overlapY(z2):
            [x11,y11,x12,y12] = [x1,y1,x1+w1,y1+h1]
            [x21,y21,x22,y22] = [x2,y2,x2+w2,y2+h2]
            
            s1 = w1 * h1
            
            # possible ?
            if s1 == 0: s1 = 1.0
            
            #intersection
            nx1 = max(x11,x21)
            nx2 = min(x12,x22)
            ny1 = max(y11,y21)
            ny2 = min(y12,y22)
            h = abs(nx2 - nx1)
            w = abs(ny2 - ny1)
            
            inter = h * w
            if inter > 0 :
                fOverlap = inter/s1
            else:
                # if overX and Y this is not possible !
                fOverlap = 0.0
            
        return  fOverlap     
    
    def findSignificantOverlap(self,TOverlap,ref,run):
        """
            return 
        """ 
        pref,rowref= ref
        prun, rowrun= run
        if pref != prun: return  False
        
        return rowref.ratioOverlap(rowrun) >=TOverlap 
        
     
    def testCPOUM(self, TOverlap, srefData, srunData, bVisual=False):
        """
            TOverlap: Threshols used for comparing two surfaces
            
            
            Correct Detections:
            under and over segmentation?
        """

        cntOk = cntErr = cntMissed = 0
        
        RefData = etree.XML(srefData.strip("\n").encode('utf-8'))
        RunData = etree.XML(srunData.strip("\n").encode('utf-8'))
#         try:
#             RunData = libxml2.parseMemory(srunData.strip("\n"), len(srunData.strip("\n")))
#         except:
#             RunData = None
#             return (cntOk, cntErr, cntMissed)        
        lRun = []
        if RunData is not None:
            lpages = RunData.xpath('//%s' % ('PAGE'))
            for page in lpages:
                pnum=page.get('number')
                #record level!
                lRows = page.xpath(".//%s" % ("ROW"))
                lORows = map(lambda x:XMLDSTABLEROWClass(0,x),lRows)
                for row in lORows:
                    row.fromDom(row._domNode)
                    row.setIndex(row.getAttribute('id'))
                    lRun.append((pnum,row))            
#         print (lRun)
        
        lRef = []
        lPages = RefData.xpath('//%s' % ('PAGE'))
        for page in lPages:
            pnum=page.get('number')
            lRows = page.xpath(".//%s" % ("ROW"))
            lORows = map(lambda x:XMLDSTABLEROWClass(0,x),lRows)
            for row in lORows:    
                row.fromDom(row._domNode)
                row.setIndex(row.getAttribute('id'))
                lRef.append((pnum,row))  


        refLen = len(lRef)
#         bVisual = True
        ltisRefsRunbErrbMiss= list()
        lRefCovered = []
        for i in range(0,len(lRun)):
            iRef =  0
            bFound = False
            bErr , bMiss= False, False
            runElt = lRun[i]
#             print '\t\t===',runElt
            while not bFound and iRef <= refLen - 1:  
                curRef = lRef[iRef]
                if runElt and curRef not in lRefCovered and self.findSignificantOverlap(TOverlap,runElt, curRef):
                    bFound = True
                    lRefCovered.append(curRef)
                iRef+=1
            if bFound:
                if bVisual:print("FOUND:", runElt, ' -- ', lRefCovered[-1])
                cntOk += 1
            else:
                curRef=''
                cntErr += 1
                bErr = True
                if bVisual:print("ERROR:", runElt)
            if bFound or bErr:
                ltisRefsRunbErrbMiss.append( (int(runElt[0]), curRef, runElt,bErr, bMiss) )
             
        for i,curRef in enumerate(lRef):
            if curRef not in lRefCovered:
                if bVisual:print("MISSED:", curRef)
                ltisRefsRunbErrbMiss.append( (int(curRef[0]), curRef, '',False, True) )
                cntMissed+=1
        ltisRefsRunbErrbMiss.sort(key=lambda xyztu:xyztu[0])

#         print cntOk, cntErr, cntMissed,ltisRefsRunbErrbMiss
        return (cntOk, cntErr, cntMissed,ltisRefsRunbErrbMiss)              
                
        
    def testCompare(self, srefData, srunData, bVisual=False):
        """
        as in Shahad et al, DAS 2010

        Correct Detections 
        Partial Detections 
        Over-Segmented 
        Under-Segmented 
        Missed        
        False Positive
                
        """
        dicTestByTask = dict()
        if self.bEvalCluster:
#             dicTestByTask['CLUSTER']= self.testCluster(srefData,srunData,bVisual)
            dicTestByTask['CLUSTER100']= self.testCluster2(1.0,srefData,srunData,bVisual)
            dicTestByTask['CLUSTER90']= self.testCluster2(0.9,srefData,srunData,bVisual)
            dicTestByTask['CLUSTER80']= self.testCluster2(0.8,srefData,srunData,bVisual)
#             dicTestByTask['CLUSTER50']= self.testCluster2(0.5,srefData,srunData,bVisual)

        else:
            dicTestByTask['T80']= self.testGeometry(0.50,srefData,srunData,bVisual)
#             dicTestByTask['T50']= self.testCPOUM(0.50,srefData,srunData,bVisual)

    
        return dicTestByTask    
    
    
    def createRowsWithCuts2(self,table,lYCuts):
        """
            input: lcells, horizontal lcuts
            output: list of rows populated with appropriate cells  (main overlap)
            
            Algo: create cell chunks and determine (a,b) for the cut (a.X +b = Y)
                  does not solve everything ("russian mountains" in weddings)
        """
        from tasks.TwoDChunking import TwoDChunking
        if lYCuts   == []:
            return
        
        #reinit rows
        self._lrows = []
        
        #build horizontal chunks
        hchk = TwoDChunking()
        hchk.HorizonalChunk(table.getPage(),tag=XMLDSTABLECELLClass)
        
        
#         #get all texts
#         lTexts = []
#         [ lTexts.extend(colcell.getObjects()) for col in table.getColumns() for colcell in col.getObjects()]
#         lTexts.sort(lambda x:x.getY())
#         
#         #initial Y: table top border
#         prevCut = self.getY()
#         
#         # ycuts: features or float 
#         try:lYCuts = map(lambda x:x.getValue(),lYCuts)
#         except:pass
#         
#         itext = 0
#         irowIndex = 0
#         lrowcells = []
#         lprevrowcells = []
#         prevRowCoherenceScore = 0
#         for irow,cut in enumerate(lYCuts):
#             yrow = prevCut
#             y2 = cut
#             h  = cut - prevCut 
#             lrowcells =[] 
#             while lTexts[itext].getY() <= cut:
#                 lrowcells.append(lTexts[itext])
#                 itext += 1
#             if lprevrowcells == []:
#                 pass
#             else:
#                 # a new row: evaluate if this is better to create it or to merge ltext with current row
#                 # check coherence of new texts
#                 # assume columns!
#                 coherence = self.computeCoherenceScoreForRows(lrowcells)
#                 coherenceMerge = self.computeCoherenceScoreForRows(lrowcells+lprevrowcells)
#                 if prevRowCoherenceScore + coherence > coherenceMerge:
#                     cuthere
#                 else:
#                     merge
#          
         
         
                     
    def createRowsWithCuts(self,lYCuts,table,tableNode,bTagDoc=False):
        """
        REF XML
        """
        prevCut = None
#         prevCut = table.getY()
        
        lYCuts.sort()
        for index,cut in enumerate(lYCuts):
            # first correspond to the table: no rpw
            if prevCut is not None:
                rowNode= etree.Element("ROW")
                if bTagDoc:
                    tableNode.append(rowNode)
                else:
                    tableNode.append(rowNode)
                rowNode.set('y',str(prevCut))
                rowNode.set('height',str(cut - prevCut))
                rowNode.set('x',str(table.getX()))
                rowNode.set('width',str(table.getWidth()))
                rowNode.set('id',str(index-1))

            prevCut= cut
        #last
        cut=table.getY2()
        rowNode= etree.Element("ROW")
        tableNode.append(rowNode)
        rowNode.set('y',str(prevCut))
        rowNode.set('height',str(cut - prevCut))
        rowNode.set('x',str(table.getX()))
        rowNode.set('width',str(table.getWidth()))        
        rowNode.set('id',str(index))


    def createRefCluster(self,doc):
        """
            Ref: a row = set of textlines
        """            
        self.ODoc = XMLDSDocument()
        self.ODoc.loadFromDom(doc,listPages = range(self.firstPage,self.lastPage+1))        
  
  
        root=etree.Element("DOCUMENT")
        refdoc=etree.ElementTree(root)
        

        for page in self.ODoc.getPages():
            pageNode = etree.Element('PAGE')
            pageNode.set("number",page.getAttribute('number'))
            pageNode.set("pagekey",os.path.basename(page.getAttribute('imageFilename')))
            pageNode.set("width",page.getAttribute('width'))
            pageNode.set("height",page.getAttribute('height'))

            root.append(pageNode)   
            lTables = page.getAllNamedObjects(XMLDSTABLEClass)
            for table in lTables:
                dRows={}
                tableNode = etree.Element('TABLE')
                tableNode.set("x",table.getAttribute('x'))
                tableNode.set("y",table.getAttribute('y'))
                tableNode.set("width",table.getAttribute('width'))
                tableNode.set("height",table.getAttribute('height'))
                pageNode.append(tableNode)
                for cell in table.getAllNamedObjects(XMLDSTABLECELLClass):
                    try:dRows[int(cell.getAttribute("row"))].extend(cell.getObjects())
                    except KeyError:dRows[int(cell.getAttribute("row"))] = cell.getObjects()
        
                for rowid in sorted(dRows.keys()):
                    rowNode= etree.Element("ROW")
                    tableNode.append(rowNode)
                    for elt in dRows[rowid]:
                        txtNode = etree.Element("TEXT")
                        txtNode.set('id',elt.getAttribute('id'))
                        rowNode.append(txtNode)
                        
        return refdoc
        
    def createRef(self,doc):
        """
            create a ref file from the xml one
        """
        self.ODoc = XMLDSDocument()
        self.ODoc.loadFromDom(doc,listPages = range(self.firstPage,self.lastPage+1))        
  
  
        root=etree.Element("DOCUMENT")
        refdoc=etree.ElementTree(root)
        

        for page in self.ODoc.getPages():
            #imageFilename="..\col\30275\S_Freyung_021_0001.jpg" width="977.52" height="780.0">
            pageNode = etree.Element('PAGE')
            pageNode.set("number",page.getAttribute('number'))
            pageNode.set("pagekey",os.path.basename(page.getAttribute('imageFilename')))
            pageNode.set("width",str(page.getAttribute('width')))
            pageNode.set("height",str(page.getAttribute('height')))

            root.append(pageNode)   
            lTables = page.getAllNamedObjects(XMLDSTABLEClass)
            for table in lTables:
                print (table)
                dRows={}
                tableNode = etree.Element('TABLE')
                tableNode.set("x",str(table.getAttribute('x')))
                tableNode.set("y",str(table.getAttribute('y')))
                tableNode.set("width",str(table.getAttribute('width')))
                tableNode.set("height",str(table.getAttribute('height')))
                for cell in table.getAllNamedObjects(XMLDSTABLECELLClass):
                    print (cell)
                    try:dRows[int(cell.getAttribute("row"))].append(cell)
                    except KeyError:dRows[int(cell.getAttribute("row"))] = [cell]
                lYcuts = []
                for rowid in sorted(dRows.keys()):
#                     print rowid, min(map(lambda x:x.getY(),dRows[rowid]))
                    lYcuts.append(min(list(map(lambda x:x.getY(),dRows[rowid]))))
                if lYcuts != []:
                    pageNode.append(tableNode)
                    self.createRowsWithCuts(lYcuts,table,tableNode)

        return refdoc
    
    def createRefPerPage(self,doc):
        """
            create a ref file from the xml one
            
            for DAS 2018
        """
        self.ODoc = XMLDSDocument()
        self.ODoc.loadFromDom(doc,listPages = range(self.firstPage,self.lastPage+1))        
  
  

        dRows={}
        for page in self.ODoc.getPages():
            #imageFilename="..\col\30275\S_Freyung_021_0001.jpg" width="977.52" height="780.0">
            pageNode = etree.Element('PAGE')
#             pageNode.set("number",page.getAttribute('number'))
            #SINGLER PAGE pnum=1
            pageNode.set("number",'1')

            pageNode.set("imageFilename",page.getAttribute('imageFilename'))
            pageNode.set("width",page.getAttribute('width'))
            pageNode.set("height",page.getAttribute('height'))

            root=etree.Element("DOCUMENT")
            refdoc=etree.ElementTree(root)
            root.append(pageNode)
               
            lTables = page.getAllNamedObjects(XMLDSTABLEClass)
            for table in lTables:
                tableNode = etree.Element('TABLE')
                tableNode.set("x",table.getAttribute('x'))
                tableNode.set("y",table.getAttribute('y'))
                tableNode.set("width",table.getAttribute('width'))
                tableNode.set("height",table.getAttribute('height'))
                pageNode.append(tableNode)
                for cell in table.getAllNamedObjects(XMLDSTABLECELLClass):
                    try:dRows[int(cell.getAttribute("row"))].append(cell)
                    except KeyError:dRows[int(cell.getAttribute("row"))] = [cell]
        
                lYcuts = []
                for rowid in sorted(dRows.keys()):
#                     print rowid, min(map(lambda x:x.getY(),dRows[rowid]))
                    lYcuts.append(min(list(map(lambda x:x.getY(),dRows[rowid]))))
                self.createRowsWithCuts(lYcuts,table,tableNode)

            
            self.outputFileName = os.path.basename(page.getAttribute('imageFilename')[:-3]+'ref')
#             print(self.outputFileName)
            self.writeDom(refdoc, bIndent=True)

        return refdoc    
    
    #         print refdoc.serialize('utf-8', True)
#         self.testCPOUM(0.5,refdoc.serialize('utf-8', True),refdoc.serialize('utf-8', True))
            
    def createRefPartition(self,doc):
        """
            Ref: a row = set of textlines
            :param doc: dox xml
            returns a doc (ref format): each column contains a set of ids (textlines ids)
        """            
        self.ODoc = XMLDSDocument()
        self.ODoc.loadFromDom(doc,listPages = range(self.firstPage,self.lastPage+1))        
  
  
        root=etree.Element("DOCUMENT")
        refdoc=etree.ElementTree(root)
        

        for page in self.ODoc.getPages():
            pageNode = etree.Element('PAGE')
            pageNode.set("number",page.getAttribute('number'))
            pageNode.set("pagekey",os.path.basename(page.getAttribute('imageFilename')))
            pageNode.set("width",str(page.getAttribute('width')))
            pageNode.set("height",str(page.getAttribute('height')))

            root.append(pageNode)   
            lTables = page.getAllNamedObjects(XMLDSTABLEClass)
            for table in lTables:
                dCols={}
                tableNode = etree.Element('TABLE')
                tableNode.set("x",table.getAttribute('x'))
                tableNode.set("y",table.getAttribute('y'))
                tableNode.set("width",str(table.getAttribute('width')))
                tableNode.set("height",str(table.getAttribute('height')))
                pageNode.append(tableNode)
                for cell in table.getAllNamedObjects(XMLDSTABLECELLClass):
                    try:dCols[int(cell.getAttribute("row"))].extend(cell.getObjects())
                    except KeyError:dCols[int(cell.getAttribute("row"))] = cell.getObjects()
        
                for rowid in sorted(dCols.keys()):
                    rowNode= etree.Element("ROW")
                    tableNode.append(rowNode)
                    for elt in dCols[rowid]:
                        txtNode = etree.Element("TEXT")
                        txtNode.set('id',elt.getAttribute('id'))
                        rowNode.append(txtNode)
                        
        return refdoc
                

    def createRunPartition(self,doc):
        """
            Ref: a row = set of textlines
            :param doc: dox xml
            returns a doc (ref format): each column contains a set of ids (textlines ids)
        """            
#         self.ODoc = doc #XMLDSDocument()
#         self.ODoc.loadFromDom(doc,listPages = range(self.firstPage,self.lastPage+1))        
  
  
        root=etree.Element("DOCUMENT")
        refdoc=etree.ElementTree(root)
        

        for page in self.ODoc.getPages():
            pageNode = etree.Element('PAGE')
            pageNode.set("number",page.getAttribute('number'))
            pageNode.set("pagekey",os.path.basename(page.getAttribute('imageFilename')))
            pageNode.set("width",str(page.getAttribute('width')))
            pageNode.set("height",str(page.getAttribute('height')))

            root.append(pageNode)   
            tableNode = etree.Element('TABLE')
            tableNode.set("x","0")
            tableNode.set("y","0")
            tableNode.set("width","0")
            tableNode.set("height","0")
            pageNode.append(tableNode)
            
            table = page.getAllNamedObjects(XMLDSTABLEClass)[0]
            lRows  = table.getRows()
            for row in lRows:
                cNode= etree.Element("ROW")
                tableNode.append(cNode)
                for elt in row.getAllNamedObjects(XMLDSTEXTClass):
                    txtNode= etree.Element("TEXT")
                    txtNode.set('id',elt.getAttribute('id'))
                    cNode.append(txtNode)
                        
        return refdoc            
            
if __name__ == "__main__":

    
    rdc = RowDetection()
    #prepare for the parsing of the command line
    rdc.createCommandLineParser()
#     rdc.add_option("--coldir", dest="coldir", action="store", type="string", help="collection folder")
    rdc.add_option("--docid", dest="docid", action="store", type="string", help="document id")
    rdc.add_option("--dsconv", dest="dsconv", action="store_true", default=False, help="convert page format to DS")
    rdc.add_option("--createref", dest="createref", action="store_true", default=False, help="create REF file for component")
    rdc.add_option("--createrefC", dest="createrefCluster", action="store_true", default=False, help="create REF file for component (cluster of textlines)")
    rdc.add_option("--evalC", dest="evalCluster", action="store_true", default=False, help="evaluation using clusters (of textlines)")
    rdc.add_option("--cell", dest="bCellOnly", action="store_true", default=False, help="generate cell candidate from BIO (no row)")
    rdc.add_option("--nocolumn", dest="bNoColumn", action="store_true", default=False, help="no existing table/colunm)")
#     rdc.add_option("--raw", dest="bRaw", action="store_true", default=False, help="no existing table/colunm)")

    rdc.add_option("--YC", dest="YCut", action="store_true", default=False, help="use Ycut")

    rdc.add_option("--thhighsupport", dest="thhighsupport", action="store", type="int", default=33,help="TH for high support", metavar="NN")

    rdc.add_option('-f',"--first", dest="first", action="store", type="int", help="first page to be processed")
    rdc.add_option('-l',"--last", dest="last", action="store", type="int", help="last page to be processed")

    #parse the command line
    dParams, args = rdc.parseCommandLine()
    
    #Now we are back to the normal programmatic mode, we set the component parameters
    rdc.setParams(dParams)
    doc = rdc.loadDom()
    doc = rdc.run(doc)
    if doc is not None and rdc.getOutputFileName() != '-':
        rdc.writeDom(doc, bIndent=True) 
    
