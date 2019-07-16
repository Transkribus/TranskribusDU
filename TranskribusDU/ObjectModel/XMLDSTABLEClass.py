# -*- coding: utf-8 -*-
"""

    XMLDS TABLE
    Hervé Déjean
    cpy Xerox 2017
    
    a class for table from a XMLDSDocument

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
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import

from .XMLDSObjectClass import XMLDSObjectClass
from .XMLDSCELLClass import XMLDSTABLECELLClass
from .XMLDSTableColumnClass import XMLDSTABLECOLUMNClass
from .XMLDSTableRowClass import XMLDSTABLEROWClass
from .XMLDSTEXTClass import XMLDSTEXTClass

from config import ds_xml_def as ds_xml
from copy import deepcopy
import numpy as np
from shapely.geometry import MultiPolygon 
from shapely.geometry.collection import GeometryCollection

class  XMLDSTABLEClass(XMLDSObjectClass):
    """
        TABLE class
        
        
        need a TableClass for table primitive, independenty for mXMLDS
    """
    name = ds_xml.sTABLE
    def __init__(self,domNode = None):
        XMLDSObjectClass.__init__(self)
        XMLDSObjectClass.id += 1
        self._name = ds_xml.sTABLE
        self._domNode = domNode

        self._lspannedCells=[]
        
        self._lcells=[]
        # need to define dim later on
        self._npcells=None
        
        self._lcolumns= []
        self._lrows = []
    
        self._nbCols= None
        self._nbRows = None
    
    
    def getCells(self): return self._lcells
    
  
    def getCellsbyColumns(self):
        self._lcells.sort(key=lambda x:x.getIndex()[1])        
        if self.getColumns() == []:
            self.buildColumnFromCells()
        self._lcells= []
        [self._lcells.append(cell) for col in self.getColumns() for cell in col.getCells()]
        return self.getCells()
    
    def getCellsbyRow(self):
        self._lcells.sort(key=lambda x:x.getIndex()[0])        
        if self.getRows() == []:
            self.buildRowFromCells()
        self._lcells= []
        [self._lcells.append(cell) for row in self.getRows() for cell in row.getCells()]
        return self.getCells()
    
    def addCell(self,cell):
        """
            add a  cell
            update row and col data structure? no 
            but insert cell  at the right position
        """
        self._lcells.append(cell)
        self.addObject(cell)
        
    def delCell(self,cell):
        try:
            self._lcells.remove(cell)
#             self._lObjects.remove(cell)
        except:pass #print ('cell already deleted?',cell)


    def getNbRows(self): 
        if self._nbRows is not None:return  self._nbRows
        self._nbRows= len(self.getRows())
        return  self._nbRows
    def getNbColumns(self):
        if self._nbCols is not None: return self._nbCols
        self._nbCols = len(self.getColumns())
        return self._nbCols
        
    def getColumns(self): return self._lcolumns
    def getRows(self): return self._lrows
    
    def addRow(self,row):
        if len(self.getRows()) == row.getIndex():
            self._lrows.append(row)
        else:
            return None
        return row
    
    
    def displayPerRow(self):
        '''
            use rows
        '''
        for row in self.getRows():
            for cell in row.getCells():
                print(cell, cell.getFields(),end='')
            print()
      
    def addColumn(self,col):
        """
            if no col: padding to create one
        """
        if len(self._lcolumns) == col.getIndex():
            self._lcolumns.append(col)
        else:
            icol =len(self._lcolumns)
            while icol < col.getIndex():
                paddingcol =XMLDSTABLECOLUMNClass(icol)
                paddingcol.setPage(self.getPage())
                self._lcolumns.append(paddingcol)
                icol += 1
            self._lcolumns.append(col) 
        return col
    
    
    def eraseColumns(self):
        """
            delete all columns
        """
        self._lcolumns = []
        self._nbCols= None

    def eraseRows(self):
        """
            delete all rows
        """
        self._lrows = []
        self._nbRows= None
        
    def tagMe(self,sLabel=None):
        super(XMLDSObjectClass,self).tagMe(sLabel)
        for r in self.getRows():r.tagMe()
        for c in self.getColumns():c.tagMe()
    
    def createRowsWithCuts(self,lYCuts,bTakeAll=False):
        """
            input: horizontal lcuts
            output: list of rows populated with appropriate cells  (main overlap)
        """
        if lYCuts == []:
            return
        
        #reinit rows? yes
        self._lrows = []
        self._nbRows = None
#         lCells =self.getCells()
        prevCut = self.getY()
        # 
        try:
            lYCuts = list(map(lambda x:x.getValue(),lYCuts))
        except:
            pass
            
        irowIndex = 0
        for irow,cut in enumerate(lYCuts):
#             cut = cut-10   # 10 for ABP
            row= XMLDSTABLEROWClass(irowIndex)
            row.setParent(self)
            row.setY(prevCut)
            # row too narow from table border
#             if cut.getValue()-prevCut > 0:
            if bTakeAll or cut - prevCut > 0:
#                 row.addAttribute('height',cut.getValue()-prevCut)
                row.setHeight(cut - prevCut)
                row.setX(self.getX())
                row.setWidth(self.getWidth())
                row.addAttribute('points',"%s,%s,%s,%s,%s,%s,%s,%s"%(self.getX(), self.getY(),self.getX2(), self.getY(), self.getX2(), self.getY2(), self.getX(), self.getY2()))
                self.addRow(row)
#                 row.tagMe() 
                irowIndex+=1                
            else:
                del(row)
#             prevCut= cut.getValue()
            prevCut= cut
            
        #last row
#         if lYCuts != []:

        row= XMLDSTABLEROWClass(irowIndex)
        row.setParent(self)
        row.setY(prevCut)
        row.setHeight(self.getY2()-prevCut)
        row.setX(self.getX())
        row.setWidth(self.getWidth())
        row.addAttribute('index',row.getIndex())
        row.addAttribute('points',"%s,%s,%s,%s,%s,%s,%s,%s"%(self.getX(), self.getY(),self.getX2(), self.getY(), self.getX2(), self.getY2(), self.getX(), self.getY2()))
        self.addRow(row)       
#         row.tagMe() 
        
        ##  recreate correctly cells : nb cells: #rows x #col
        
        
        
        

    def testPopulate0(self):
        """
            test shapely library
            Take the cells and populate with textlines 
        """
        lpcell = [cell.toPolygon() for cell in self.getCells()]
        for text in self.getAllNamedObjects(XMLDSTEXTClass):
            for pcell in lpcell:
                pcell.intersection(text.toPolygon())
                        
    def testPopulate(self):
        """
            test shapely library
            Take the cells and populate with textlines 
        """
#         return self.testPopulate0()
        from rtree import index
        
#         print (len(self.getCells()))
#         # create cell index
        lIndCells =   index.Index()
        for pos, cell  in enumerate(self.getCells()):
            lIndCells.insert(pos, cell.toPolygon().bounds)

        for text in self.getAllNamedObjects(XMLDSTEXTClass):
            ll  = list(lIndCells.intersection(text.toPolygon().bounds))
            bestcell = None
            bestarea = 0
            if ll != []:
                print (text.getAttribute("id"),[self.getCells()[i] for i in ll])
                for i,x in enumerate(ll): 
                    a =  text.toPolygon().intersection(self.getCells()[x].toPolygon()).area
                    if a  > bestarea:
                        bestarea = a 
                        bestcell = x
                print (text.getAttribute("id"),self.getCells()[bestcell],bestarea,text.toPolygon().area)
                
    
        
    def reintegrateCellsInColRow(self,lObj=[]):
        """
            after createRowsWithCuts, need for refitting cells in the rows (and merge them)
            
            1- create new regular cells from rows and columns 
            2- for each cell:
                for each text: compute overlap
                    store best for text
            3- assign text to best cell
            
            
        
        """
        
#         import numpy as np
        from shapely.geometry import Polygon 
#         
        lCells = []
        NbrowsToDel=0
        lRowWithPB = []
        for icol,col in enumerate(self.getColumns()):
            polycol = col.toPolygon().buffer(0)
            if not polycol.is_valid: polycol=polycol.convex_hull
            rowCells = [] 
            lColWithPb = []

            for irow, row in enumerate(self.getRows()):
                polyrow= row.toPolygon().buffer(0)
                if not polyrow.is_valid :polyrow = polyrow.convex_hull
                if polycol.is_valid and polyrow.is_valid and polycol.intersection(polyrow).area> 0.1 :#(polyrow.area*0.25):
                    cell=XMLDSTABLECELLClass()
                    inter  = polycol.intersection(polyrow)
                    if not inter.is_valid: inter =inter.convex_hull
                    x,y,x2,y2 = inter.bounds
                    cell.setXYHW(x,y, y2-y,x2-x)
                    ## due to rox/col defined with several lines
                    if isinstance(inter,MultiPolygon) or isinstance(inter,GeometryCollection):
                        linter= list(inter.geoms)
                        linter.sort(key=lambda x:x.area,reverse=True)
                        inter = linter[0]
#                     elfi if isinstance(inter,MultiPolygon):
                    if isinstance(inter,Polygon):
                        rowCells.append(cell)    
                        ll = list(inter.exterior.coords)
                        cell.addAttribute('points', " ".join(list("%s,%s"%(x,y) for x,y in ll)))
                        cell.setIndex(irow, icol)
                        cell.setPage(self.getPage())
                        row.addCell(cell)
                        col.addCell(cell)
                        cell.setParent(self)
#                         print (irow,icol,cell,cell.getAttribute('points'))    
                    else:
#                         print([x.area for x in inter])
                        print (irow,icol,type(inter),list(inter.geoms))
#                         sss
                else:
                    print("EMPTY?",polycol.intersection(polyrow).area,irow,icol,polycol.is_valid,polyrow.is_valid,polycol.intersection(polyrow))
#                     lColWithPb.add(icol)
                    lRowWithPB.append(irow)
                    lColWithPb.append(icol)
                    #empty cell zone!!
#                     cell=XMLDSTABLECELLClass()
#                     rowCells.append(cell)    
# #                     ll = list(inter.exterior.coords)
#                     #cell.addAttribute('points', " ".join(list("%s,%s"%(x,y) for x,y in ll)))
#                     cell.setIndex(irow, icol)
#                     cell.setPage(self.getPage())
#                     row.addCell(cell)
#                     col.addCell(cell)
#                     cell.setParent(self)        
#             print (lColWithPb,lRowWithPB)            
            if False and len(lColWithPb) >0  and len(lRowWithPB)>len(lColWithPb):
#                 print (len(lColWithPb),len(lRowWithPB))
                NbrowsToDel+=1
            else:   
                lCells.extend(rowCells)
        for text in lObj:
            cell = text.bestRegionsAssignment(lCells,bOnlyBaseline=False)
#             print(text.getContent(),cell)
            if cell:
                cell.addObject(text,bDom=True)     


        
        #update with real cells
        self._lcells = lCells
#         print (len(lCells))
        # DOM tagging
#         for cell in self.getCells():
#             cell.tagMe2()
#             for o in cell.getObjects():
#                 try:o.tagMe()
#                 except AttributeError:pass
        
#         # update rows!!!!
#         print (self.getRows()[1].getX())
#         self.buildRowFromCells()
#         print (self.getRows())
#         ss            
    def reintegrateCellsInColRowold(self):
        """
            after createRowsWithCuts, need for refitting cells in the rows (and merge them)
            
            1- create new regular cells from rows and columns 
            2- for each cell:
                for each text: compute overlap
                    store best for text
            3- assign text to best cell
            
            
        
        """
        
#         for cell in self.getCellsbyColumns():
#         print self.getNbRows(), self.getNbColumns()
        lCells = []
        for icol,col in enumerate(self.getColumns()):
            lcolTexts = []
            [ lcolTexts.extend(colcell.getObjects()) for colcell in col.getObjects()]
            # texts tagged OTHER as well?
            rowCells = [] 
            for irow, row in enumerate(self.getRows()):
                cell=XMLDSTABLECELLClass()
                rowCells.append(cell)
                cell.setXYHW(col.getX(),row.getY(), row.getY2() - row.getY(),col.getX2() - col.getX())
                cell.setIndex(irow, icol)
                cell.setPage(self.getPage())
                cell.setParent(self)
                row.addCell(cell)
                col.addCell(cell)
                ## spanning information missing!!!!
                
            for text in lcolTexts:
                cell = text.bestRegionsAssignmentOld(rowCells,bOnlyBaseline=False)
                if cell:
                    cell.addObject(text,bDom=True)     

            lCells.extend(rowCells)
        #delete fake cells
        for cell in self.getCells():
#             cell.getNode().unlinkNode()
            if cell.getNode().getparent() is not None:
                cell.getNode().getparent().remove(cell.getNode())
            del cell
        
        #update with real cells
        self._lcells = lCells
    
        # DOM tagging
        for cell in self.getCells():
            cell.tagMe2()
            for o in cell.getObjects():
#                 print o, o.getContent(),o.getParent(), o.getParent().getNode()
                try:o.tagMe()
                except AttributeError:pass
        
        # update rows!!
        #self.buildRowFromCells()

    
    def buildRowFromCells(self):
        """
            build row objects and 2dzones  from cells
            
            Rowspan: create row
        """
        self._lrows=[]
        self.getCells().sort(key=(lambda x:x.getIndex()[0]))
        for cell in self.getCells():
            irow,_= cell.getIndex()
#             rowSpan = int(cell.getAttribute('rowSpan'))
            try: row = self.getRows()[irow]
            except IndexError:
                row = XMLDSTABLEROWClass(irow)
                row = self.addRow(row)
                if row is not None:
                    row.setPage(self.getPage())
            if row is not None:row.addCell(cell)
        
        
        for row in self.getRows():
            row.resizeMe(XMLDSTABLECELLClass)
#             print (row.getIndex(),row.getParent())
                
    def buildColumnFromCells(self):
        """
            build column objects and 2dzones  from cells
        """
        self.getCells().sort(key=(lambda x:x.getIndex()[1]))
        self._lcolumns= []
        for cell in self.getCells():
#             print (cell, cell.getRowSpan(), cell.getColSpan(), cell.getObjects())
            _,jcol= cell.getIndex()
            try: col = self.getColumns()[jcol]
            except IndexError:
                col = XMLDSTABLECOLUMNClass(jcol)
                col = self.addColumn(col)
                if col is not None:col.setPage(self.getPage())
#                 print col.getPage(), self.getPage()
            if col is not None:col.addCell(cell)
            
        
        for col in self.getColumns():
            try:
                col.resizeMe(XMLDSTABLECELLClass)
            except:pass
#             node = col.tagMe(col.tagname)


        
    ## SPANNING PART ###
    # must  replace buildColumnFromCells and buildRowFromCells see also getCellsbyRow?
    def buildColumnRowFromCells(self):
        """
            => rather 'grid' table: (segment)
        """
#         print 'nb cells:' , len(self.getCells())
        # first despan RowSpan cells
        self.getCells().sort(key=(lambda x:x.getIndex()[0]))
        lNewCells = []
        for cell in self.getCells():
            # create new non-spanned cell if needed
#             print cell, cell.getRowSpan(), cell.getColSpan()
            iRowSpan = cell.getIndex()[0]
            while iRowSpan < cell.getIndex()[0] + cell.getRowSpan():
#                 print 'row:', cell, cell.getRowSpan(),iRowSpan
                newCell = XMLDSTABLECELLClass(cell.getNode())
                newCell.setName(XMLDSTABLECELLClass.name)
                newCell.setPage(self.getPage())
                newCell.setParent(self)
                newCell.setObjectsList(cell.getObjects())
                newCell._lAttributes = deepcopy(cell.getAttributes())
                newCell.copyXYHW(cell)
                newCell.addAttribute('rowSpan',newCell.getRowSpan())
#                 newCell.setIndex(newCell.getIndex()[0]+iRowSpan, newCell.getIndex()[1])
                newCell.setIndex(iRowSpan, cell.getIndex()[1])
                newCell.setSpannedCell(cell)
#                 cell.setSpannedCell(cell)
                newCell.bDeSpannedRow = True
                lNewCells.append(newCell)
                iRowSpan +=1

        # col span
        #sort them by col?
#         self.getCells().sort(key=(lambda x:x.getIndex()[1]))
        lNewCells.sort(key=(lambda x:x.getIndex()[1]))
        lNewCells2 = []
        for cell in lNewCells: #self.getCells():
            # create new non-spanned cell if needed
            iColSpan = cell.getIndex()[1]
            while iColSpan < cell.getIndex()[1] + cell.getColSpan():
                newCell = XMLDSTABLECELLClass(cell.getNode())
                newCell.setName(XMLDSTABLECELLClass.name)
                newCell.setParent(self)
                newCell._lAttributes = deepcopy(cell.getAttributes())
                newCell.copyXYHW(cell)
                newCell.setObjectsList(cell.getObjects())
                newCell.addAttribute('colSpan',newCell.getColSpan())
#                 newCell.setIndex(newCell.getIndex()[0], newCell.getIndex()[1]+iColSpan)
                newCell.setIndex(cell.getIndex()[0], iColSpan)
                newCell.setSpannedCell(cell)
#                 cell.setSpannedCell(cell)
                newCell.bDeSpannedCol = True
#                 cell.bDeSpannedCol = True
                lNewCells2.append(newCell)
#                 print '\tnex col cell:',newCell, iColSpan+1, cell.getColSpan()
                iColSpan +=1
#         self.getCells().extend(lNewCells)
        self._lcells = lNewCells
#         print '-- nb cells:', len(self.getCells())
        
    def assignElementsToCells(self):
        """
            assign elements (textlines) to the right cell 
            assuming rows and columns are ok
            
            see IE_test mergeLineAndCells
        """
        #where to get all elements? take the elements of the page?
    
    
    def buildNDARRAY(self):
        if self.getRows() == []:
            self.buildRowFromCells()
        lce=[]
        [lce.append(cell) for row in self.getRows() for cell in row.getCells()]
        self._npcells = np.array(lce,dtype=object)
        self._npcells = self._npcells.reshape(self.getNbRows(),self.getNbColumns())
    
    
    def getNPArray(self):
        return self._npcells
    
    
    
    ###############  IE_TAGGING METHOD  ######################
    
    def tagColumn(self,field):
        """
            tag column with field
        """
        for col in self.getColumns():
            lcellHeader = col.getCellHeader()
            for cell in lcellHeader:
                if cell.match(field.getLabels()):
                    col.addField(field.tag)

    def tagRow(self,field):
        """
            tag row with field
        """
        for row in self.getRows():
            lcellHeader = row.getCellHeader()
            for cell in lcellHeader:
                if cell.match(field.getLabels()):
                    row.addField(field.tag)
    
    def addField(self,tag):
        """
            all cell contains a tag information
        """
        [cell.addField(tag) for cell in self.getCells()]
        
        
    
    ############### MNING  ###############  
    """    
        use of pm/spm
    """
    def mineColumns(self):
        """
        get patterns for each column  (even by concatenating several pages)
        """  
        # create itemset at cell level and item at line level
        # then pattern mining (spm)
        ## item features  : content, position in cell
        # return the most frequent patterns
    
    
    
    ###############  LOAD FROM DSXML ######################    
    
    def fromDom(self,domNode):
        """
            load cells
        """
        self.setName(domNode.tag)
        self.setNode(domNode)
        # get properties
        for prop in domNode.keys():
            self.addAttribute(prop,domNode.get(prop))
            if prop =='x': self._x= float(domNode.get(prop))
            elif prop =='y': self._y = float(domNode.get(prop))
            elif prop =='height': self._h = float(domNode.get(prop))
            elif prop =='width': self.setWidth(float(domNode.get(prop)))
                    
            
#         ctxt = domNode.doc.xpathNewContext()
#         ctxt.setContextNode(domNode)
#         ldomElts = ctxt.xpathEval('./%s'%(ds_xml.sCELL))
#         ctxt.xpathFreeContext()
        
        ldomElts = domNode.findall('./%s'%(ds_xml.sCELL))
        for elt in ldomElts:
            myObject= XMLDSTABLECELLClass(elt)
            self.addCell(myObject)
            myObject.setPage(self.getPage())
            myObject.fromDom(elt)   
        
#         self._lspannedCells = self._lcells[:]
        
        self.buildColumnRowFromCells()
        self.buildColumnFromCells()
        self.buildRowFromCells()
        self.getCellsbyRow()


#         self.displayPerRow()
#         print  self.getNbRows(), self.getNbColumns()
        
    
    
