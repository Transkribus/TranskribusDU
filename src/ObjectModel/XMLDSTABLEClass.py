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

from XMLDSObjectClass import XMLDSObjectClass
from XMLDSCELLClass import XMLDSTABLECELLClass
from XMLDSColumnClass import XMLDSTABLECOLUMNClass
from XMLDSRowClass import XMLDSTABLEROWClass

from config import ds_xml_def as ds_xml

import numpy as np
from theano.tensor.basic import irow

class  XMLDSTABLEClass(XMLDSObjectClass):
    """
        TABLE class
        
        
        need a TableClass for table primitive, independenty for mXMLDS
    """
    name = ds_xml.sLINE_Elt
    def __init__(self,domNode = None):
        XMLDSObjectClass.__init__(self)
        XMLDSObjectClass.id += 1
        
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
            but insert cell  at the rihht position
        """
        self._lcells.append(cell)
        self.addObject(cell)
        
    def delCell(self,cell):
        try:self._lcells.remove(cell)
        except:pass


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
                print cell, cell.getFields(),
            print
                
    
      
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
    
    
    def createRowsWithCuts(self,lYCuts):
        """
            input: a table and cells
            output: list of rows populated with appropriate cells  (main overlap)
        """
        if lYCuts == []:
            return
        
        #reinit rows? yes
        self._lrows = []
        lCells =self.getCells()
        prevCut = self.getY()
        lYCuts.sort(key = lambda x:x.getValue())
        irowIndex=0
        for irow,cut in enumerate(lYCuts):
            row= XMLDSTABLEROWClass(irowIndex)
            row.setParent(self)
            row.addAttribute('y',prevCut)
            # row too narow from table border
            if cut.getValue()-prevCut > 0:
                row.addAttribute('height',cut.getValue()-prevCut)
                row.addAttribute('x',self.getX())
                row.addAttribute('width',self.getWidth())
                row.tagMe()
                for c in lCells:
                    if c.overlap(row):
                        row.addObject(c)      
                        c.setIndex(irow,c.getIndex()[1])
                        row.addCell(c)
                self.addRow(row)
                irowIndex+=1                
            else:
                del(row)
            prevCut= cut.getValue()
        #last row
#         if lYCuts != []:

        row= XMLDSTABLEROWClass(irow)
        row.setParent(self)
        row.addAttribute('y',prevCut)
        row.addAttribute('height',self.getY2()-prevCut)
        row.addAttribute('x',self.getX())
        row.addAttribute('width',self.getWidth())
        row.tagMe()
        for c in lCells:
            if c.overlap(row):
                row.addObject(c)      
                c.setIndex(irow,c.getIndex()[1])
                row.addCell(c)
        self.addRow(row)        
        
        ##  recreate correctly cells : nb cells: #rows x #col
    
    def reintegrateCellsInColRow(self):
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
        lCells=[]
        for icol,col in enumerate(self.getColumns()):
#             print col, col.getX(),col.getX2(), col.getObjects()
            for irow, row in enumerate(self.getRows()):
#                 print row, row.getObjects()
#                 print 'cell:', col.getX(),row.getY(),col.getX2(),row.getY2()
                cell=XMLDSTABLECELLClass()
                lCells.append(cell)
                cell.addAttribute('x',col.getX())
                cell.addAttribute('y',row.getY())
                cell.addAttribute('height',row.getY2() - row.getY())
                cell.addAttribute('width',col.getX2() - col.getX())
                cell.setIndex(irow, icol)
                cell.setPage(self.getPage())
                for colcell in col.getObjects():
#                     print colcell,colcell.signedRatioOverlap(cell), colcell.overlap(cell)
                    if colcell.signedRatioOverlap(cell):
                        cell.setObjectsList(colcell.getObjects()[:])
                        for o in cell.getObjects():
                            o.setParent(cell)
        
        for cell in self.getCells():
            cell.getNode().unlinkNode()
            del cell
        self._lcells = lCells
        for cell in self.getCells():
            cell.tagMe()
            for o in cell.getObjects():
#                 print o, o.getParent(), o.getParent().getNode()
                try:o.tagMe()
                except AttributeError:pass
        
        # update rows!!!!
        self.buildRowFromCells()

    
    def buildRowFromCells(self):
        """
            build row objects and 2dzones  from cells
            
            Rowspan: create row
        """
        self._lrows=[]
        
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
#             print row.tagMe(row.tagname)
                
    def buildColumnFromCells(self):
        """
            build column objects and 2dzones  from cells
        """
        self._lcolumns= []
        for cell in self.getCells():
            _,jcol= cell.getIndex()
            try: col = self.getColumns()[jcol]
            except IndexError:
                col = XMLDSTABLECOLUMNClass(jcol)
                col = self.addColumn(col)
                if col is not None:col.setPage(self.getPage())
#                 print col.getPage(), self.getPage()
            if col is not None:col.addCell(cell)
            
#         print self.getColumns()
        
        for col in self.getColumns():
            col.resizeMe(XMLDSTABLECELLClass)
#             node = col.tagMe(col.tagname)


        
    ## SPANNING PART ###
    # must  replace buildColumnFromCells and buildRowFromCells see also getCellsbyRow?
    def buildColumnRowFromCells(self):
        """
            => rather 'grid' table: (segment)
        """
#         print 'nb cells:' , len(self.getCells())
        # first despan RowSpan cells
        for cell in self.getCells():
            # create new non-spanned cell if needed
#             print cell, cell.getRowSpan(), cell.getColSpan()
            iRowSpan = 1
            while iRowSpan < cell.getRowSpan():
#                 print 'row:', cell, cell.getRowSpan(),iRowSpan
                newCell = XMLDSTABLECELLClass(cell.getNode())
                newCell.setName(XMLDSTABLECELLClass.name)
                newCell.setPage(self.getPage())
                newCell.setObjectsList(cell.getObjects())
                newCell._lAttributes = cell.getAttributes().copy()
                newCell.addAttribute('rowSpan',1)
                newCell.setIndex(cell.getIndex()[0]+iRowSpan, cell.getIndex()[1])
                newCell.setSpannedCell(cell)
                cell.setSpannedCell(cell)
                newCell.bDeSpannedRow = True
                self.addCell(newCell)
                iRowSpan +=1
        # col span
        for cell in self.getCells():
            # create new non-spanned cell if needed
            iColSpan = 1
            while iColSpan < cell.getColSpan():
#                 print 'col:', cell, cell.getColSpan(), iColSpan
                newCell = XMLDSTABLECELLClass(cell.getNode())
                newCell.setName(XMLDSTABLECELLClass.name)
                newCell._lAttributes = cell.getAttributes().copy()
                newCell.setObjectsList(cell.getObjects())
                newCell.addAttribute('colSpan',1)
                newCell.setIndex(cell.getIndex()[0], cell.getIndex()[1]+iColSpan)
                newCell.setSpannedCell(cell)
                cell.setSpannedCell(cell)
                newCell.bDeSpannedCol = True
                self.addCell(newCell)
#                 print '\tnex col cell:',newCell, iColSpan+1, cell.getColSpan()
                iColSpan +=1
#         print '-- nb cells:', len(self.getCells())
        
    def assignElementsToCells(self):
        """
            assign elements (textlines) to the right cell 
            assuming rows and columns are ok
            
            see IE_test mergeLineAndCells
        """
        #where to get all elements? take the elements of the page?
    
    
    def buildNDARRAY(self):
#         print self.getRows()
#         print self.getColumns()
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
        self.setName(domNode.name)
        self.setNode(domNode)
        # get properties
        prop = domNode.properties
        while prop:
            self.addAttribute(prop.name,prop.getContent())
            # add attributes
            prop = prop.next
            
        ctxt = domNode.doc.xpathNewContext()
        ctxt.setContextNode(domNode)
        ldomElts = ctxt.xpathEval('./%s'%(ds_xml.sCELL))
        ctxt.xpathFreeContext()
        for elt in ldomElts:
            myObject= XMLDSTABLECELLClass(elt)
            self.addCell(myObject)
            myObject.setPage(self.getPage())
            myObject.fromDom(elt)   
        
        self._lspannedCells = self._lcells[:]
        
        self.buildColumnRowFromCells()
        self.buildColumnFromCells()
        self.buildRowFromCells()
        self.getCellsbyRow()
#         self.displayPerRow()
#         print  self.getNbRows(), self.getNbColumns()
        
    
    