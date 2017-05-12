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
    
        self._lcells=[]
        # need to define dim later on
        self._npcells=None
        
        self._lcolumns= []
        self._lrows = []
    
        self._nbCols= None
        self._nbRows = None
    
    
    def getCells(self): return self._lcells
    
    
    def addCell(self,cell):
        """
            add a  cell
            update row and col data structure? no 
        """
        self._lcells.append(cell)
#         self._npcells[cell.getIndex()]
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
                print cell
            print
                
    
      
    def addColumn(self,col):
        if len(self._lcolumns) == col.getIndex():
            self._lcolumns.append(col)
        else:
            return None
        return col
    
    
    def createRowsWithCuts(self,lYCuts):
        """
            input: a table and cells
            oupput: list of rows populatee with appropriate cells  (main overlap)
        """
        
        #reinit rows?
        lCells =self.getCells()
        prevCut = self.getY()
        lYCuts.sort(key = lambda x:x.getValue())
        for irow,cut in enumerate(lYCuts):
            row= XMLDSTABLEROWClass(irow)
            row.setParent(self)
            row.addAttribute('y',prevCut)
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
    
    def buildRowFromCells(self):
        """
            build row objects and 2dzones  from cells
        """
        self._lrows=[]
        
        for cell in self.getCells():
            irow,_= cell.getIndex()
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
            if col is not None:col.addCell(cell)
        for col in self.getColumns():
            col.resizeMe(XMLDSTABLECELLClass)
            col.tagMe(col.tagname)
    
    
    
    def buildNDARRAY(self):
        # dtype=np.dtype(XMLDSTABLECELLClass)
        self._npcells = np.reshape(self.getCells(),(self.getNbRows(),self.getNbColumns()))
    
    ###############  TAGGING METHOD  ######################
    
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
        
    ## see COLUMN, ROW, CELL Object
    
    
    ###############  LOAD FROM DSXML ######################    
    
    def fromDom(self,domNode):
        """
            only contains TEXT?
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
        
        self.buildColumnFromCells()
        self.buildRowFromCells()

    
    