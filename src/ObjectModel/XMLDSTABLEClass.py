# -*- coding: utf-8 -*-
"""

    XMLDS TABLE
    Hervé Déjean
    cpy Xerox 2013
    
    a class for table from a XMLDocument

"""

from XMLDSObjectClass import XMLDSObjectClass
from XMLDSCELLClass import XMLDSTABLECELLClass
from XMLDSColumnClass import XMLDSTABLECOLUMNClass
from XMLDSRowClass import XMLDSTABLEROWClass

from config import ds_xml_def as ds_xml
from astropy.wcs.docstrings import row

class  XMLDSTABLEClass(XMLDSObjectClass):
    """
        LINE class
    """
    name = ds_xml.sLINE_Elt
    def __init__(self,domNode = None):
        XMLDSObjectClass.__init__(self)
        XMLDSObjectClass.id += 1
        
        self._domNode = domNode
    
        self._lcells=[]
        self._lcolumns= []
        self._lrows = []
    
    def getCells(self): return self._lcells
    
    
    def addCell(self,cell):
        """
            add a  cell
            update row and col data structure? no 
        """
        self._lcells.append(cell)
        self.addObject(cell)


    def getColumns(self): return self._lcolumns
    def getRows(self): return self._lrows
    
    def addRow(self,row):
        if len(self.getRows()) == row.getIndex():
            self._lrows.append(row)
        else:
            return None
        return row
    
      
    def addColumn(self,col):
        if len(self._lcolumns) == col.getIndex():
            self._lcolumns.append(col)
        else:
            return None
        return col

    def buildRowFromCells(self):
        """
            build row objects and 2dzones  from cells
        """
        for cell in self.getCells():
            irow,_= cell.getIndex()
            try: row = self.getRows()[irow]
            except IndexError:
                row = XMLDSTABLEROWClass(irow)
                row = self.addRow(row)
                row.setPage(self.getPage())
            row.addCell(cell)
        for row in self.getRows():
            row.resizeMe(XMLDSTABLECELLClass)
            row.tagMe(row.tagname)
                
    def buildColumnFromCells(self):
        """
            build column objects and 2dzones  from cells
        """
        for cell in self.getCells():
            _,jcol= cell.getIndex()
            try: col = self.getColumns()[jcol]
            except IndexError:
                col = XMLDSTABLECOLUMNClass(jcol)
                col = self.addColumn(col)
                col.setPage(self.getPage())
            col.addCell(cell)
        for col in self.getColumns():
            col.resizeMe(XMLDSTABLECELLClass)
            col.tagMe(col.tagname)
    
    
    def deleteRows(self,iStart,iEnd):
        """
            delete rows[iStart:iEnd]
            merge cells  
        """
        
    
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

    
    