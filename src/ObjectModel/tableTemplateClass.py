# -*- coding: utf-8 -*-
"""

    Table Template

    Hervé Déjean
    cpy Xerox 2017
    
    READ project
    
    
    
"""
from templateClass import templateClass

import numpy as np
from ObjectModel.XMLDSTABLEClass import XMLDSTABLEClass

class tableTemplateClass(templateClass):
    """

        table template class
        
        Describes a table regarding an IE extraction problem:

        what is the pattern?
        a list of tagging instruction for cells
            colheader = 1
            rowheader = -1
       
       
       
        
        For table, but a list (and sublist) can be organized with a table (and then ndarray syntax can be used for decoration) 
             -> after 
    """
    
    def __init__(self):
        templateClass.__init__(self)
        
        # list of labelling instruction
        ## instrution: slice, fiedname
        self._lLabellingInstruction = []
        
    def __str__(self):return 'tableTemplate'
    def __repr__(self):return 'tableTemplate'



    @classmethod
    def string2slice(cls,s):
        """
            convert a string into slice
            string: 
        """
        lx = s.split(":")
        if len(lx) == 2:
            start=lx[0]
            stop=lx[1]
            step=None
        elif len(lx) == 3:
            start=lx[0]
            stop=lx[1]
            step=lx[2]            
        ss = slice(start,stop,step)
        return ss
        
    
    
        
    def buildFromPattern(self,p):
        """
            a small language ? a la numpy
            ([:],(fieldname,value))
             ([:],(fieldname,))
            ([:,2],(fieldname,value))
             
            ex: column 2(+1) , row ingoring first row
            ([1:,2],(fieldname,None))
                         
            fieldname: headers
            
            tag at cell level: add field to cell.getFields()
            
            
            at full column/row? yes for the moment
           
           date 
           page 1 ([:],(date,1870))
           page 2 [2:], ( date,1870))
           page 2 [3:], ( date,1871))

           page 1 ([:1], ((firstname,None) , (lastname,None))) 
           page 1 ([1:1], ((firstname,Helmut) , (lastname,Goetz)))   # for GT only?

        """
        ## reformalute the pattern with complete values: start:stop:step for each  dimension (2D)
        ## split with ','  : then by :   if just ':' NoneNoneNone,   

        for index,lFields in p:
            print index, lFields
            self._lLabellingInstruction.append((index,lFields))
            
    def labelTable(self,table):
        """
            use template to label cells
            
            
        Remember that a slicing tuple can always be constructed as obj and used in the x[obj] notation. 
        Slice objects can be used in the construction in place of the [start:stop:step] notation. For example, x[1:10:5,::-1] can also be implemented as obj = (slice(1,10,5), slice(None,None,-1)); x[obj] . 
        This can be useful for constructing generic code that works on arrays of arbitrary dimension.            
            
        """
        
        for sslice, lFields in self._lLabellingInstruction:
            for field in lFields:
                if field is not None:
                    for cell in np.nditer(table.getNPArray()[sslice],['refs_ok'],op_dtypes=np.dtype(object)):
                        cell[()].addField(field.cloneMe())
        
    def registration(self,o):
        raise "SOFTWARE ERROR: your component must define a testRun method"


    def describeMe(self):
        """
            a plain text description of this template 
        """
        raise "SOFTWARE ERROR: your component must define a testRun method"
        
    def tagDom(self,dom):
        raise "SOFTWARE ERROR: your component must define a testRun method" 
    
    
# --- AUTO-TESTS --------------------------------------------------------------------------------------------
#def test_template():
if __name__ == "__main__":
    from XMLDSCELLClass import XMLDSTABLECELLClass
    from recordClass import fieldClass
    table=XMLDSTABLEClass()
    
    cell1=XMLDSTABLECELLClass()
    cell1.setIndex(0, 0)
    
    cell2=XMLDSTABLECELLClass() 
    cell2.setIndex(0, 1)
    cell2.setContent('Maria Schmidt')
    cell3=XMLDSTABLECELLClass()
    cell3.setIndex(1, 0)    
    cell3.setContent('kindlein')
    
    cell4=XMLDSTABLECELLClass()
    cell4.setIndex(1, 1)    
    
    table.addCell(cell1)
    table.addCell(cell2)
    table.addCell(cell3)
    table.addCell(cell4)
    table.buildColumnFromCells()
    table.buildRowFromCells()
    table.buildNDARRAY()
    print table.getNPArray()
    # ([1:,2],(fieldname,None))
    stemplate='[((slice(1,None),slice(0,1)) ,["name", "fistname"]),((slice(1,2),slice(1,None)) ,["ledig"])]'
    
    myTemplate=tableTemplateClass()
    
    # build ltemplate with record
    ltemplate = eval(stemplate)
    myTemplate.buildFromPattern(ltemplate)
    myTemplate.labelTable(table)
    
    for cell in table.getCells():
        print cell.getIndex(), cell.getFields()
    
    ## extract data usinf field information
    
    field = fieldClass('name')

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
          