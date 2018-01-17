# -*- coding: utf-8 -*-
"""

    XMLDS CELL
    Hervé Déjean
    cpy Xerox 2017
    
    a class for table cell from a XMLDocument
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

from .XMLDSObjectClass import XMLDSObjectClass
from .XMLDSTEXTClass import XMLDSTEXTClass
from .XMLDSLINEClass import XMLDSLINEClass
from config import ds_xml_def as ds_xml

class  XMLDSTABLECELLClass(XMLDSObjectClass):
    """
        LINE class
    """
    name = ds_xml.sCELL
    def __init__(self,domNode = None):
        XMLDSObjectClass.__init__(self)
        XMLDSObjectClass.id += 1
        self._domNode = domNode
        self._index= (0,0)
        
        self._spannedCell =None
        self.bDeSpannedRow = False
        self.bDeSpannedCol =False
        self.tagName=ds_xml.sCELL
        # contains a list of fields (tag)
        self._lFields = []
        self.setName(self.name)
        
        #default
        self.addAttribute('rowSpan', 1)
        self.addAttribute('colSpan', 1)
    
    def __str__(self):
        return "%s %s"%(self.getName(),self.getIndex())    
    def __repr__(self):
        return "%s %s"%(self.getName(),self.getIndex())       
    def getIndex(self): return self._index
    def setIndex(self,i,j): self._index = (i,j)
    def getColSpan(self): return int(self.getAttribute('colSpan'))
    def getRowSpan(self): return int(self.getAttribute('rowSpan'))
    

    def isDeSpannedRow(self): return self.bDeSpannedRow
    def isDeSpannedCol(self): return self.bDeSpannedCol
    
    def setSpannedCell(self,c): self._spannedCell = c
    def getSpannedCell(self): return self._spannedCell
    

        
    
    ### dom tagging
    def tagMe2(self):
        self.tagMe()
        self.getNode().set('row',str(self.getIndex()[0]))
        self.getNode().set('col',str(self.getIndex()[1]))        
        self.getNode().set('rowSpan',str(self.getRowSpan()))
        self.getNode().set('colSpan',str(self.getColSpan()))        
    
    ########### LOAD FROM DSXML ################
    
    def fromDom(self,domNode):
        """
            only contains TEXT?
        """
        self.setName(domNode.tag)
        self.setNode(domNode)
        # get properties
        for prop in domNode.keys():
            self.addAttribute(prop,domNode.get(prop))
            
        self.setIndex(int(self.getAttribute('row')),int(self.getAttribute('col')))
        
        # add default values if missinf
        try:  self.getRowSpan()
        except: self.addAttribute('rowSpan',"1")
        try: self.getColSpan()
        except:  self.addAttribute('colSpan',"1")
        # onlt sTEXT?
#         ldomElts = ctxt.xpathEval('./%s'%(ds_xml.sTEXT))
        ldomElts = domNode.findall('./*')

        for elt in ldomElts:
            if elt.tag ==XMLDSTEXTClass.name:
                myObject= XMLDSTEXTClass(elt)
            elif elt.tag == XMLDSLINEClass.name:
                myObject= XMLDSLINEClass(elt)
            else:
                myObject= None
            if myObject is not None:
                self.addObject(myObject)
                myObject.setPage(self.getPage())
                myObject.fromDom(elt)
        
    
        
