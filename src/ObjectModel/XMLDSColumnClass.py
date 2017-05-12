# -*- coding: utf-8 -*-
"""

    XMLDS COLUMN
    Hervé Déjean
    cpy Xerox 2017
    
    a class for table column from a XMLDocument

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
from config import ds_xml_def as ds_xml

class  XMLDSTABLECOLUMNClass(XMLDSObjectClass):
    """
        LINE class
    """
    name = ds_xml.sCOL_Elt
    def __init__(self,index=None,domNode = None):
        XMLDSObjectClass.__init__(self)
        XMLDSObjectClass.id += 1
        self._domNode = domNode
        self._index= index
        self._lcells=[]
        self.tagName= 'COL'
        self.setName(XMLDSTABLECOLUMNClass.name)
    
    def __str__(self):
        return "%s %s"%(self.getName(),self.getIndex())        
    
    def getIndex(self): return self._index
    def setIndex(self,i): self._index = i
    
    def getCells(self): return self._lcells
    def addCell(self,c): 
        if c not in self.getCells():
            self._lcells.append(c)
            self.addObject(c)                

    ########## LABELLING ##############
    def addField(self,field):
        [cell.addField(field) for cell in self.getCells()]





     
#     ## possible
#     def fromDom(self,domNode):
#         """
#             only contains TEXT?
#         """
#         self.setName(domNode.name)
#         self.setNode(domNode)
#         # get properties
#         prop = domNode.properties
#         while prop:
#             self.addAttribute(prop.name,prop.getContent())
#             # add attributes
#             prop = prop.next
#         self.setIndex(int(self.getAttribute('irow')),int(self.getAttribute('icol')))
#             
#         ctxt = domNode.doc.xpathNewContext()
#         ctxt.setContextNode(domNode)
#         ldomElts = ctxt.xpathEval('./%s'%(ds_xml.sCELL))
#         ctxt.xpathFreeContext()
#         for elt in ldomElts:
#             myObject= XMLDSTEXTClass(elt)
#             self.addObject(myObject)
#             myObject.setPage(self.getPage())
#             myObject.fromDom(elt)
        
         
        
