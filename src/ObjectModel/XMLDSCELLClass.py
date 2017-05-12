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

from XMLDSObjectClass import XMLDSObjectClass
from XMLDSTEXTClass import XMLDSTEXTClass
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
        self._colSpan= 1
        self._rowSpan= 1

        self.tagName=ds_xml.sCELL
        # contains a list of fields (tag)
        self._lFields = []
    
    def __str__(self):
        return "%s %s"%(self.getName(),self.getIndex())    
    
    def getIndex(self): return self._index
    def setIndex(self,i,j): self._index = (i,j)
    def getColSpan(self): return self._colSpan
    def getRowSapn(self): return self._rowSpan
    
    def getFields(self): return self._lFields
    
    def addField(self,field):
        """
            self is supposed to contain field (record field)
        """
        if field not in self.getFields():
            self.getFields().append(field)
        return field
    
    ############# TAGGING
    
    
    ### dom tagging
    def tagMe2(self):
        self.tagMe()
        self.getNode().setProp('row',str(self.getIndex()[0]))
        self.getNode().setProp('col',str(self.getIndex()[1]))        
    
    ########### LOAD FROM DSXML ################
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
        self.setIndex(int(self.getAttribute('row')),int(self.getAttribute('col')))
        ctxt = domNode.doc.xpathNewContext()
        ctxt.setContextNode(domNode)
        
        # onlt sTEXT?
        ldomElts = ctxt.xpathEval('./%s'%(ds_xml.sTEXT))
        ctxt.xpathFreeContext()
        for elt in ldomElts:
            myObject= XMLDSTEXTClass(elt)
            self.addObject(myObject)
            myObject.setPage(self.getPage())
            myObject.fromDom(elt)
        
         
        
