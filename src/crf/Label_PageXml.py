# -*- coding: utf-8 -*-

"""
    Defining the labels of a graph for a PageXml document
    

    Copyright Xerox(C) 2016 JL. Meunier

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
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""

from Label import Label
from xml_formats.PageXml import PageXml
 
class Label_PageXml(Label):

    #where the labels can be found in the data
    sCustAttr_STRUCTURE     = "structure"
    sCustAttr2_TYPE         = "type"
    
    def parseNodeLabel(self, nd):
        """
        Parse the graph node label and return its class index
        """
        try:
            sLabel = PageXml.getCustomAttr(nd.node, self.sCustAttr_STRUCTURE, self.sCustAttr2_TYPE)
        except KeyError:
            sLabel = self._sOTHER
        cls = self.dClsByLabel[sLabel]        
        return cls

def test_init():
    #Forcing some TASK SPECIFIC labels
    Label_PageXml._lsLabel = ['catch-word', 'header', 'heading', 'marginalia', 'page-number']
    obj = Label_PageXml()
    assert obj.dLabelByCls['header'] == 2
    assert obj.dClsByLabel[2] == 'header'