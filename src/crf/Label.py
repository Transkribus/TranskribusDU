# -*- coding: utf-8 -*-

"""
    Defining the labels of a graph for a document

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
class Label:
    
    _sOTHER = "OTHER"    #if not None, any missing label will be represented as class "OTHER" (index 0)
    _lsLabel = None      #DEFINE YOUR LABELS IN A SUB-CLASS, as a list of string
 
    def __init__(self):
        """
        set those properties:
            self.lsLabel    - list of label names
            dLabelByCls     - dictionary name -> id
            dClsByLabel     - dictionary id -> name
            self.nCls       - number of different labels
        """
        if self._sOTHER: 
            assert self._sOTHER not in self._lsLabel, "the label for class 'OTHER' conflicts with a task-specific label"
            self.lsLabel = [self._sOTHER] + self._lsLabel
        else:
            self.lsLabel = self._lsLabel
            
        self.dLabelByCls = { sLabel:i for i,sLabel in enumerate(self.lsLabel) }          
        self.dClsByLabel = { i:sLabel for i,sLabel in enumerate(self.lsLabel) }
        self.nCls = len(self.lsLabel)

    def parseNodeLabel(self, node):
        """
        Parse the graph node label and return its class index
        """
        raise Exception("Method must be overridden")
        
    def parseLabels(self, lNode):
        """
        Parse the label of each graph node from teh dataset, and set the node label
        return the set of observed class (set of integers in N+)
        """
        setSeensLabels = set()
        for nd in lNode:
            cls = self.parseNodeLabel(nd)
            nd.cls = cls
            setSeensLabels.add(cls)
        return setSeensLabels

