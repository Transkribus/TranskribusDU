# -*- coding: utf-8 -*-

"""
    a type of nodes 
    
    It defines:
    - the labels of this node type
    - how to read and write the XML corresponding to this node type
    

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

class NodeType:

    #The labels for those graphs
    _sOTHER_LABEL   = "OTHER"
    lsLabel         = None #list of labels
    sDefaultLabel   = None #when no annotation, do we set automatically to this label? (e.g."OTHER")
    dLabelByCls     = None    
    dClsByLabel     = None
    nCls            = None
    
    def __init__(self, sNodeTypeName, lsLabel, lsIgnoredLabel=None, bOther=True):
        """
        Those labels MUST BE THE SAME AS FOUND IN THE DATASET !!
        
        We get a list of label of interest and a list of labels that must be ignored, and become OTHER
        
        Note: set lsIgnoredLabel to "*" to accept anything of no interest!
        
        *** Any other observed label will raise a (detailed) ValueError exception. ***
        
        if bOther is True, absence of label become OTHER, otherwise, ValueError exception 
        
        set those properties:
            self.lsLabel    - list of label names
            dLabelByCls     - dictionary name -> id
            dClsByLabel     - dictionary id -> name
            self.nCls       - number of different labels
        """
        self.name = sNodeTypeName.strip()
        self._n = len(self.name)
        
        self.lsXmlLabel        = lsLabel
        self.lsXmlIgnoredLabel = lsIgnoredLabel
            
        self.lsLabel        = [self.getInternalLabelName(s) for s in lsLabel]

        self.dXmlLabel2Label = { sXml :s    for sXml, s in zip(self.lsXmlLabel, self.lsLabel) }
        self.dLabel2XmlLabel = { s    :sXml for sXml, s in zip(self.lsXmlLabel, self.lsLabel) }

        if lsIgnoredLabel or bOther:
            #we need to deal with a class "other"
            self.sDefaultLabel = "%s_%s"%(self.name, self._sOTHER_LABEL)
            self.lsLabel = [self.sDefaultLabel] + self.lsLabel
            
            self.dIgnoredXmlLabel = { sXml :True    for sXml, s in self.lsXmlIgnoredLabel } if self.lsXmlIgnoredLabel else None
        else:
            self.sDefaultLabel  = None

        self.nCls = len(self.lsLabel)        

    def checkIsIgnored(self, sXmlLabel):
        """
        return True or raises an exception
        """
        return self.dIgnoredXmlLabel[sXmlLabel]
     
    def getInternalLabelName(self, sXmlLabel):
        """
        return the name seen by the classifier from the user-visible label
        """
        return "%s_%s"%(self.name, sXmlLabel.strip())
    
    def setXpathExpr(self, o):
        """
        set any Xpath related information to extract the nodes from an XML file
        """
        raise Exception("Method must be overridden")
    
    def getLabelNameList(self):
        """
        Return the list of label known to the classifier (slightly different from their XML value)
        """
        return self.lsLabel

    @classmethod
    def parseDomNodeLabel(cls, domnode):
        """
        return the internal label of the DOM node
        raise a ValueError if the label is missing while bOther was not True, or if the label is neither a valid one nor an ignored one
        """
        raise Exception("Method must be overridden")

    @classmethod
    def setDomNodeLabel(cls, node, sLabel):
        """
        Set the DOM node label in the format-dependent way
        """
        raise Exception("Method must be overridden")

    def _iter_GraphNode(self, doc, domNdPage, page):
        """
        Parse a DOM page
        
        Get the DOM, the DOM page node, the page object

        iterator on the DOM, that returns nodes  (of class Block)
        """    
        raise Exception("Method must be overridden")
                        
    def _get_GraphNodeText(self, doc, domNdPage, domNd, ctxt=None):
        """
        Extract the text of a DOM node
        
        Get the DOM, the DOM page node, the page object DOM node, and optionally an xpath context

        return a unicode string
        """    
        raise Exception("Method must be overridden")
