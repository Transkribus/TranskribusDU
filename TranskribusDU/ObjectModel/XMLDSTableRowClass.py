# -*- coding: utf-8 -*-
"""

    XMLDS ROW
    Hervé Déjean
    cpy Xerox 2017
    
    a class for table row from a XMLDocument

"""
from __future__ import absolute_import
from __future__ import  print_function
from __future__ import unicode_literals

from .XMLDSObjectClass import XMLDSObjectClass
from config import ds_xml_def as ds_xml

class  XMLDSTABLEROWClass(XMLDSObjectClass):
    """
        LINE class
    """
    name = ds_xml.sROW
    def __init__(self,index=None,domNode = None):
        XMLDSObjectClass.__init__(self)
        XMLDSObjectClass.id += 1
        self._domNode = domNode
        self.tagName = 'ROW'
        self._index= index
        self._lcells=[]
        self.setName(XMLDSTABLEROWClass.name)
    
    def __repr__(self):
        return "%s %s"%(self.getName(),self.getIndex())  
    def __str__(self):
        return "%s %s"%(self.getName(),self.getIndex())          
        
    def getID(self): return self.getIndex()
    def getIndex(self): return self._index
    def setIndex(self,i): self._index = i
    
    def getCells(self): return self._lcells
    def addCell(self,c): 
        if c not in self.getCells():
            self._lcells.append(c)            
            self.addObject(c)
#             if c.getNode() is not None and self.getNode() is not None:
#                 c.getNode().unlinkNode()
#                 self.getNode().addChild(c.getNode())


    def computeSkewing(self):
        """
            input: self
            output: skewing ange
            compute text skewing in the row 
        """
        def getX(lSegment):
            lX = list()
            for x1,y1,x2,y2 in lSegment:
                lX.append(x1)
                lX.append(x2)
            return lX        
        
        def getY(lSegment):
            lY = list()
            for x1,y1,x2,y2 in lSegment:
                lY.append(y1)
                lY.append(y2)
            return lY
                
        import numpy as np
        from util.Polygon import Polygon
        if self.getCells():
            dRowSep_lSgmt=[]
            # alternative: compute the real top ones? for wedding more robust!!
            for cell in self.getCells():
#                 lTopText = filter(lambda x:x.getAttribute('DU_row') == 'B', [text for text in cell.getObjects()])
                try:lTopText = [cell.getObjects()[0]]
                except IndexError:lTopText = []
                for text in lTopText:
                    sPoints = text.getAttribute('points')
                    spoints = ' '.join("%s,%s"%((x,y)) for x,y in zip(*[iter(sPoints.split(','))]*2))
                    it_sXsY = (sPair.split(',') for sPair in spoints.split(' '))
                    plgn = Polygon((float(sx), float(sy)) for sx, sy in it_sXsY)
                    
                    lT, lR, lB, lL = plgn.partitionSegmentTopRightBottomLeft()
                    dRowSep_lSgmt.extend(lT)
            if dRowSep_lSgmt != []:
                X = getX(dRowSep_lSgmt)
                Y = getY(dRowSep_lSgmt)
                lfNorm = [np.linalg.norm([[x1,y1], [x2,y2]]) for x1,y1,x2,y2 in dRowSep_lSgmt]
                #duplicate each element 
                W = [fN for fN in lfNorm for _ in (0,1)]
        
                # a * x + b
                a, b = np.polynomial.polynomial.polyfit(X, Y, 1, w=W)
                xmin, xmax = min(X), max(X)
                y1 = a + b * xmin
                y2 = a + b * xmax
                ro = XMLDSTABLEROWClass(self.getIndex())
                #[(x1, ymin), (x2, ymax)
                ro.setX(xmin)
                ro.setY(y1)
                ro.setWidth(xmax-xmin)
                ro.setHeight(y2-y1)
                ro.setPage(self.getPage())
                ro.setParent(self.getParent())
                ro.addAttribute('points',','.join([str(xmin),str(y1),str(xmax),str(y2)]))
                ro.tagMe()
        


    
    ########## TAGGING ##############
    def addField(self,tag):
        [cell.addField(tag) for cell in self.getCells()]


        
