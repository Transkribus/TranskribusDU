# -*- coding: utf-8 -*-
"""

    XMLDSGRAPHLINE class 
    
    Hervé Déjean
    cpy Xerox 2016
    READ project 

    
    a class for gaphical line

"""

from XMLDSObjectClass import XMLDSObjectClass

class  XMLDSGRAPHLINEClass(XMLDSObjectClass):
    """
        GRAPHLINE class
    """
    def __init__(self,domNode = None):
        XMLDSObjectClass.__init__(self)
        XMLDSObjectClass.id += 1
        
        self._domNode = domNode
        
        self.skewAngle= None
        self.lPoints = None
        self.avgY    = None
        self.length  = None

    def computePoints(self):
        """
            points= x,y,x,y,x,y
        """
        if self.lPoints is None:
            self.lPoints = self.getAttribute('points')
#             print 'after split?',self.lPoints
        if self.lPoints is not None:
            lX = map(lambda x:float(x),self.lPoints.split(','))[0::2]
            lY = map(lambda x:float(x),self.lPoints.split(','))[1::2]
            self.lPoints = zip(lX,lY)
            try:
                self.avgY = 1.0 * sum(lY)/len(lY)
            except ZeroDivisionError:
                print 'ZeroDivisionError:', lY, self.lPoints
            self.length= lX[-1]-lX[0]
            self.addAttribute('x', lX[0])
            self.addAttribute('x2', lX[-1])
            self.addAttribute('y', lY[0])
            self.addAttribute('y2', lY[-1])    
            ## take X if not far from avgY?
#             import numpy as np
#             a,b = np.polyfit(lX, lY, 1)
#             self.setAngle(a)
#             ymax = a * self.getX2() +b
#             ymin = a*self.getX() + b
#             import libxml2
#             verticalSep  = libxml2.newNode('PAGEBORDER')
#             verticalSep.setProp('points', '%f,%f,%f,%f'%(self.getX(),ymin,self.getX2(),ymax))         
#             self.getParent().getNode().addChild(verticalSep)
            
    """
        TO simulate 'DS' objects
        needed??? yes for neighborhood
        
    """            
    def getAngle(self) : return self.skewAngle
    def setAngle(self,a): self.skewAngle = a  
    def getX(self): return float(self.getAttribute('x'))
    def getY(self): return float(self.getAttribute('y')) #return self.avgY
    def getX2(self): return float(self.getAttribute('x2')) #return float(self.getAttribute('x'))+self.getWidth()
    def getY2(self): return float(self.getAttribute('y2')) #return self.avgY
    def getHeight(self): return  abs(float(self.getAttribute('height')))
    def getWidth(self): return abs(float(self.getAttribute('width')))
    
    
    def setPoints(self,lp): self.lPoints = lp
    
    def fromDom(self,domNode):
        """
            only contains points attribute
        """
        self.setName(domNode.name)
        self.setNode(domNode)
        prop = domNode.properties
        while prop:
            self.addAttribute(prop.name,prop.getContent())
            # add attributes
            prop = prop.next
        self.computePoints()

