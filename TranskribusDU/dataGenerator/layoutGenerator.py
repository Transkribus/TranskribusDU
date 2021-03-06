# -*- coding: utf-8 -*-
"""


    layoutGenerator.py

    create (generate)  annotated data (document layout) 
     H. Déjean
    

    copyright Naver labs Europe 2017
    READ project 

    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
"""
from __future__ import absolute_import
from __future__ import  print_function
from __future__ import unicode_literals
from  lxml import etree

try:basestring
except NameError:basestring = str

import numpy as np
from dataGenerator.generator import Generator 
from dataGenerator.numericalGenerator import numericalGenerator


class layoutZoneGenerator(Generator):
    
    def __init__(self,config,configKey=None,x=None,y=None,x2= None,y2=None,h=None,w=None,alpha=0):
        Generator.__init__(self,config,configKey)
        if x is None:
            self._x  = numericalGenerator(None,None)
            self._x.setLabel("x")
            self._y  = numericalGenerator(None,None)
            self._y.setLabel("y")
            self._h  = numericalGenerator(None,None)
            self._h.setLabel("height")   
            self._w  = numericalGenerator(None,None)
            self._w.setLabel("width")
            
            self._x2  = numericalGenerator(None,None)
            self._x2.setLabel("x2")
            self._y2  = numericalGenerator(None,None)
            self._y2.setLabel("y2")
        else:
            self._x = x
            self._y = y
            self._x2 = x2
            self._y2 = y2            
            self._h = h
            self._w = w
            
        # text orientation
        self._rotationAngle = alpha

        # keep track of the last positions for further content 
        self._lastYposition = None 
        self._lastXPosition = None
        
        self._page= None
        
        # correspond to _structure
        self._structure = [
                            [ (self.getX(),1,100),(self.getY(),1,100),(self.getX2(),1,100),(self.getY2(),1,100),(self.getHeight(),1,100),(self.getWidth(),1,100),100]
                            ]
        
        # standard deviation used for numerical values
        self._TH = None
        
    def getX(self): return self._x
    def getY(self): return self._y
    def getX2(self): return self._x2
    def getY2(self): return self._y2
    def getHeight(self): return self._h
    def getWidth(self): return self._w
    
    
    def getObjects(self): return self._structure
    
    def setX(self,v): self._x.setUple(v)
    def setY(self,v): self._y.setUple(v)
    def setX2(self,v): self._x2.setUple(v)
    def setY2(self,v): self._y2.setUple(v)    
    def setHeight(self,v): self._h.setUple(v)
    def setWidth(self,v): self._w.setUple(v)

    def setPositionalGenerators(self,x,y,x2,y2,h,w):
        if x is not None:self.setX(x)
        if x2 is not None:self.setX2(x2)
        if y is not None:self.setY(y)
        if y2 is not None:self.setY2(y2)
        if h is not None:self.setHeight(h)
        if w is not None:self.setWidth(w)
        # x2 , y2
    
    def getPositionalGenerators(self):
        return self.getX(), self.getY(),self.getX2(), self.getY2(), self.getHeight(), self.getWidth()


    def getPoints(self, angle):
        """
            return a 'points' representation with skew angle = angle
            use x2,y2?
        """
        if self.getX()._generation is None:
            return None
        theta= np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c,-s), (s, c)))  
        xy = [self.getX()._generation,self.getY()._generation]
        xx,yy = np.dot(R,xy)
        xy2 = [self.getX()._generation + self.getWidth()._generation,self.getY()._generation]
        xx2,yy2 = np.dot(R,xy2)   
        xy3 = [self.getX()._generation + self.getWidth()._generation,self.getY()._generation+self.getHeight()._generation]
        xx3,yy3 = np.dot(R,xy3)   
        xy4 = [self.getX()._generation,self.getY()._generation+self.getHeight()._generation]
        xx4,yy4 = np.dot(R,xy4)   
        return (xx,yy,xx2,yy2,xx3,yy3,xx4,yy4)
          
    def setPage(self,p): self._page = p
    def getPage(self):return self._page
        
    def updateStructure(self,Gen):
        """ 
        add Gen in structure
        """
        for struct in self._structure:
            struct.insert(-1,Gen)
            
            
    def PageXmlFormatAnnotatedData(self,linfo,obj):
        """
            PageXML export format 
        """
        self.domNode = etree.Element(obj.getLabel())
        # for listed elements
        if obj.getNumber() is not None:
            self.domNode.set('number',str(obj.getNumber()))        
        for info,tag in linfo:
            if isinstance(tag,Generator):
                self.domNode.append(tag.XMLDSFormatAnnotatedData(info,tag))
            else:
                self.domNode.set(tag,str(info))
        
        return self.domNode
    
    
    def XMLDSFormatAnnotatedData(self,linfo,obj):
        """
            here noiseGenerator? 
        """
        self.domNode = etree.Element(obj.getLabel())
        
        # for listed elements
        if obj.getNumber() is not None:
            self.domNode.set('number',str(obj.getNumber()))      
            
        # points here??
#         try:
#             self.domNode.set('points',"%f,%f %f,%f %f,%f %f,%f" % self.getPoints(2))
#         except:pass
        for info,tag in linfo:
            if isinstance(tag,Generator):
                self.domNode.append(tag.XMLDSFormatAnnotatedData(info,tag))
            else:
                self.domNode.set(tag,str(info))
        
        return self.domNode    
        
    def exportAnnotatedData(self,foo):
        """
             build a full version of generation: integration of the subparts (subtree)
        """
        self._GT=[]
        for obj in self._generation:
            if obj:
                if isinstance(obj._generation,basestring):
                    self._GT.append((obj._generation,obj.getLabel()))
                elif type(obj._generation) in [int,float]:
                    self._GT.append((obj._generation,obj.getLabel()))
                else:        
                    if obj is not None:
                        #
#                         obj.generate()
                        self._GT.append( (obj.exportAnnotatedData([]),obj))
    
        return self._GT           


            
    def serialize(self):
        raise Exception( 'must be instantiated')
    

    
    
if __name__ == '__main__':
    TH=30
    myZone = layoutZoneGenerator({},'',numericalGenerator(5,TH),numericalGenerator(30,TH),numericalGenerator(20,TH),numericalGenerator(100,TH))
    myZone.instantiate()
    myZone.generate()
    print(myZone._generation)
    
    myZone.setLabel('TEXT')
    print(myZone.exportAnnotatedData([])) 
    
    
    