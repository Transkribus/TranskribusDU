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



from  lxml import etree

try:basestring
except NameError:basestring = str

from .generator import Generator 
from .numericalGenerator import numericalGenerator


class layoutZoneGenerator(Generator):
    def __init__(self,x=None,y=None,h=None,w=None):
        Generator.__init__(self)
        if x is None:
            self._x  = numericalGenerator(None,None)
            self._x.setLabel("x")
            self._y  = numericalGenerator(None,None)
            self._y.setLabel("y")
            self._h  = numericalGenerator(None,None)
            self._h.setLabel("height")   
            self._w  = numericalGenerator(None,None)
            self._w.setLabel("width")
        else:
            self._x = x
            self._y = y
            self._h = h
            self._w = w
            
        
        self._page= None
        
        # correspond to _structure
        self._structure = [
                            [ (self.getX(),1,100),(self.getY(),1,100),(self.getHeight(),1,100),(self.getWidth(),1,100),100]
                            ]
        
        # standard deviation used for numerical values
        self._TH = None
        
    def getX(self): return self._x
    def getY(self): return self._y
    def getHeight(self): return self._h
    def getWidth(self): return self._w
    
    
    def getObjects(self): return self._structure
    
    def setX(self,v): self._x.setUple(v)
    def setY(self,v): self._y.setUple(v)
    def setHeight(self,v): self._h.setUple(v)
    def setWidth(self,v): self._w.setUple(v)

    def setPositionalGenerators(self,x,y,h,w):
        self.setX(x)
        self.setY(y)
        self.setHeight(h)
        self.setWidth(w)
    
    def setPage(self,p): self._page = p
    def getPage(self):return self._page
        
    
    def addSkewing(self,angle):
        """
            rotate with angle
        """
        
        
    
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
    myZone= layoutZoneGenerator(numericalGenerator(5,TH),numericalGenerator(30,TH),numericalGenerator(20,TH),numericalGenerator(100,TH))
    myZone.instantiate()
    myZone.generate()
    print(myZone._generation)
    
    myZone.setLabel('TEXT')
    print(myZone.exportAnnotatedData([])) 
    
    
    