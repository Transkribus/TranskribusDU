# -*- coding: latin-1 -*-
"""

    XML object class 
    
    Hervé Déjean
    cpy Xerox 2009
    
    a class for object from a XMLDocument

"""
from __future__ import absolute_import
from __future__ import  print_function
from __future__ import unicode_literals

from .XMLDSObjectClass import XMLDSObjectClass

class  XMLDSTOKENClass(XMLDSObjectClass):
    """
        LINE class
    """
    def __init__(self,domNode = None):
        XMLDSObjectClass.__init__(self)
        XMLDSObjectClass.id += 1
        self._domNode = domNode
    
    def fromDom(self,domNode):
        """
            only contains TEXT?
        """
        
        # must be PAGE        
        self.setName(domNode.tag)
        self.setNode(domNode)
        # get properties
        for prop in domNode.keys():
            self.addAttribute(prop,domNode.get(prop))        

        self.addContent(domNode.text)
        
        self.addAttribute('x2', float(self.getAttribute('x'))+self.getWidth())
        self.addAttribute('y2',float(self.getAttribute('y'))+self.getHeight() )                
            
        
#     def getSetOfListedAttributes(self,TH,lAttributes,myObject):
#         """
#             from TEXT
#         """ 
        
    def getSetOfListedAttributes(self,TH,lAttributes,myObject):
        """
            Generate a set of features: X start of the lines
            
            
        """
        from spm.feature import featureObject
     
        if self._lBasicFeatures is None:
            self._lBasicFeatures = []
        # needed to keep canonical values!
        elif self.getSetofFeatures() != []:
            return self.getSetofFeatures()
               
        lHisto = {}
        for elt in self.getAllNamedObjects(myObject):
            for attr in lAttributes:
                try:lHisto[attr]
                except KeyError:lHisto[attr] = {}
                if elt.hasAttribute(attr):
#                     if elt.getWidth() >500:
#                         print elt.getName(),attr, elt.getAttribute(attr) #, elt.getNode()
                    try:
                        try:lHisto[attr][round(float(elt.getAttribute(attr)))].append(elt)
                        except KeyError: lHisto[attr][round(float(elt.getAttribute(attr)))] = [elt]
                    except TypeError:pass
        
        for attr in lAttributes:
            for value in lHisto[attr]:
#                 print attr, value, lHisto[attr][value]
                if  len(lHisto[attr][value]) > 0.1:
                    ftype= featureObject.NUMERICAL
                    feature = featureObject()
                    feature.setName(attr)
#                     feature.setName('f')
                    feature.setTH(TH)
                    feature.addNode(self)
                    feature.setObjectName(self)
                    feature.setValue(value)
                    feature.setType(ftype)
                    self.addFeature(feature)
    
    
        if 'text' in lAttributes:
            if len(self.getContent()):
                ftype= featureObject.EDITDISTANCE
                feature = featureObject()
#                     feature.setName('content')
                feature.setName('f')
                feature.setTH(90)
                feature.addNode(self)
                feature.setObjectName(self)
                feature.setValue(self.getContent().split()[0])
                feature.setType(ftype)
                self.addFeature(feature)            
        
        if 'tokens' in lAttributes:
            if len(self.getContent()):
                for token in self.getContent().split():
                    if len(token) > 4:
                        ftype= featureObject.EDITDISTANCE
                        feature = featureObject()
                        feature.setName('token')
                        feature.setTH(TH)
                        feature.addNode(self)
                        feature.setObjectName(self)
                        feature.setValue(token.lower())
                        feature.setType(ftype)
                        self.addFeature(feature)         
        
        if 'xc' in lAttributes:
            ftype= featureObject.NUMERICAL
            feature = featureObject()
#                 feature.setName('xc')
            feature.setName('xc')
            feature.setTH(TH)
            feature.addNode(self)
            feature.setObjectName(self)
            feature.setValue(round(self.getX()+self.getWidth()/2))
            feature.setType(ftype)
            self.addFeature(feature) 
#           
        if 'virtual' in lAttributes:
            ftype= featureObject.BOOLEAN
            feature = featureObject()
            feature.setName('f')
            feature.setTH(TH)
            feature.addNode(self)
            feature.setObjectName(self)
            feature.setValue(self.getAttribute('virtual'))
            feature.setType(ftype)
            self.addFeature(feature)
                            
        if 'bl' in lAttributes:
            for inext in self.next:
                ftype= featureObject.NUMERICAL
                feature = featureObject()
                baseline = self.getBaseline()
                nbl = inext.getBaseline()
                if baseline and nbl:
                    feature.setName('bl')
                    feature.setTH(TH)
                    feature.addNode(self)
                    feature.setObjectName(self)
                    # avg of baseline?
                    avg1= baseline.getY()+(baseline.getY2() -baseline.getY())/2
                    avg2= nbl.getY() +(nbl.getY2()-nbl.getY())/2

                    feature.setValue(round(abs(avg2-avg1)))
                    feature.setType(ftype)
                    self.addFeature(feature)
            
        if 'linegrid' in lAttributes:
            #lgridlist.append((ystart,rowH, y1,yoverlap))
            for ystart,rowh,_,_  in self.lgridlist:
                ftype= featureObject.BOOLEAN
                feature = featureObject()
                feature.setName('linegrid%s'%rowh)
                feature.setTH(TH)
                feature.addNode(self)
                feature.setObjectName(self)
                feature.setValue(ystart)
                feature.setType(ftype)
                self.addFeature(feature) 
            
        return self.getSetofFeatures()        