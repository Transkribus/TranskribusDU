# -*- coding: latin-1 -*-
"""

    XML object class 
    
    Hervé Déjean
    cpy Xerox 2009
    
    a class for object from a XMLDocument

"""

from objectClass import objectClass
import libxml2 

class  XMLObjectClass(objectClass):
    """
        object class
    """
    orderID = 0
    def __init__(self):
        objectClass.__init__(self)
        XMLObjectClass.id += 1
        self._orderedID =  XMLObjectClass.id

        self._domNode = None
        self.tagname="MYTAG"
        
        # needed for mapping dataobject to layoutObject ???
        self._nextObject = None
        self._previousObject = None
        
        
        # associated 'templates' 
        self._ltemplates = None

    def __str__(self):
        return  "%s[%s] %s" % (self.getName(),self.getID(),self.getContent()[:10].encode('utf-8'))
    
    def __repr__(self):
        return "%s[%s] %s" % (self.getName(),self.getID(),self.getContent()[:10].encode('utf-8'))
        return "%s %s %s" % (self.getName(),self.getContent().encode('utf-8'),self.getAttributes())
    
    def getID(self): return self._orderedID
    #def __eq__(self,xmlo):   _eq__ or __cmp__
    def isBeforeThan(self,xmlo):
        """
            is self before xmlo in the original xml order
        """
        if isinstance(xmlo,XMLObjectClass):
            return self.getOrderedID() < xmlo.getOrderedID()
        
    def setOrderedID(self,i): self._orderedID = i
    def getOrderedID(self): return self._orderedID
    
    def getNode(self): return self._domNode
    def setNode(self,c): self._domNode = c
    
    
    def resetTemplate(self): self._ltemplates = None
    def addTemplate(self,t): 
        try:self._ltemplates.append(t)
        except AttributeError: self._ltemplates = [t]
    
    def getTemplates(self): return self._ltemplates
    
    def tagMe(self,sLabel=None):
        """
             create a dom elt and add it to the doc
        """
        newNode = libxml2.newNode(self.tagName)
        
        newNode.setProp('x',str(self.getX()))
        newNode.setProp('y',str(self.getY()))
        newNode.setProp('height',str(self.getHeight()))
        newNode.setProp('width',str(self.getWidth()))
        if self.getParent():
            self.getParent().getNode().addChild(newNode)
        else:
            self.getPage().getNode().addChild(newNode)
        

        # add attributres
        for att in self.getAttributes():
            newNode.setProp(att,str(self.getAttribute(att)))
        
        self.setNode(newNode)
#         for o in self.getObjects():
#             o.setParent(self)
#             o.tagMe()
            
        return newNode
    
    def fromDom(self,domNode):
        
        ## if domNode in mappingTable:
        ## -> ise the fromDom of the specific object
        
        self.setName(domNode.name)
#         print self.getName()
        self.setNode(domNode)
        # get properties
        prop = domNode.properties
        while prop:
            self.addAttribute(prop.name,prop.getContent())
            # add attributes
            prop = prop.next
        child = domNode.children
        ## if no text: add a category: text, graphic, image, whitespace
        while child:
            if child.type == 'text':
                if self.getContent() is not None:
                    self.addContent(child.getContent().decode("UTF-8"))
                else:
                    self.setContent(child.getContent().decode("UTF-8"))
                pass
            elif child.type =="comment":
                pass
            elif child.type == 'element':
                newChild = XMLObjectClass()
                # create sibling relation?
                newChild.fromDom(child)
                self.addObject(newChild)
                ## create a space ???? for <BR>
#                 if child.name.lower() =='br':
#                     self.addContent(' ')
            child = child.next

    def getSetOfListedAttributes(self,TH,lAttributes,myObject):
        """
        
            move to XMLObjectClass ??
            
            Generate a set of features: 
            
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
                except KeyError:
                    lHisto[attr] = {}
                if elt.hasAttribute(attr):
                    try:lHisto[attr][round(float(elt.getAttribute(attr)))].append(elt)
                    except KeyError: lHisto[attr][round(float(elt.getAttribute(attr)))] = [elt]
        
        if lHisto != {}:         
            for attr in lAttributes:
                for value in lHisto[attr]:
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
                            
            
        return self.getSetofFeatures()
    
    def getContentAnaphoraAttributes(self,TH,lAttributes,myObject):
        """
        
            textual content + position wrt parent
    
        """
        from spm.feature import featureObject
     
        if self._lBasicFeatures is None:
            self._lBasicFeatures = []
        # needed to keep canonical values!
        elif self.getSetofFeatures() != []:
            return self.getSetofFeatures()
               
              
        lHisto = {}
        lHisto['position']={}
        for elt in self.getAllNamedObjects(myObject):
            ## if elt is first in elt.getParent()
            position=  elt.getParent().getObjects().index(elt)
            if position == len(elt.getParent().getObjects())-1:position=-1
            try:lHisto['position'][str(position)].append(elt)
            except KeyError : lHisto['position'][str(position)]= [elt]
            for attr in lAttributes:
                try:lHisto[attr]
                except KeyError:lHisto[attr] = {}
                if elt.hasAttribute(attr):
                    try:
                        try:lHisto[attr][round(float(elt.getAttribute(attr)))].append(elt)
                        except KeyError: lHisto[attr][round(float(elt.getAttribute(attr)))] = [elt]
                    except TypeError:pass
                elif attr == 'text':
                    try:lHisto[attr][elt.getContent()].append(elt)
                    except KeyError: lHisto[attr][elt.getContent()] = [elt]
        
        for attr in lAttributes:
            for value in lHisto[attr]:
                if  len(lHisto[attr][value]) > 0.1:  # 0.1: keep all!
                    if attr not in ['position','text']:
                        ftype= featureObject.NUMERICAL
                    else:
                        ftype= featureObject.EDITDISTANCE
                    feature = featureObject()
                    feature.setName(attr)
#                     feature.setName('f')
                    feature.setTH(TH)
                    feature.addNode(self)
                    feature.setObjectName(self)
                    feature.setValue(value)
                    feature.setType(ftype)
                    self.addFeature(feature)
         
      
            
        return self.getSetofFeatures()         


    def getSetOfMutliValuedFeatures(self,TH,lMyFeatures,myObject):
        """
            define a multivalued features
        """
        from spm.feature import multiValueFeatureObject

        #reinit 
        self._lBasicFeatures = None
        
        mv =multiValueFeatureObject()
        name= "multi" #'|'.join(i.getName() for i in lMyFeatures)
        mv.setName(name)
        mv.addNode(self)
        mv.setObjectName(self)
        mv.setTH(TH)
        mv.setObjectName(self)
        mv.setValue(map(lambda x:x,lMyFeatures))
        mv.setType(multiValueFeatureObject.COMPLEX)
        self.addFeature(mv)
        return self._lBasicFeatures            
