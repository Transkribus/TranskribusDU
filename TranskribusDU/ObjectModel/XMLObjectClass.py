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


from .objectClass import objectClass
from lxml   import etree

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
        return  "%s[%s] %s" % (self.getName(),self.getID(),self.getContent())
    
    def __repr__(self):
        return "%s[%s] %s" % (self.getName(),self.getID(),self.getContent())
        return "%s %s %s" % (self.getName(),self.getContent(),self.getAttributes())
    
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
        newNode = etree.Element(self.tagName)
        
        newNode.set('x',str(self.getX()))
        newNode.set('y',str(self.getY()))
        newNode.set('height',str(self.getHeight()))
        newNode.set('width',str(self.getWidth()))
        if self.getID():
            newNode.set('id',str(self.getID()))
            
        if self.getParent():
            self.getParent().getNode().append(newNode)
        else:
            self.getPage().getNode().append(newNode)
        

        # add attributres
        for att in self.getAttributes():
            newNode.set(att,str(self.getAttribute(att)))
        
        # add content:!
        newNode.text = self.getContent()
        
        # add children!!!!
        
        
        self.setNode(newNode)
#         for o in self.getObjects():
#             o.setParent(self)
#             o.tagMe()
            
        return newNode
    
    def fromDom(self,domNode):
        
        ## if domNode in mappingTable:
        ## -> ise the fromDom of the specific object
        
        self.setName(domNode.tag)
#         print self.getName()
        self.setNode(domNode)
        # get properties
        for prop in domNode.keys():
            self.addAttribute(prop,domNode.get(prop))
#         child = domNode.children
        ## if no text: add a category: text, graphic, image, whitespace
        for  child in domNode:
            if child.text is not None:
                if self.getContent() is not None:
                    self.addContent(child.text)
                else:
                    self.setContent(child.text)
                pass
            elif child.tag == etree.Comment:
                pass
            else : #if child.tag == etree.Element:
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
    
    
    
if __name__ == "__main__":

    NS_PAGE_XML         = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
    
    NS_XSI ="http://www.w3.org/2001/XMLSchema-instance"
    XSILOCATION ="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd"  

# <PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd">
# 
# 
# xmlns = "http://www.topografix.com/GPX/1/1"
# xsi = "http://www.w3.org/2001/XMLSchema-instance"
# schemaLocation = "http://www.topografix.com/GPX/1/1 http://www.topografix.com/GPX/1/1/gpx.xsd"
# version = "1.1"
# ns = "{xsi}"
# 
# getXML = etree.Element("{" + xmlns + "}gpx", version=version, attrib={"{xsi}schemaLocation": schemaLocation}, creator='My Product', nsmap={'xsi': xsi, None: xmlns})
# print(etree.tostring(getXML, xml_declaration=True, standalone='Yes', encoding="UTF-8", pretty_print=True))

    #Schema for Transkribus PageXml
    XSL_SCHEMA_FILENAME = "pagecontent.xsd"


#     xmlPAGERoot = etree.Element('{%s}PcGts'%NS_PAGE_XML,attrib={"{"+NS_XSI+"}schemaLocation" : XSILOCATION},nsmap={ None: NS_PAGE_XML})
#     xmlPageDoc = etree.ElementTree(xmlPAGERoot)
# 
#     print (etree.QName(xmlPAGERoot.tag).localname)
# #     print(etree.tostring(xmlPAGERoot))
#     tag = etree.QName('http://www.w3.org/1999/xhtml', 'html')
#     print(tag)


    root=etree.Element('{sss}root',nsmap={ None: "sss"})
    doc = etree.ElementTree(root)
    elem = etree.Element("{sss}tag1", first="1", second="2")
    elem2 = etree.Element("{sss}tag2", first="1", second="2")
    root.append(elem)
    root.append(elem2)
    elem.text ='sdsd'
    elem.tail="tail"
    comment = etree.Comment("my comment")
    root.append(comment)
    
    for x in root.xpath('.//@*'): print (x)
        
    for p in root.findall('.//*[@first]'):
        p.set('first',"4")

    for x in root:
        if x.tag  != etree.Comment:
            print(etree.QName(x.tag))
    ss=elem.text
    elem2.text=elem.tail
    print(ss)
    root[-1] = root[0]
    print (etree.tostring(doc,encoding='UTF-8',xml_declaration=True,pretty_print=True))
    print (elem.getnext())
    if elem.get('sdsdsd'): print("ddddddddddd")
    else: print ('aaaaaaaaa')
    

        
