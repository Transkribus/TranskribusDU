# -*- coding: utf-8 -*-
"""

    XML page object class 
    
    Hervé Déjean
    cpy Xerox 2009, 2017
    
    a class for page object from a XMLDocument

"""

from __future__ import absolute_import
from __future__ import  print_function
from __future__ import unicode_literals

from config import ds_xml_def
from .XMLDSObjectClass import XMLDSObjectClass
from .XMLDSLINEClass import XMLDSLINEClass
from .XMLDSTEXTClass import XMLDSTEXTClass
from .XMLDSBASELINEClass import XMLDSBASELINEClass
from .XMLDSGRAHPLINEClass import XMLDSGRAPHLINEClass
from .XMLDSTABLEClass import XMLDSTABLEClass
from .XMLDSCOLUMN import XMLDSCOLUMNClass

class  XMLDSPageClass(XMLDSObjectClass):
    """
        PAGE class
        
        or image class ?????
        
    """
    orderID = 0
    def __init__(self,domNode = None):
        XMLDSObjectClass.__init__(self)
        XMLDSObjectClass.id += 1
        self._domNode = domNode
        self._number = None
        
        self._BB = None
        
        
        ### attributes for building vertical zones in the page
        self._VX1Info = []
        self._VX2Info = []
        self._VXCInfo = []
        self._VWInfo = []
        self._HGLFeatures = []
        self._VGLFeatures = []
        
        self._X1X2 = []
        self.lf_XCut = []

        
        
        ## list of vertical page zones Templates
        # by default: one column Template
        self._verticalZoneTemplates = []
        
        self.dVSeparator = {}
        # INIT: emptyfeatures?
        self.dVSeparator['INIT']=[]
                  
        # index them by verticalTemplates?? 
        ## or add attribtute to the region -> region.myTemplate=..
        self.lVerticalObjects={}
        self.lVerticalObjects['INIT'] = [self]
        
        
    
    def __repr__(self):
        return "%s %s %d" % (self.getName(),self.getNumber(), len(self.getObjects()))
    def __str__(self):
        return "%s %s" % (self.getName(),self.getNumber())
    
    def getNumber(self): return  self._number
    def setNumber(self,n): self._number = n
    
    def getBB(self): return self._BB
    def setBB(self,b): self._BB = b
    
    
    def getX(self): return 0
    def getY(self): return 0
    def getX2(self):
        return self.getWidth()
    def getY2(self): return self.getHeight()
    
    def fromDom(self,domNode,lEltNames):
        """
            load a page from dom 
        """
        self.setName(domNode.tag)
        self.setNode(domNode)
        # get properties
        for  prop in domNode.keys():
            self.addAttribute(prop,domNode.get(prop))
            
#         ctxt = domNode.doc.xpathNewContext()
#         ctxt.setContextNode(domNode)
#         ldomElts = ctxt.xpathEval('./*')
#         ctxt.xpathFreeContext()
        ldomElts = domNode.findall('./*')
        for elt in ldomElts:
            ### GT elt
            if elt.tag =='MARGIN':
                elt = list(elt)[0]  #elt=elt.children
            if elt.tag in lEltNames:
                if elt.tag == ds_xml_def.sCOL_Elt:
                    myObject= XMLDSCOLUMNClass(elt)
                    self.addObject(myObject)
                    myObject.setPage(self)
                    myObject.fromDom(elt)                      
                elif elt.tag  == ds_xml_def.sLINE_Elt:
                    myObject= XMLDSLINEClass(elt)
                    self.addObject(myObject)
                    myObject.setPage(self)
                    myObject.fromDom(elt)
                elif elt.tag  == ds_xml_def.sTEXT:
                    myObject= XMLDSTEXTClass(elt)
                    self.addObject(myObject)
                    myObject.setPage(self)
                    myObject.fromDom(elt)
                elif elt.tag == "BASELINE":
                    myObject= XMLDSBASELINEClass(elt)
                    self.addObject(myObject)
                    myObject.setPage(self)
                    myObject.fromDom(elt)       
                elif elt.tag in ['SeparatorRegion', 'GRAPHELT']:
                    myObject= XMLDSGRAPHLINEClass(elt)
                    self.addObject(myObject)
                    myObject.setPage(self)
                    myObject.fromDom(elt)
                elif elt.tag == ds_xml_def.sTABLE:
                    myObject= XMLDSTABLEClass(elt)
                    self.addObject(myObject)
                    myObject.setPage(self)
                    myObject.fromDom(elt)                    
                else:
                    myObject= XMLDSObjectClass()
                    myObject.setNode(elt)
                    myObject.setName(elt.tag)
                    self.addObject(myObject)
                    myObject.setPage(self)
                    myObject.fromDom(elt)
        
            else:
                pass


    #TEMPLATE
#     def setVerticalTemplates(self,lvm):
#         self._verticalZoneTemplates = lvm
        
    def resetVerticalTemplate(self): 
        self._verticalZoneTemplates = []
        self.dVSeparator = {}

    def addVerticalTemplate(self,vm): 
        #if vm not in self._verticalZoneTemplates:
        self._verticalZoneTemplates.append(vm)
        self.dVSeparator[vm]=[]
        
    def getVerticalTemplates(self): return self._verticalZoneTemplates
    
    #REGISTERED CUTS
    def resetVSeparator(self,template):
        try:
            self.dVSeparator[template] = []
        except KeyError:
            pass
        
    def addVSeparator(self,template, lcuts):
        try:
            self.dVSeparator[template].extend(lcuts)
        except KeyError:
            self.dVSeparator[template] = lcuts
        
    def getdVSeparator(self,template): 
        try:return self.dVSeparator[template]
        except KeyError: return []  
    
    
    #OBJECTS (from registered cut regions)
    
    def addVerticalObject(self,Template,o): 
        try:self.lVerticalObjects[Template]
        except KeyError: self.lVerticalObjects[Template]=[]
        
        if o not in self.lVerticalObjects[Template]:
            self.lVerticalObjects[Template].append(o)
        # store in self.getObjects()?????
        
    def getVerticalObjects(self,Template):
        """
            for a given template , get the objects of the page
        """
        return self.lVerticalObjects[Template]
    
    

    def setHGLFeatures(self,f): self._HGLFeatures.append(f)
    def getHGLFeatures(self): return self._HGLFeatures
        
    def setVGLFeatures(self,f): self._VGLFeatures.append(f)
    def getVGLFeatures(self): return self._VGLFeatures
    
    def getVX1Info(self): return self._VX1Info
    def setVX1Info(self,lInfo):
        """
            list of X1 features for the H structure of the page
                corresponds to information to segment the page vertically
        """
        self._VX1Info = lInfo

    def getVX2Info(self): return self._VX2Info
    def setVX2Info(self,lInfo):
        """
            list of X1 features for the H structure of the page
                corresponds to information to segment the page vertically
        """
        self._VX2Info = lInfo

    def getX1X2(self): return self._X1X2
    def setX1X2(self,x): self._X1X2.append(x) 

    def getVWInfo(self): return self._VWInfo
    def setVWInfo(self,lInfo):
        self._VWInfo = lInfo

    def getVXCInfo(self): return self._VXCInfo
    def setVXCInfo(self,lInfo):
        self._VXCInfo = lInfo        


    def addBoundingBox(self,lElts=None):
        """
            XMLDSObjectClass level?
        """
        if self.getBB() is not None:
            return self.getBB()
        
        minbx = 9e9
        minby = 9e9
        maxbx = 0
        maxby = 0
        
        if lElts is None:
            lElts= self.getAllNamedObjects(XMLDSTEXTClass)
        
        for elt in lElts:
#            if len(elt.getContent()) > 0 and elt.getHeight()>4 and elt.getWidth()>4:
                if elt.getX()>=0 and elt.getX() < minbx: minbx = elt.getX()
                if elt.getY()>=0 and elt.getY() < minby: minby = elt.getY()
                if elt.getX() + elt.getWidth() > maxbx: maxbx = elt.getX() + elt.getWidth()
                if elt.getY() + elt.getHeight()  > maxby: maxby = elt.getY() + elt.getHeight()
        
        self._BB = [minbx,minby,maxby-minby,maxbx-minbx]
        
        return minbx,minby,maxby-minby,maxbx-minbx         
         
    
    def generateAllNSequences(self,lElts,lengthMax):
        """
            generate all sequences of length N from elt.next
        """
        lSeq=[]
        for elt in lElts:
            lSeq.append([elt])
        i=0
        bOK=True
        lFinal=[]
        while bOK:
            seq=lSeq[i]
            if len(seq) < lengthMax:
                for nexte in seq[-1].next:
                    newseq=seq[:]
                    newseq.append(nexte)
                    lSeq.append(newseq)
            else:
                lFinal.append(seq)
            if i < len(lSeq)-1:
                i+=1
            else:
                bOK=False
#         print lFinal
        return [lFinal]
            
    #### SEGMENTATION
    

    def createVerticalZonesOLD(self,Template,tagLevel=XMLDSTEXTClass):
        """
           Create 'columns' according to X cuts and populate with objects (text)    
           
        """
        #reinit lVerticalObjects
        self.lVerticalObjects[Template]=[]
        prevCut=0
        for xcut in self.getdVSeparator(Template):
            region=XMLDSObjectClass()
            region.setName('VRegion')
            region.addAttribute('x', prevCut)
            region.addAttribute('y', 0)
            region.addAttribute('height', self.getAttribute('height'))
            region.addAttribute('width', str(xcut.getValue()-prevCut))
            region.setPage(self)
            prevCut=xcut.getValue()
#             print self, region.getX(), region.getY(), region.getHeight(),region.getWidth()    
            lclippedObjects=[]
            for subObject in self.getAllNamedObjects(XMLDSTEXTClass):
                co = subObject.clipMe(region)
                if co :
                    co.setParent(region)
                    lclippedObjects.append(co)
            if lclippedObjects != []:
                region.setObjectsList(lclippedObjects)
#                 print '\t=',  lclippedObjects
            self.addVerticalObject(Template,region)

        #last column
        region=XMLDSObjectClass()
        region.setName('VRegion')
        region.addAttribute('x', prevCut)
        region.addAttribute('y', 0)
        region.addAttribute('height', self.getAttribute('height'))
        region.addAttribute('width', str(self.getWidth() - prevCut))
        prevCut=xcut.getValue()
#             print self, region.getX(), region.getY(), region.getHeight(),region.getWidth()    
        lclippedObjects=[]
        #
        for subObject in self.getAllNamedObjects(tagLevel):
            co = subObject.clipMe(region)
            if co :
                co.setParent(region) 
                lclippedObjects.append(co)
        if lclippedObjects != []:
            region.setObjectsList(lclippedObjects)
#                 print '\t=',  lclippedObjects
        self.addVerticalObject(Template,region)       
    
    
    def createVerticalZones(self,Template,tagLevel=XMLDSTEXTClass):
        """
           Create 'columns' according to X cuts and populate with objects (text)    
           
        """
        #reinit lVerticalObjects
        self.lVerticalObjects[Template]=[]
        
        # create regions
        prevCut=0
        for xcut in self.getdVSeparator(Template):
            region=XMLDSObjectClass()
            region.setName('VRegion')
            region.addAttribute('x', prevCut)
            region.addAttribute('y', 0)
            region.addAttribute('height', self.getAttribute('height'))
            region.addAttribute('width', str(xcut.getValue()-prevCut))
            region.setPage(self)
            prevCut=xcut.getValue()
            self.addVerticalObject(Template,region)

        #last column
        region=XMLDSObjectClass()
        region.setName('VRegion')
        region.addAttribute('x', prevCut)
        region.addAttribute('y', 0)
        region.addAttribute('height', self.getAttribute('height'))
        region.addAttribute('width', str(self.getWidth() - prevCut))
        prevCut=xcut.getValue()
        self.addVerticalObject(Template,region)      

        
        ## populate regions
        for subObject in self.getAllNamedObjects(tagLevel):
            region= subObject.bestRegionsAssignment(self.getVerticalObjects(Template))
            if region:
                region.addObject(subObject)
        

    def getSetOfMutliValuedFeatures(self,TH,lMyFeatures,myObject):
        """
            define a multivalued features 
        """
        from spm.feature import multiValueFeatureObject

        if self._lBasicFeatures is None:
            self._lBasicFeatures = []
        # needed to keep canonical values!
        elif self.getSetofFeatures() != []:
            return self.getSetofFeatures()
   
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
            
        if self.getSetofFeatures() == []:
            feature = multiValueFeatureObject()
            feature.setName('EMPTY')
            feature.setTH(TH)
            feature.addNode(self)
            feature.setObjectName(self)
            feature.setValue(True)
            feature.setType(multiValueFeatureObject.BOOLEAN)
            self.addFeature(feature)            


        return self.getSetofFeatures()                 
    
    def getSetOfVInfoFeatures(self,TH,lAttributes,myObject):
        """
            
        """
        from spm.feature import featureObject
     
        if self._lBasicFeatures is None:
            self._lBasicFeatures = []   
        # needed to keep canonical values!
        elif self.getSetofFeatures() != []:
            return self.getSetofFeatures()
      

            for attr in lAttributes:
                name= attr[0].getName()
                value = attr[0].getValue()
                feature = featureObject()
                feature.setName(name)
                feature.setTH(TH)
                feature.addNode(self)
                feature.setObjectName(self)
                feature.setValue(value)
                feature.setType(feature.NUMERICAL)
                self.addFeature(feature) 
      
            
        if self.getSetofFeatures()   == []:
            feature = featureObject()
            feature.setName('EMPTY')
            feature.setTH(TH)
            feature.addNode(self)
            feature.setObjectName(self)
            feature.setValue(True)
            feature.setType(featureObject.BOOLEAN)
            self.addFeature(feature)            


        return self.getSetofFeatures()          
        
    def getSetOfFeaturesPageSize(self,TH,lAttributes,myObject):
        """
            features: BB X Y H W
            
        """
        from spm.feature import featureObject
     
        if self._lBasicFeatures is None:
            self._lBasicFeatures = []
        # needed to keep canonical values!
        elif self.getSetofFeatures() != []:
            return self.getSetofFeatures()
   

        feature = featureObject()
        feature.setName('h')
        feature.setTH(TH)
        feature.addNode(self)
        feature.setObjectName(self)
        feature.setValue(round(float(self.getAttribute('height'))))
        feature.setType(feature.NUMERICAL)
        self.addFeature(feature)

        feature = featureObject()
        feature.setName('w')
        feature.setTH(TH)
        feature.addNode(self)
        feature.setObjectName(self)
        feature.setValue(round(float(self.getAttribute('width'))))
        feature.setType(feature.NUMERICAL)
        self.addFeature(feature)
      
            
        if self.getSetofFeatures()  == []:
            feature = featureObject()
            feature.setName('EMPTY')
            feature.setTH(TH)
            feature.addNode(self)
            feature.setObjectName(self)
            feature.setValue(True)
            feature.setType(featureObject.BOOLEAN)
            self.addFeature(feature)           

        return self.getSetofFeatures()           
                 
        
    def getSetOf2DAttributes(self,TH,lAttribute,myObject):
        """
            features: 2D (x,y) blocks in the page
        """
        from spm.feature import TwoDFeature
     
        if self._lBasicFeatures is None:
            self._lBasicFeatures = []
        # needed to keep canonical values!
        elif self.getSetofFeatures() != []:
            return self.getSetofFeatures()
   

        for elt in self.getAllNamedObjects(myObject):  
            if len(elt.getObjects())<5:
#             if elt.getY() <100:          
                feature = TwoDFeature()
                feature.setName('2D')
                feature.setTH(TH)
                feature.addNode(self)
                feature.setObjectName(self.getName())
                feature.setValue((round(elt.getX()),round(elt.getY())))
                feature.setType(feature.COMPLEX)
                self.addFeature(feature)  
      
        return self.getSetofFeatures()          

    def getSetOfFeaturesBB(self,TH,lAttributes,myObject):
        """
            features: BB X Y H W
            
        """
        from spm.feature import featureObject
     
        if self._lBasicFeatures is None:
            self._lBasicFeatures = []
        # needed to keep canonical values!
        elif self.getSetofFeatures() != []:
            return self.getSetofFeatures()
   
        #build BB 
        if self.getBB() is None:
            self.addBoundingBox()
        x,y,h,w = self.getBB()

            
        feature = featureObject()
        feature.setName('lm')
        feature.setTH(TH)
        feature.addNode(self)
        feature.setObjectName(self.getName())
        feature.setValue(round(x))
        feature.setType(feature.NUMERICAL)
        self.addFeature(feature)  

        feature = featureObject()
        feature.setName('rm')
        feature.setTH(TH)
        feature.addNode(self)
        feature.setObjectName(self.getName())
        feature.setValue(round(x+w))
        feature.setType(feature.NUMERICAL)
        self.addFeature(feature)  
      
            
        if self.getSetofFeatures()  == []:
            feature = featureObject()
            feature.setName('EMPTY')
            feature.setTH(TH)
            feature.addNode(self)
            feature.setObjectName(self.getName())
            feature.setValue(True)
            feature.setType(featureObject.BOOLEAN)
            self.addFeature(feature)  

        return self.getSetofFeatures()          
        
        
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
            
        lHisto={}
        for elt in self.getAllNamedObjects(myObject):
            if float(elt.getAttribute('width')) > 00:
                for attr in lAttributes:
                    try:lHisto[attr]
                    except KeyError:lHisto[attr] = {}
                    if elt.hasAttribute(attr):
                        try:lHisto[attr][round(float(elt.getAttribute(attr)))].append(elt)
                        except: lHisto[attr][round(float(elt.getAttribute(attr)))] = [elt]
        
        for attr in lAttributes:
            for value in lHisto[attr]:
                print (attr, value)
                if  len(lHisto[attr][value]) > 0.1:
                    ftype= featureObject.NUMERICAL
                    feature = featureObject()
                    feature.setName(attr)
                    l = sum(x.getHeight() for x in lHisto[attr])
                    feature.setWeight(l)
                    feature.setTH(TH)
                    feature.addNode(self)
                    feature.setObjectName(self)
                    feature.setValue(value)
                    feature.setType(ftype)
                    self.addFeature(feature)  
        
        if  self.getSetofFeatures()   == []:
            feature = featureObject()
            feature.setName('EMPTY')
            feature.setTH(TH)
            feature.addNode(self)
            feature.setObjectName(self)
            feature.setValue(True)
            feature.setType(featureObject.BOOLEAN)
            self.addFeature(feature)        

        
        return self.getSetofFeatures() 
