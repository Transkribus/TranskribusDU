# -*- coding: utf-8 -*-
"""

    XML object class 
    
    Hervé Déjean
    cpy Xerox 2009
    
    a class for TEXT from a XMLDocument

"""

from XMLDSObjectClass import XMLDSObjectClass
from config import ds_xml_def as ds_xml
from XMLDSTOKENClass import XMLDSTOKENClass


class  XMLDSTEXTClass(XMLDSObjectClass):
    """
        TEXT (chunk) class
    """
    def __init__(self,domNode = None):
        XMLDSObjectClass.__init__(self)
        XMLDSObjectClass.id += 1
        self._domNode = domNode
        
        self.Obaseline=None
        self.setName(ds_xml.sTEXT)
    
#     def getX(self): return float(self.getAttribute('x'))
#     def getY(self): return float(self.getAttribute('y'))
    def getX2(self):
        return float(self.getAttribute('x'))+self.getWidth()
    def getY2(self): 
        return float(self.getAttribute('y'))+self.getHeight()        
#     def getHeight(self): return float(self.getAttribute('height'))
#     def getWidth(self): return float(self.getAttribute('width'))
        
    def fromDom(self,domNode):
        """
            only contains TEXT?
            
            attributes x y  id height width  (all!)
            
        """
#         self.setName(domNode.name)
        self.setNode(domNode)
        # get properties
        prop = domNode.properties
        while prop:
            self.addAttribute(prop.name,prop.getContent())
            prop = prop.next
        try: 
            self._id = self.getAttribute('id')
        except:pass
        self.addAttribute('x2', float(self.getAttribute('x'))+self.getWidth())
        self.addAttribute('y2',float(self.getAttribute('y'))+self.getHeight() )

        if self.hasAttribute('blpoints'): 
            from ObjectModel.XMLDSBASELINEClass import XMLDSBASELINEClass
            b= XMLDSBASELINEClass()
            b.fromDom(domNode)
            b.setParent(self.getParent())
            self.setBaseline(b)
        
        ctxt = domNode.doc.xpathNewContext()
        ctxt.setContextNode(domNode)        
        ldomElts = ctxt.xpathEval('./%s'%(ds_xml.sTOKEN))
        for elt in ldomElts:
            try:
                myObject= XMLDSTOKENClass(elt)
                self.addObject(myObject)
                myObject.setPage(self.getParent().getPage())
                myObject.fromDom(elt)
            except: pass #print 'issue with token'
          
        ctxt.xpathFreeContext()        
    
    def setBaseline(self,ob): self.Obaseline = ob
    def getBaseline(self):
        return self.Obaseline
    
    def computeBaseline(self):
        
        if self.getBaseline() is not None:
            return self.getBaseline()
#         lHisto={}
        lY=[]
        lX=[]
        
        # test if TOKEN has position (not in GT)!
        for token in self.getAllNamedObjects(XMLDSTOKENClass):
            try:
                lX.append(token.getX())
                lX.append(token.getX2())
                lY.append(token.getY2())
                lY.append(token.getY2())
            except  TypeError:
                pass            

        import numpy as np
        
        if len(lX) > 0:
            a,bx = np.polyfit(lX, lY, 1)
            
            lPoints = ','.join(["%d,%d"%(xa,ya) for xa,ya  in zip(lX, lY)])
#             print 'ANLGE:',math.degrees(math.atan(a))
            ymax = a*self.getWidth()+bx     
            from ObjectModel.XMLDSBASELINEClass import XMLDSBASELINEClass
            b= XMLDSBASELINEClass()
            b.setNode(self)
#             b.addAttribute("points",lPoints)
            b.setAngle(a)
            b.setBx(bx)
            b.setPoints(lPoints)
            b.setParent(self)
            self.setBaseline(b)
            b.computePoints()
          
    def getTokens(self):
        """
            if dom tokens: rturn them
            else split content
        """
        if self.getAllNamedObjects(XMLDSTOKENClass) != []:
            return self.getAllNamedObjects(XMLDSTOKENClass)
        else:
            for token in self.getContent().split():
                oT=XMLDSTOKENClass()
                oT.setParent(self)
                oT.setPage(self.getPage())
                self.addObject(oT)
                oT.setContent(token)
            return self.getAllNamedObjects(XMLDSTOKENClass)

          
            
    def getSetOfFeaturesXPos(self,TH,lAttr,myObject):

        from spm.feature import featureObject
     
        if self._lBasicFeatures is None:
            self._lBasicFeatures = []
            
        ftype= featureObject.NUMERICAL
        feature = featureObject()
        feature.setName('x')
        feature.setTH(TH)
        feature.addNode(self)
        feature.setObjectName(self)
        feature.setValue(round(self.getX()))
        feature.setType(ftype)
        self.addFeature(feature)
        
        ftype= featureObject.NUMERICAL
        feature = featureObject()
        feature.setName('x2')
        feature.setTH(TH)
        feature.addNode(self)
        feature.setObjectName(self)
        feature.setValue(round(self.getX()+self.getWidth()))
        feature.setType(ftype)
        self.addFeature(feature)
                  
        ftype= featureObject.NUMERICAL
        feature = featureObject()
        feature.setName('xc')
        feature.setTH(TH)
        feature.addNode(self)
        feature.setObjectName(self)
        feature.setValue(round(self.getX()+self.getWidth()/2))
        feature.setType(ftype)
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

