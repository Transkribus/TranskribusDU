# -*- coding: utf-8 -*-
"""

    XML object class 
    
    Hervé Déjean
    cpy Xerox 2009
    
    a class for object from a XMLDocument

"""
from __future__ import absolute_import
from __future__ import  print_function
from __future__ import unicode_literals


from .XMLObjectClass import XMLObjectClass
from config import ds_xml_def as ds_xml
from shapely.geometry import polygon,Polygon,LineString
from lxml import etree

class  XMLDSObjectClass(XMLObjectClass):
    """
        DS object class
        this is a 2D object
    """
    orderID = 0
    name = None
    def __init__(self):
        XMLObjectClass.__init__(self)
        
        self._lElements = []
        self._page  = None
        self._id= None
        
        self.Xnearest=[[],[]]  # top bottom
        self.Ynearest=[[],[]]  # left right      
        
        
        ## need to have x,y,w,h directly: otherwise: too slow
        self._x = None
        self._y = None
        self._h = None
        self._w = None  
        
        self._poly = None
          
          
        
    def getPage(self): return self._page
    def setPage(self,p): self._page= p 
    
    def getID(self): return self._id
    
#     def getElements(self): return self._lElements
#     def addElement(self,e): self._lElements.append(e)
    

    def getX(self): return self._x #float(self.getAttribute('x'))
    def getY(self): return self._y # return float(self.getAttribute('y'))
    def getX2(self): return self.getX() + self.getWidth()
    def getY2(self): return self.getY() +self.getHeight()        
    def getHeight(self): return self._h  #return float(self.getAttribute('height'))
    def getWidth(self): return self._w #return float(self.getAttribute('width'))    

    def setX(self,x): self.addAttribute('x',x); self._x = float(x)
    def setY(self,y): self.addAttribute('y',y); self._y = float(y)
    def setWidth(self,w): self.addAttribute('width',w);self._w = float(w)
    def setHeight(self,h):self.addAttribute('height',h);self._h = float(h)
    
    def setDimensions(self,x,y,h,w):
        self.setX(x)
        self.setY(y)
        self.setHeight(h)
        self.setWidth( w)
    
    def toPolygon(self):
        """
        return a shapely polygon using points!!!
        points="375.12,98.88,924.0,101.52,924.0,113.52,375.12,110.88" 
        """
        if self._poly is not None:
            return self._poly

        x  = [float(x) for x in self.getAttribute("points").replace(" ",",").split(',')]
        if len(x) <3*2:
            return   LineString(list(zip(*[iter(x)]*2)))
        self._poly = polygon.orient(Polygon(list(zip(*[iter(x)]*2))))
        if not self._poly.is_valid:self._poly= self._poly.convex_hull
        return     self._poly 
    
    def addObject(self,o,bDom=False): 
        ## move dom node as well
        ##     why ??? 30/05/2017    add Boolean : 30/08/2017
        if o not in self.getObjects():
            self.getObjects().append(o)
            o.setParent(self)
            if bDom: 
                if o.getNode() is not None and self.getNode() is not None:
                    o.getNode().getparent().remove(o.getNode())
                    self.getNode().append(o.getNode())
               
    

    def removeObject(self,o,bDom=False):
        """
            remove o from self.getObjects()
            unlink if bDom
        """
        self.getObjects().remove(o)
        if bDom: 
            if o.getNode() is not None and self.getNode() is not None:
                o.getNode().getparent().remove(o.getNode())
    
    def resizeMe(self,objectType):
        return
        assert len(self.getAllNamedObjects(objectType)) != 0
        
        minbx = 9e9
        minby = 9e9
        maxbx = 0
        maxby = 0
        for elt in self.getAllNamedObjects(objectType):
            if elt.getX() < minbx: minbx = elt.getX()
            if elt.getY() < minby: minby = elt.getY()
            if elt.getX() + elt.getWidth() > maxbx: maxbx = elt.getX() + elt.getWidth()
            if elt.getY() + elt.getHeight()  > maxby: maxby = elt.getY() + elt.getHeight()
        assert minby != 9e9
        self.setX( minbx)
        self.setY( minby)
        self.setWidth(maxbx-minbx)
        self.setHeight(maxby-minby)
        self.addAttribute('x2', maxbx)
        self.addAttribute('y2', maxby)

        self._BB = [minbx,minby,maxby-minby,maxbx-minbx]    
    

    def setXYHW(self,x,y,h,w):
        self.setX(x)
        self.setY(y)
        self.setHeight(h)
        self.setWidth(w)
        
    def copyXYHW(self,o):
        self._x = o.getX()
        self._y  =o.getY()
        self._h = o.getHeight()
        self._w = o.getWidth()
          
    def fromDom(self,domNode):
        
        ## if domNode in mappingTable:
        ## -> ise the fromDom of the specific object
        from .XMLDSLINEClass import XMLDSLINEClass
        from .XMLDSTEXTClass import XMLDSTEXTClass
        from .XMLDSBASELINEClass import XMLDSBASELINEClass
        from .XMLDSTABLEClass import XMLDSTABLEClass        
        
        
        self.setName(domNode.tag)
        self.setNode(domNode)
        # get properties
        for prop in domNode.keys():
            self.addAttribute(prop,domNode.get(prop))
            if prop =='x': self._x= float(domNode.get(prop))
            elif prop =='y': self._y = float(domNode.get(prop))
            elif prop =='height': self._h = float(domNode.get(prop))
            elif prop =='width': self.setWidth(float(domNode.get(prop)))
        
        self.addAttribute('x2', self.getX()+self.getWidth())
        self.addAttribute('y2',self.getY()+self.getHeight() )
        
        self._id = self.getAttribute('id')
        if self.getID() is None:
            self._id = XMLDSObjectClass.orderID
            XMLDSObjectClass.orderID+= 1
        ## if no text: add a category: text, graphic, image, whitespace
        for child in domNode:
#             if child.type == 'text':
#                 if self.getContent() is not None:
#                     self.addContent(child.getContent().decode("UTF-8"))
#                 else:
#                     self.setContent(child.getContent().decode("UTF-8"))
#                 pass
            if child.tag == etree.Comment:
                pass
            else: #if child.type == 'element':
               
                if child.tag == ds_xml.sTABLE:
                    myObject= XMLDSTABLEClass(child)
                    self.addObject(myObject)
                    myObject.setPage(self)
                    myObject.fromDom(child)      
                elif child.tag  == ds_xml.sLINE_Elt:
                    myObject= XMLDSLINEClass(child)
                    # per type?
                    self.addObject(myObject)
                    myObject.setPage(self.getPage())
                    myObject.fromDom(child)
                elif child.tag  == ds_xml.sTEXT:
                    myObject= XMLDSTEXTClass(child)
                    self.addObject(myObject)
                    myObject.setPage(self.getPage())
                    myObject.fromDom(child)

                elif child.tag == ds_xml.sBaseline:
                    myObject= XMLDSBASELINEClass(child)
                    self.addObject(myObject)
                    myObject.setPage(self.getPage())
                    myObject.fromDom(child)      
                else:
                    myObject= XMLDSObjectClass()
                    myObject.setNode(child)
                    # per type?
                    self.addObject(myObject)
                    myObject.setPage(self.getPage())
                    myObject.fromDom(child)   
         
        
         

    def bestRegionsAssignment(self,lRegions,bOnlyBaseline=False):
        """
            find the best (max overlap for self) region  for self
            bOnlyBaseline: reduce the height so that baseline position is more important
        """
        from rtree import index
        
        assert self.toPolygon().convex_hull.is_valid
         
        txtidx = index.Index()
        lP = []
        [lP.append(e.toPolygon()) for e in lRegions if e.toPolygon().is_valid]
        for i,elt in enumerate(lRegions):
            txtidx.insert(i, lP[i].bounds)
        lSet = txtidx.intersection(self.toPolygon().bounds)
        lOverlap = []
        for ei in lSet:
            if lP[ei].is_valid:
                intersec= self.toPolygon().intersection(lP[ei]).area
                if intersec >0:
                    lOverlap.append((ei,lP[ei],intersec))
        if lOverlap != []:        
            lOverlap.sort(key=lambda xyz:xyz[-1])
#             print ("??",self,lRegions[lOverlap[-1][0]])
            return lRegions[lOverlap[-1][0]]
        
        return None
        
    def bestRegionsAssignmentOld(self,lRegions,bOnlyBaseline=False):
        """
            find the best (max overlap for self) region  for self
            bOnlyBaseline: reduce the height so that baseline position is more important
        """

        if bOnlyBaseline:
            #backup height
            Hbackup = self.getHeight()
            Ybackup= self.getY()
            self.setHeight(1)
            self.setY(Hbackup+self.getY())
        lOverlap=[]        
        for region in lRegions:
#             lOverlap.append(self.signedRatioOverlap(region))
            lOverlap.append(self.signedRatioOverlapY(region))
#             print(self.getX(),self.getWidth(),region, self.signedRatioOverlapX(region))

        if bOnlyBaseline:
            #restaure height
            self.setHeight(Hbackup)
            self.setY(Ybackup)
        
        if lOverlap ==[] : return None
        return lRegions[lOverlap.index(max(lOverlap))]
        
    def clipMe(self,clipRegion,lSubObject=[]):
        """
        
            return the list of new subobjects considering the clip region
            -->  if subobjects is not 'clippable' ? how to cut a TOKEN????
            
            recursive: if a sub object is cut: go down and take only clipped) subobj if the zones
            -> text/token
            
            region: a XMLDSObjectClass!!
            create new objects?  YES::
            
        """
        ## if self has no subojects: need to be inserted in the region???
        ## current object
        if self.overlap(clipRegion):
            myNewObject =  XMLDSObjectClass()
#             print self.getAttributes()           
            ## ID missing! - > use self id and extend?
            newX = max(clipRegion.getX(),self.getX())
            newY = max(clipRegion.getY(),self.getY())
            newW = min(self.getX2(),clipRegion.getX2()) - newX
            newH = min(self.getY2(),clipRegion.getY2()) - newY
            
            myNewObject.setX(newX)
            myNewObject.setY(newY)
            myNewObject.setHeight(newH)
            myNewObject.setWidth(newW)            
            
#             print self.getID(),self.getName(),self.getContent()
#             print '\tnew dimensions',myNewObject.getX(),myNewObject.getY(),myNewObject.getWidth(),myNewObject.getHeight()
            
            if lSubObject == []:
                lSubObject= self.getObjects()
        
            for subObject in lSubObject:
                ## get clipped dimensions
                if subObject.getAttribute("x"): 
                    newSub= subObject.clipMe(clipRegion)
                    if newSub:
                        myNewObject.addObject(newSub)
                # bug: tokne: no x: add them
                else:
                    myNewObject.addObject(subObject)
            return myNewObject 
        else:
            return None
        
         
    def signedRatioOverlapY(self,zone):
        """
            return the overlap ratio betwenn self and zone
        """     
        [a1,a2] = self.getY(),self.getY() + self.getHeight()
        [b1,b2] = zone.getY(),zone.getY() + zone.getHeight()
        if min(a2, b2) >=  max(a1, b1): return min(a2, b2) -  max(a1, b1)
        else: return -1  
        
    def signedRatioOverlapX(self,zone):
        
        [a1,a2] = self.getX(),self.getX()+ self.getWidth()
        [b1,b2] = zone.getX(),zone.getX()+ zone.getWidth()
        if  min(a2, b2) >=   max(a1, b1): return    min(a2, b2) -   max(a1, b1)
        else: return -1  
    
    def signedRatioOverlap(self,zone):
        """
         overlap self and zone
         return surface of self in zone 
        """
        [x1,y1,h1,w1] = self.getX(),self.getY(),self.getHeight(),self.getWidth()
        [x2,y2,h2,w2] = zone.getX(),zone.getY(),zone.getHeight(),zone.getWidth()
        
        fOverlap = 0.0
        
        if self.overlapX(zone) and self.overlapY(zone):
            [x11,y11,x12,y12] = [x1,y1,x1+w1,y1+h1]
            [x21,y21,x22,y22] = [x2,y2,x2+w2,y2+h2]
            
            s1 = w1 * h1
            
            # possible ?
            if s1 == 0: s1 = 1.0
            
            #intersection
            nx1 = max(x11,x21)
            nx2 = min(x12,x22)
            ny1 = max(y11,y21)
            ny2 = min(y12,y22)
            h = abs(nx2 - nx1)
            w = abs(ny2 - ny1)
            
            inter = h * w
            if inter > 0 :
                fOverlap = inter/s1
            else:
                # if overX and Y this is not possible !
                fOverlap = 0.0
            
        return  fOverlap   
                           
    def ratioOverlap(self,zone):
        """
         overlap self and zone
        """
        [x1,y1,h1,w1] = self.getX(),self.getY(),self.getHeight(),self.getWidth()
        [x2,y2,h2,w2] = zone.getX(),zone.getY(),zone.getHeight(),zone.getWidth()
        
        fOverlap = 0.0
        
        if self.overlapX(zone) and self.overlapY(zone):
            [x11,y11,x12,y12] = [x1,y1,x1+w1,y1+h1]
            [x21,y21,x22,y22] = [x2,y2,x2+w2,y2+h2]
            
            s1 = 1.0* w1 * h1
            if s1 == 0: s1 = 1.0
            s2 = 1.0*w2 * h2
            if s2 == 0: s2 = 1.0
            
            nx1 = max(x11,x21)
            nx2 = min(x12,x22)
            # borderline : line and // 22/05:2017: why? 
            if nx1 == nx2:
                nx1 += 1
            ny1 = max(y11,y21)
            ny2 = min(y12,y22)
            # borderline : line and  // 22/05/2017: why? 
            if ny1 == ny2:
                ny1 = ny2 - 1
            
            h = abs(nx2 - nx1)
            w = abs(ny2 - ny1)

            fOverlap = (h*w)/ (max(s1,s2))
        return  fOverlap  
    
    def overlap(self,zone):
        return self.overlapX(zone) and self.overlapY(zone)
    
    def overlapX(self,zone):
        
    
        [a1,a2] = self.getX(),self.getX()+ self.getWidth()
        [b1,b2] = zone.getX(),zone.getX()+ zone.getWidth()
        return min(a2, b2) >=   max(a1, b1) 
        
    def overlapY(self,zone):
        [a1,a2] = self.getY(),self.getY() + self.getHeight()
        [b1,b2] = zone.getY(),zone.getY() + zone.getHeight()
        return min(a2, b2) >=  max(a1, b1)   
        
        
    def getSetOfX1X2Attributes(self,TH,foo,myObject):
        """
            input: feature threshold (eq) 
        """
        from spm.feature import featureObject,TwoDFeature
     
        if self._lBasicFeatures is None:
            self._lBasicFeatures = []
        # needed to keep canonical values!
        elif self.getSetofFeatures() != []:
            return self.getSetofFeatures()
        
        for elt in self.getAllNamedObjects(myObject):
            ftype= featureObject.COMPLEX
            feature = TwoDFeature()
            feature.setName("x1x2")
            feature.setTH(TH)
            feature.addNode(elt)
            feature.setObjectName(self)
            feature.setValue((elt.getX(),elt.getX2()))
            feature.setType(ftype)
            self.addFeature(feature)        
        
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
                except KeyError:lHisto[attr] = {}
                if elt.hasAttribute(attr):
                    try:
                        try:lHisto[attr][round(float(elt.getAttribute(attr)))].append(elt)
                        except KeyError: lHisto[attr][round(float(elt.getAttribute(attr)))] = [elt]
                    except TypeError:pass

        # empty object
        if lHisto =={}:
            return self.getSetofFeatures()

        for attr in lAttributes:
            for value in lHisto[attr]:
#                 print attr, value, lHisto[attr][value]
                if  len(lHisto[attr][value]) > 0.1:
                    ftype= featureObject.NUMERICAL
                    feature = featureObject()
                    feature.setName(attr)
#                     feature.setName('f')
                    feature.setTH(TH)
                    for o in lHisto[attr][value]:
                        feature.addNode(o)
                    feature.setObjectName(self)
                    feature.setValue(value)
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
                            
            
        return self.getSetofFeatures()
    
         
#     def getSetOfMutliValuedFeatures(self,TH,lMyFeatures,myObject):
#         """
#             define a multivalued features
#             move to XMLObject? 
#         """
#         from spm.feature import multiValueFeatureObject
# 
#         #reinit 
#         self._lBasicFeatures = None
#         
#         mv =multiValueFeatureObject()
#         name= "multi" #'|'.join(i.getName() for i in lMyFeatures)
#         mv.setName(name)
#         mv.addNode(self)
#         mv.setObjectName(self)
#         mv.setTH(TH)
#         mv.setObjectName(self)
#         mv.setValue(map(lambda x:x,lMyFeatures))
#         mv.setType(multiValueFeatureObject.COMPLEX)
#         self.addFeature(mv)
#         return self._lBasicFeatures    
#             
