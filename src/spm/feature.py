# -*- coding: utf-8 -*-
""" 
    H. Déjean
    copyright Xerox 2016
    
    READ project 
    
    feature classes 
"""


class  featureObject(object):
    
    """
        a feature created from an element (elt terminal) 
        a feature created from a signature (Node())
            corresponds to the signature
        
        a feature has a type
        the type defines the way the feature is compared
        as well as the fuzzy matching
        
        TYPE1 : comp1, fuzzydistance1
        TYPE2: comp2, fuzzydistance1
        
    """
    NUMERICAL       = 0 # use 
    BOOLEAN         = 1       #   use instead a numerical value ?? 
    EDITDISTANCE    = 2 # textual value
    COMPLEX         = 3       # signature
    PATTERN         = 4      # for content
    DISTANCE        = 5    # two dimension feature [x,y] 
    OBJECT          = 6    # programmatic object with __eq__  (and __hash__)
    ##GRAPHICAL LINES
    ## IMAGES   : ideally, pixel level
    #font: ? fuzzy with bold, italic, fontname, color ...???  -> use traditional font familty tree (serif, sans serif)
    
    
    id  = 0
    
    def __init__(self):

        self._objectName = None ### associate to a document object 
        self._type = None  # numerical 
        
        
        self._name = None
        ##  feature._value can be a list 
        self._value = None
        self._id = featureObject.id

        featureObject.id += 1
        
        self._TH = 0.5
        
        ##12/08/2016
        self._weight = 1.0
        
        
        # take care: refers to the lnde of the canonical feature (most frequent nearest feature)
        # for canonical feature: list of nodes of all nearest features
        self._lnodes= set()
        
        self._bAvailable = True

        self.hash = None
    
        # pointer to the canonical feature (abstract feature)
        self._canonical  = None
        self.bCanonical=False
        
    
    def __hash__(self):
        return hash((self.getName(),self.getStringValue()))
            
    def __repr__(self):
        return  "'%s=%s'" % (self.getName(),self.getStringValue())
    def __str__(self):
        return "'%s=%s'" % (self.getName(),self.getStringValue())
    
    def getID(self): return self._id
    def getClassName(self): return self.__class__.__name__
    def isAvailable(self):
        return  self._bAvailable
    def setAvailable(self): self._bAvailable = True
    def setNonAvailable(self): self._bAvailable = False

    def getWeight(self): return self._weight
    def setWeight(self,w): self._weight = w
    

    def matchLCS(self,perc, t1, t2):
        (s1, n1) = t1
        (s2, n2) = t2
    
        nmax = max(n1, n2)
        nmin = min(n1, n2)
        
        if nmax <=0: return False,0
        #cases that obviously fail
        if nmin < (nmax * perc / 100.0 ): return False, 0
    
        #LCS
        n = self.lcs(s1, s2)
        val = round(100.0 * n / nmax)
        return ((val >= perc), val)
    
        
    
    #--------- LCS code
    # Return the length of the longest common string of a and b.
    def lcs(self,a, b):
    
        na, nb = len(a), len(b)
    
        #switch a and b if b is shorter
        if nb < na:
            a, na, b, nb = b, nb, a, na
    
        curRow = [0]*(na+1)
    
        for i in range(nb):
            prevRow, curRow = curRow, [0]*(na+1)
            for j in range(na):
                if b[i] == a[j]:
                    curLcs = max(1+prevRow[j], prevRow[j+1], curRow[j])
                else:
                    curLcs = max(prevRow[j+1], curRow[j])
                curRow[j+1] = curLcs
        return curRow[na] 

    def __ne__(self,other):
        return not (self == other) 
   
    def __eq__(self,other):
        """
            + _objectName?
        """
        try:other.getClassName()
        except:return False
        
        if self.getClassName() != other.getClassName():
            return False
        if self.getName() == other.getName():
#             print self.getName() , other.getName()
            if self.getType() == other.getType():
                distance = 9e9
                if self._type == featureObject.NUMERICAL:
                    try:
                        distance = abs(float(self.getValue()) - float(other.getValue()))
                    except:
                        pass
                    return distance <= self.getTH()
                elif self._type == featureObject.EDITDISTANCE:
                    # EDITDISTANCE
                    if self.getValue() == other.getValue():
                        return True
                    else:
#                        try:
                            bNear,val = self.matchLCS(self.getTH(), (self.getValue(),len(self.getValue())), (other.getValue(),len(other.getValue())))
                            return bNear #self.getValue() == other.getValue()
#                        except:
#                            print self.getValue(),other.getValue()
#                            ddd
                elif self._type == featureObject.COMPLEX:
                    return self.getValue() == other.getValue()
                elif self._type == featureObject.BOOLEAN:
                    return self.getValue() == other.getValue()
                return False
                
        
        return False
       
       
     
#     def __repr__(self):  return  "(%s=%s %s)" % (self.getName(),self.getValue(),self.getType())
    
    
    def getName(self): return self._name
    def setName(self,n): self._name  = n
 
    def getType(self): return self._type
    def setType(self,t): self._type  = t
    
    
    def setCanonical(self,f): 
        self._canonical = f
    def getCanonical(self): return self._canonical
    
    def getValue(self): return self._value
    def getStringValue(self): 
        try:
            float(self._value)
            return str(self.getValue())
        except ValueError:
            try:
                return str(self.getValue().encode('utf-8'))
            except AttributeError:
                return str(self.getValue())
        except TypeError:
            try:
                return str(self.getValue().encode('utf-8'))
            except AttributeError:
                return str(self.getValue())
    
    
    def setValue(self,v):
        self._value  = v
        return v
    
    def setTH(self,th): self._TH = th
    def getTH(self): return self._TH
           
           
    # specific object from which the feature was generated
    # can be different from nodes
    def setObjectName(self,o): self._objectName = o
    def getObjectName(self): return self._objectName
    
    def addNode(self,n):
        self._lnodes.add(n)
#         try:self._lnodes.index(n)
#         except ValueError:self._lnodes.append(n)
#         if n not in self._lnodes:
#             self._lnodes.append(n)
 
    def getNodes(self): return self._lnodes
    
        


class TwoDFeature(featureObject):
    """
        self.value is define as a tuple: (x,y) 
    """
    
    def __eq__(self,other):
        try: other.getClassName()
        except AttributeError:return False
        if self.getClassName() == other.getClassName():
            if self.getName() == other.getName() and len(self.getValue()) == len(other.getValue()):
                # assume the same semantical order
                for i,x in enumerate(self.getValue()):
                    if abs(float(x) - float(other.getValue()[i])) >self.getTH():
                        return False
                return True
            return False
        return False
                
class multiValueFeatureObject(featureObject):
    """
    
        multivalue featureObject: define as a set of unitary features (of the same type: yes)
        TH: % of common values  
    """
    def __init__(self):
        
        featureObject.__init__(self)
        self._value = []
        self._TH= 1.0
            
    def getStringValue(self):
        if self.getName() == 'EMPTY':
            return 'EMPTY'
        if self.getType() == 1:
            return self.getName()
        return "|".join(map(lambda x:x.getStringValue(),self.getValue()))
            
#     def __str__(self): return self.getStringValue()
#     def __repr__(self): return self.getStringValue()
    
    def __eq__(self,other):
        try: other.getClassName()
        except AttributeError:return False
        if self.getClassName() == other.getClassName():
            if self.getName() == other.getName():
                nbCommon = 0
                for x in self.getValue():
                    if x in other.getValue(): nbCommon +=1
#                 print self, other, self.getTH() , nbCommon,self.getTH() * len(self.getValue()),nbCommon >= ( self.getTH() * len(self.getValue()))
                return nbCommon >= ( self.getTH() * len(self.getValue())) #and self.getTH() * len(other.getValue()))
        return False
class emptyFeatureObject(featureObject):
    
    def __init__(self):
        featureObject.__init__(self)
    
    def __hash__(self):
        sH ="" 
        for s in "EMPTYFEATURE":
            sH += str(ord(s))
        return int(sH)        
     
    def __eq__(self,other):
        if other:
            return other.__class__.__name__ == "emptyFeatureObject"
        return False
    def getName(self): return "EMPTY"
    def __repr__(self):  return  "(EMPTY)"     
    
    

