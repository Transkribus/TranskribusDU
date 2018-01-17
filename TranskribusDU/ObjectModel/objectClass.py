# -*- coding: utf-8 -*-
"""

    object class 
    
    Hervé Déjean
    cpy Xerox 2009
    
    a class for object

"""
from __future__ import absolute_import
from __future__ import  print_function
from __future__ import unicode_literals
from .sequenceAPI import sequenceAPI

class  objectClass(sequenceAPI):
    """
        object class
    """
    
    id = 0
    def __init__(self):
        sequenceAPI.__init__(self)
        self._name = None
        self._parent = None
        # sub object
        self._lObjects = []
        
        # characteristics
        self._lAttributes = {}

        self._content = None
        
        
        # list of attributes used as features
        self._myFeatures = []
        # associated list of multivalued features (used for parsing)
        self._myMVFeatures = []
        
        
        # list of structures
        self._lStructures = []
        
        
        
        ## FOR IE 
        self._lFields = []
        
    
    def __str__(self): return self.getName()
    def __repr__(self): return "%s"%self.getName()
    
    def getName(self): return self._name
    def setName(self,n): self._name  = n
    def getParent(self): return self._parent
    def setParent(self,o): self._parent = o
    
    def getObjects(self): 
        return self._lObjects
    
    def setObjectsList(self,l): self._lObjects = l
    def addObject(self,o): 
        if o not in self.getObjects():
            self.getObjects().append(o)
            o.setParent(self)
            
    def addContent(self,c):
        c=c.replace(u'\n',u' ')
        if self.getContent() is not None:
            self._content +=  u' ' + c
        else:
            self._content =  u' ' + c
        self._content= self._content.strip()
    def setContent(self,c): self._content = c.strip()
    
    def getContent(self):
        if self._content is not None:
            return self._content
        else: 
            if self._lObjects != []:
                c = u''
                for x in self._lObjects:
                    if x.getContent() is None: pass
                    else:c+= x.getContent()+ u' '
                self._content = c.strip()
                return self._content
        return self._content
            
    def print_(self):
        print(self.getContent())
    
    def getAllNamedObjects(self,objectName):
        
        lList = []
        ## current elements  
        try:
            if isinstance(self,objectName):
                lList.append(self)
        except TypeError:
            if self.getName() == objectName:
                lList.append(self)
                
        # children
        for elt in self.getObjects():
            lList.extend(elt.getAllNamedObjects(objectName))
        return lList
            
    def getNamedObjects(self,objName):
        ##
        l= []
        for o in self.getObjects():
            try:
                if isinstance(o,objName):
                    l.append(o)
            except TypeError:
                if self.getName() == objName:
                    l.append(o)
        return l
    
        
    def addAttribute(self,name,value): 
#         try: self._lAttributes[name] =value
#         except KeyError: self. _lAttributes[name] = value
        self._lAttributes[name] = value
        
    def hasAttribute(self,name):
        try:
            self._lAttributes[name]
            return True
        except KeyError: 
            return False
    
    def hasAttributes(self): return self._lAttributes != {}
    def getAttributes(self): 
        return self._lAttributes
    
    def getAttribute(self,name): 
        try: return self._lAttributes[name]
        except KeyError: return None


    def display(self,level=0):
        margin = " " * level
        print (margin,self.getName(), self.getContent()[:10])

        for obj in self.getObjects():
            obj.display(level+1)
            
    ########### IE part ########### 
    ### move to objectClass?
    
    def addField(self,field,value=None):
        """
            add field (record field) to this cell: this cell is supposed to contain such a field
        """
        if field not in self.getFields():
            self.getFields().append(field)
        return field

    def getFieldByName(self,name):
        lName = filter(lambda x:x.getName() == name,self.getFields())
        if lName == []:
            return None
        else:
            return lName[0]
        
    def getFields(self): return self._lFields
    def getAllFields(self):
        lF=self._lFields
        [lF.extend(x.getFields()) for x in self.getObjects()]
        return lF    
    
    def extractFields(self):
        """
            extract fields
        """
        for field in self.getFields():
            #take first tagger
            if field.getTagger() != []:
                value = field.getTagger()[0].parse(self.getContent())
                field.setValue(value)    
            
            
    # also in zoneClass
    def getMyStructures(self):return self._lStructures
    def addStructure(self,s): self._lStructures.append(s)
    def setStructures(self,l): self._lStructures  = l
    def setStructure(self,l): self._lStructures= l            
