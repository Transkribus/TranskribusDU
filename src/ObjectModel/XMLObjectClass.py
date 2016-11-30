# -*- coding: latin-1 -*-
"""

    XML object class 
    
    Hervé Déjean
    cpy Xerox 2009
    
    a class for object from a XMLDocument

"""

from objectClass import objectClass

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
        # needed for mapping dataobject to layoutObject
        self._nextObject = None
        self._previousObject = None
    
    def __repr__(self):
        return "%s[%s] %s" % (self.getName(),self.getID(),self.getContent().encode('utf-8'))
        return "%s %s %s" % (self.getName(),self.getContent().encode('utf-8'),self.getAttributes())
    
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
    
    def fromDom(self,domNode):
        
        ## if domNode in mappingTable:
        ## -> ise the fromDom of the specific object
        
        self._name = domNode.name
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
         
        
