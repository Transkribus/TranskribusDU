# -*- coding: utf-8 -*-
"""

    Hervé Déjean
    cpy Xerox 2016
    
    READ project
    
    treeTemplate Class
    
"""

from treeTemplateClass import treeTemplateClass

class virutalTreeTemplateClass(treeTemplateClass):
    """
        a virtual node for tree template
        
    """
    
    def __init__(self):
        treeTemplateClass.__init__(self)
        
        self.virtual = True
        
    def __str__(self):return 'virtualTreeTemplate:%s'%(self.getPattern())
    def __repr__(self):return 'virtualTreeTemplate:%s'%(self.getPattern())

    def buildTreeFromPattern(self,pattern):
        """
             create a tree structure corresponding to pattern
        """
        self.setPattern(pattern)
#         print 'creation:',self.getPattern()
        if isinstance(pattern,list):
            for child in self.getPattern():
                # how to identify if this a virtual or not?
                ctemplate  = treeTemplateClass()
#                 ctemplate.setPattern(child)
                ctemplate.buildTreeFromPattern(child)
                self.addChild(ctemplate)
                ctemplate.setParent(self)
#             print '--',self.getChildren()
        else:
            pass
            #terminal
            
    
    def registration(self,anobject):
        """
            'register': match  the model to an object
            can only a terminal template 
        """

        if anobject == self.getPattern():
            return self.getPattern(),None,1        
        else:
            return None,None,-1
        
















    