# -*- coding: utf-8 -*-
"""
    document class 
    Hervé Déjean
    cpy Xerox 2013

    a class for document
"""

import sys, os.path

sys.path.append(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))

from xmlDocumentClass import XMLDocument
from XMLDSObjectClass import XMLDSObjectClass
from XMLDSPageClass import XMLDSPageClass

from config import ds_xml_def as ds_xml

class  XMLDSDocument(XMLDocument):
    """ 
        Representation of a document"
        XML format  (same for html->hierarchical stuff html is a subclass of xml ?  (specific loaddom with fuzzy parsing))
    """
    
    def __init__(self,domDoc=None):
        self._type = "XMLDSDocument"
        self._dom = domDoc
        self._rootObject = None
        self.lPages = []
        self.currentlPages = []
        self.nbTotalPages = 0
        XMLDocument.__init__(self)        
    
    
    def addPage(self,p):
        self.lPages.append(p)
        
    def getPages(self): return self.lPages
    
    def loadPages(self,domdocRoot,myTAG,lPages):
        """
            load PAGE elements
        """
        ctxt = self._dom.xpathNewContext()
        ctxt.setContextNode(domdocRoot)
        ldomPages = ctxt.xpathEval('.//%s'% (myTAG))
        ctxt.xpathFreeContext()
        self.nbTotalPages =  len(ldomPages)
        if lPages == []:
            lPages = range(1,self.nbTotalPages+1)
        self.currentlPages = lPages
        for page in ldomPages:
            if page.hasProp('number') and int(page.prop('number')) in lPages:
                myPage= XMLDSPageClass(page)
                myPage.setNumber(int(page.prop('number')))
                self.addPage(myPage)
#                 self.getRootObject()._lObjects = self.getPages()
                myPage.fromDom(page,['REGION','BLOCK',ds_xml.sLINE_Elt,ds_xml.sTEXT,'BASELINE','GRAPHELT'])

    
    def loadFromDom(self,docDom = None,pageTag='PAGE',listPages = []):
        """
            For each node, create an object with the tag name. features?
        
        
            assume DOCUMENT/PAGE  
        """
        if docDom:
            self.setDom(docDom)
        if self.getDom():
            self._rootObject = XMLDSObjectClass()
            # get pages:
            self.loadPages(self.getDom().getRootElement(),ds_xml.sPAGE,listPages)
        else:
            return -1
        

    def display(self,lvl=0):
        print 'Document: ',self.getName()
        self.getRootObject().display(lvl+1)

