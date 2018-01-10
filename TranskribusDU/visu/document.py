"""The classes needed to deal with collections, models, documents...
  
$author: Jerome Fuselier
$since: October 2005
"""
 
import libxml2


   
class Document:
    """The class for the document that is displayed in the interface"""
    
    def __init__(self, path, config):
        """Default constructor.
        @param path: the path of the file to load
        @type path: str
        """    
        print "Loading..."
        self.filename = path 

        self.tree_dom = libxml2.parseFile(self.filename)
        self.root_dom = self.tree_dom.getRootElement()
        
        #create a context for our XPath queries
        self.ctxt = self.tree_dom.xpathNewContext()
        self.config = config
        
        #register any namespace
        for sName, sURI in config.getNamespaceList():
            print "Document: XPath namespace: %s=%s"%(sName, sURI)
            self.ctxt.xpathRegisterNs(sName, sURI)
        config.setXPathContext(self.ctxt)   #in order to propagate this context to the decoration objects
        
        self.lPage = []      #list of page elements in document order
        self.dNum2PageIndex = {}  #dictionary @number -> page index
        page_num_attr = config.getPageNumberAttr()
        xpExpr = "//%s"%config.page_tag
        ln = self.ctxt.xpathEval(xpExpr)
        i = 0
        for n in ln:
            self.lPage.append(n)
            self.ctxt.setContextNode(n)
            try:
              pagenum = self.ctxt.xpathEval(page_num_attr)[0].getContent()
            except IndexError:
              print "WARNING: no number attribute on page"
              pagenum = i+1
            print pagenum, " ",
            #.page_num_attr
            #self.dNum2PageIndex[n.prop('number')] = i
            self.dNum2PageIndex[pagenum] = i
            i += 1
        self.maxi = len(self.lPage)
        print
        print "Document: ", path, " with %d pages"%self.maxi
    
    def free(self):
        """
        free the memory used by libxml2
        """
        self.ctxt.xpathFreeContext()
        self.root_dom = None
        del self.lPage
        del self.dNum2PageIndex
        self.tree_dom.freeDoc()
        self.tree_dom = None
        
    def saveXML(self, filename):
        try:
            f = open(filename, "w")
            bIndent = False
            if bIndent:
                self.tree_dom.saveFormatFile(f, bIndent)
            else: 
                self.tree_dom.dump(f)
            f.close()
            print "XML SAVED INTO", filename
            return True
        except:
            print "" #beep
            print "##########################################################################"
            print "# ERROR: cannot save the XML into ", filename 
            print "##########################################################################"
            print "" #beep
            return False
            
    def getXPCtxt(self):
        return self.ctxt
 
    def xpathEval(self, xpExpr, node=None):
        """evaluate an xpath in the context of this document or this node"""
        n = node or self.root_dom
        self.ctxt.setContextNode(n)
        return self.ctxt.xpathEval(xpExpr)


    def getNextPageIndex(self):
        return (self.displayed + 1) % self.maxi
    
    def getPrevPageIndex(self):
        return (self.displayed - 1) % self.maxi
    
    def new_page(self, i):
        """we are going to display this new page"""
        self.obj_n = {}
        self.obj_deco = {}
        self.displayed = i
    
    def getCurrentPageNode(self):
        return self.lPage[self.displayed]
    
    def getDisplayedIndex(self):
        return self.displayed
        
    def getPageNumber(self, i):
        """return the page @number of the Ith page, as a string"""
        #return self.lPage[i].prop("number")
        self.ctxt.setContextNode(self.lPage[i])
        return self.ctxt.xpathEval(self.config.getPageNumberAttr())[0].getContent()
        
    def getPageIndexByNumber(self, num):
        """return the page having num as @number"""
        return self.dNum2PageIndex[str(num)]

    def getPageByIndex(self, i):
        """return the ith page, i is in [0, N-1] for N pages""" 
        return self.lPage[i]
    
    def getPageCount(self):
        return self.maxi
    
    def getDOM(self):
        return self.tree_dom
#
    def getDOMRoot(self):
        return self.root_dom
        
    def getFilename(self):
        return self.filename
        
        
        
    
    
    
