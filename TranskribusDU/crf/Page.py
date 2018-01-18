# coding: utf8

'''
This code is about the graph we build - edges and nodes   (nodes are called Blocks) and page

JL Meunier
Dec 12th, 2016


Copyright Xerox 2016

'''


class Page:
        
    def __init__(self, pnum, pagecnt, w, h, cls=None, domnode=None, domid=None):
        """
        pnum is an int
        cls is the node label, is an int in N+
        """
        self.pnum = int(pnum)
        self.pagecnt = pagecnt    #part of a document of this number of pages
        self.w = w
        self.h = h
        self.node = domnode
        self.domid = domid
        self.cls = cls #the class of the block, in [0, N]
        
        self.bEven = (pnum%2 == 0)

    def detachFromDOM(self): 
        """
        Erase any pointer to the DOM so that we can free it.
        """
        self.node.clear()
        self.node = None
    
    ##Bounding box methods: getter/setter + geometrical stuff
    def getBB(self):
        return self.x1, self.y1, self.x2, self.y2
    def setBB(self, (x1, y1, x2, y2)):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
 
    def getWidthHeight(self):
        return self.w, self.h
    
    def area(self):
        return self.w * self.h
    

    def __str__(self):
        return "Page id=%s page=%d (%f, %f)" %(self.domid, self.pnum, self.w, self.h)
    
