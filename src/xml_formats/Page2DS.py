# -*- coding: utf-8 -*-
""" 
    H. Déjean

    cpy Xerox 2011
    
    Page 2 DS  (prima format to xerox format)
    http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15
    
"""


import sys, os.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))))

import  libxml2

import common.Component as Component
import config.ds_xml_def as ds_xml
from common.trace import traceln


class primaAnalysis(Component.Component):
    #DEFINE the version, usage and description of this particular component
    usage = "[-f N.N] "
    version = "v1.23"
    description = "description: analysis Prima collection wrt Gird"
    usage = " [-f NN] [-l NN] [--tag=TAGNAME]  "
    version = "$Revision: 1.1 $"
        
    name="primaAnalysis"
    kPTTRN  = "pattern"
    kDPI = "dpi"
    kREF= 'ref'
    kREFTAG= 'reftag'   
    kDOCID= 'docid'     
    def __init__(self):

        Component.Component.__init__(self, "ContentAnalyzor", self.usage, self.version, self.description) 
        self.sPttrn     = None
        self.dpi = 300
        self.bRef = False
        self.lRefTag = ()
         
        self.xmlns='http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'
        
        self.id=1
    def setParams(self, dParams):
        """
        Always call first the Component setParams
        Here, we set our internal attribute according to a possibly specified value (otherwise it stays at its default value)
        """
        Component.Component.setParams(self, dParams)
        if dParams.has_key(self.kPTTRN)    : self.sPttrn     = dParams[self.kPTTRN]
        if dParams.has_key(self.kDPI)    : self.dpi          = int(dParams[self.kDPI])
        if dParams.has_key(self.kREF)    : self.bRef         = dParams[self.kREF]
        if dParams.has_key(self.kREFTAG)    : self.lRefTag   = tuple(dParams[self.kREFTAG])
        if dParams.has_key(self.kDOCID)    : self.sDocID   =dParams[self.kDOCID]


    def regionBoundingBox(self,sList):
        """
            points = (x,y)+ 
        """
        minx = 9e9
        miny = 9e9
        maxx = 0
        maxy = 0        
        lList = sList.getContent().split(' ')
        for x,y in [x.split(',') for x in lList]:
            minx = min(minx,int(x))
            maxx = max(maxx,int(x))
            miny = min(miny,int(y))
            maxy = max(maxy,int(y))
        return [minx,miny,maxy-miny,maxx-minx]
              
    def regionBoundingBox2010(self,lList):
        minx = 9e9
        miny = 9e9
        maxx = 0
        maxy = 0
        for elt in lList:
            if float(elt.prop("x")) < minx: minx = float(elt.prop("x"))
            if float(elt.prop("y")) < miny: miny = float(elt.prop("y"))
            if float(elt.prop("x")) > maxx: maxx = float(elt.prop("x"))
            if float(elt.prop("y")) > maxy: maxy = float(elt.prop("y"))            
        return [minx,miny,maxy-miny,maxx-minx]          
    
    
    def getBBPage(self,lList):

        minx = 9e9
        miny = 9e9
        maxh = 0
        maxw = 0        
        for reg in lList:
            [x,y,h,w] = reg
            if x < minx: minx = x
            if y < miny: miny = y
            if w + x > maxw: maxw = w + x
            if h + y > maxh: maxh = h + y
              
        return [minx,miny,maxh,maxw]
            
    
    
    def getPoints(self,curNode):
        """
            extract polylines, and convert into points(pdf)
        """
        ctxt = curNode.doc.xpathNewContext()
        ctxt.xpathRegisterNs("a", self.xmlns)
        xpath  = "./a:Coords/@%s" % ("points")
        ctxt.setContextNode(curNode)
        lPoints = ctxt.xpathEval(xpath)   
        ctxt.xpathFreeContext()

        if lPoints!= []:
            sp = lPoints[0].getContent().replace(' ',',')
            scaledP=  map(lambda x: 72.0* float(x) / self.dpi,sp.split(','))
            scaledP = str(scaledP)[1:-1].replace(' ','')
            return scaledP
        else:
            return ""
    
    def getTextLineSubStructure(self,dsNode,curNode):
        """
            curNode: TextRegion
                ->TextLine
                
                ->Word 
                    PlainText
        """
        document = curNode.doc
#         xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19"
        ctxt = document.xpathNewContext()
        ctxt.xpathRegisterNs("a", self.xmlns)
        xpath  = "./a:%s" % ("TextLine")
        ctxt.setContextNode(curNode)
        lLines = ctxt.xpathEval(xpath)
        ctxt.xpathFreeContext()
        for line in lLines:
            node = libxml2.newNode('TEXT')
            try:
                node.setProp('id',line.prop('id'))
            except:
                node.setProp('id',str(self.id))
                self.id += 1
                
            dsNode.addChild(node)
            sp= self.getPoints(line)
            # polylines
            node.setProp('points',sp)
            # BB
            ctxt = curNode.doc.xpathNewContext()
            ctxt.xpathRegisterNs("a", self.xmlns)
            xpath  = "./a:Coords/@%s" % ("points")
            ctxt.setContextNode(line)
            lPoints = ctxt.xpathEval(xpath)
            if lPoints != []:
                [x,y,h,w] = self.regionBoundingBox(lPoints[0])
                xp,yp,hp,wp  = map(lambda x: 72.0* x / self.dpi,(x,y,h,w))
                node.setProp(ds_xml.sX,str(xp))
                node.setProp(ds_xml.sY,str(yp))
                node.setProp(ds_xml.sHeight,str(hp))
                node.setProp(ds_xml.sWidth,str(wp))            
            node.setProp('font-size','20')

            ## baseline
            ## add @baselintpoints 
            ##<Baseline points="373,814 700,805 1027,785 1354,804 1681,783 2339,780"/>
#             blnode = libxml2.newNode('BASELINE')
            ctxt = line.doc.xpathNewContext()
            ctxt.xpathRegisterNs("a", self.xmlns)
            ctxt.setContextNode(line)
            xpath  = "./a:Baseline/@%s" % ("points")
            lPoints = ctxt.xpathEval(xpath)
            ctxt.xpathFreeContext()
            if lPoints!= []:
                ## 
#                 sp = self.getPoints(curNode)
                sp = lPoints[0].getContent().replace(' ',',')
                try:
                    scaledP=  map(lambda x: 72.0* float(x) / self.dpi,sp.split(','))
                    scaledP = str(scaledP)[1:-1].replace(' ','')
                    node.setProp('blpoints',scaledP)
                    dsNode.addChild(node)
                except IndexError: 
                    pass
                
#             lPoints = ctxt.xpathEval(xpath)
#             if lPoints != []:
#                 [x,y,h,w] = self.regionBoundingBox(lPoints[0])
#                 xp,yp,hp,wp  = map(lambda x: 72.0* x / self.dpi,(x,y,h,w))
#                 blnode.setProp(ds_xml.sX,str(xp))
#                 blnode.setProp(ds_xml.sY,str(yp))
#                 blnode.setProp(ds_xml.sHeight,str(hp))
#                 blnode.setProp(ds_xml.sWidth,str(wp))            
            
            ctxt = line.doc.xpathNewContext()
            ctxt.xpathRegisterNs("a", self.xmlns)
            xpath  = "./a:%s" % ("Word")
            ctxt.setContextNode(line)
            lWords= ctxt.xpathEval(xpath)
            ctxt.xpathFreeContext()
            if lWords == []:
                # get TextEquiv
                ctxt = line.doc.xpathNewContext()
                ctxt.xpathRegisterNs("a", self.xmlns)
                xpath  = "./a:TextEquiv/a:Unicode"
                ctxt.setContextNode(line)
                ltxt = ctxt.xpathEval(xpath)
                ctxt.xpathFreeContext()
                if ltxt != []:
                    #node.setContent(ltxt[0].getContent())
                    lCWords= ltxt[0].getContent().split(' ')
                    for word in lCWords:
                        wnode= libxml2.newNode('TOKEN')
                        wnode.setProp('font-color','BLACK')
                        wnode.setProp('font-size','8')
                        wnode.setContent(word)     
                        node.addChild(wnode)
               
                    
            for word in lWords:
                wnode= libxml2.newNode('TOKEN')
                wnode.setProp('font-color','BLACK')
                wnode.setProp('font-size','8')
                node.addChild(wnode)
                ctxt = curNode.doc.xpathNewContext()
                ctxt.xpathRegisterNs("a", self.xmlns)
                xpath  = ".//a:%s" % ("Points")
                ctxt.setContextNode(word)
                lPoints = ctxt.xpathEval(xpath)
                ctxt.xpathFreeContext()
                if lPoints != []:
                    [x,y,h,w] = self.regionBoundingBox(lPoints[0])
                    xp,yp,hp,wp  = map(lambda x: 72.0* x / self.dpi,(x,y,h,w)) 
                    wnode.setProp(ds_xml.sX,str(xp))
                    wnode.setProp(ds_xml.sY,str(yp))
                    wnode.setProp(ds_xml.sHeight,str(hp))
                    wnode.setProp(ds_xml.sWidth,str(wp))                            
                    ctxt = word.doc.xpathNewContext()
                    ctxt.xpathRegisterNs("a", self.xmlns)
                    xpath  = './a:TextEquiv/a:PlainText/text()'
                    ctxt.setContextNode(word)
                    ltexts= ctxt.xpathEval(xpath)
                    ctxt.xpathFreeContext()
                    if ltexts:
                        wnode.setContent(document.encodeEntitiesReentrant(ltexts[0].getContent()))
                 
                 
    def run(self):
        
        dsdom = libxml2.newDoc("1.0")
        dsroot = libxml2.newNode(ds_xml.sDOCUMENT)
        dsdom.setRootElement(dsroot)
        

        import glob
#         xmlns='http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'
#         xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19"
        ipageNumber = 1
        for pathname in sorted(glob.iglob(self.sPttrn)):
                #pathname = it.next()
                print pathname
                primedoc = self.loadDom(pathname)
                page = libxml2.newNode(ds_xml.sPAGE)
                page.setProp(ds_xml.sPageNumber,str(ipageNumber))
                ipageNumber += 1
                dsroot.addChild(page)            
                ctxt = primedoc.xpathNewContext()
                ctxt.xpathRegisterNs("a", self.xmlns)
                xpath  = "//a:%s" % ("Page")
                lPages = ctxt.xpathEval(xpath)
                ctxt.xpathFreeContext()
                lRegion =[]
                for ipage in lPages:
                    page.setProp("imageFilename",'..%scol%s%s%s'%(os.sep,os.sep,self.sDocID,os.sep)+ ipage.prop("imageFilename"))
                    imageWidth =  72 * (float(ipage.prop("imageWidth"))  / self.dpi)
                    imageHeight = 72 * (float(ipage.prop("imageHeight")) / self.dpi)
                    page.setProp("width",str(imageWidth))
                    page.setProp("height",str(imageHeight))
#                    imgNode = libxml2.newNode("IMAGE")
#                    imgNode.setProp("href",ipage.prop("imageFilename"))
#                    imgNode.setProp("x","0")
#                    imgNode.setProp("y","0")
#                    imgNode.setProp("height",str(imageHeight))
#                    imgNode.setProp("width",str(imageWidth))
#                    page.addChild(imgNode)
                    child = ipage.children
                    while child:
                        if child.type == "element":
                            if not self.bRef or (self.bRef and child.name in self.lRefTag): 
                                ## get points
                                ctxt = primedoc.xpathNewContext()
                                lpoints = self.getPoints(child)
                                ctxt.xpathRegisterNs("a", self.xmlns)
                                xpath  = "./a:Coords/@%s" % ("points")
                                ctxt.setContextNode(child)
                                lPoints = ctxt.xpathEval(xpath)
                                if lPoints !=[]:
                                    ctxt.xpathFreeContext()
                                    [x,y,h,w] = self.regionBoundingBox(lPoints[0])
                                    xp,yp,hp,wp  = map(lambda x: 72.0* x / self.dpi,(x,y,h,w)) 
                                    if child.name == "TextRegion":
                                        #get type
                                        node = libxml2.newNode("REGION")
                                        node.setProp('type',child.prop('type') )
                                        self.getTextLineSubStructure(node,child)
                                    elif child.name =="ImageRegion":
                                        node = libxml2.newNode("IMAGE")
                                    elif child.name =="LineDrawingRegion":
                                        node = libxml2.newNode("IMAGE")
                                    elif child.name =="GraphicRegion":
                                        node = libxml2.newNode("IMAGE")
                                    elif child.name =="SeparatorRegion":
                                        node = libxml2.newNode("SeparatorRegion")
                                        sp= self.getPoints(child)
                                        # polylines
                                        node.setProp('points',sp)                                         
                                    elif child.name =="TableRegion":
                                        node = libxml2.newNode("TABLE")
                                    elif child.name =="FrameRegion":
                                        node = libxml2.newNode("FRAME")
                                    elif child.name =="ChartRegion":
                                        node = libxml2.newNode("FRAME")
                                    elif child.name =="MathsRegion":
                                        node = libxml2.newNode("MATH")
                                    elif  child.name =="PrintSpace":
                                        node = libxml2.newNode("typeArea")                                     
                                    else:
                                        node = libxml2.newNode("MISC")
                                    ## ADD ROTATION INFO
                                    if child.hasProp("orientation"):
                                        rotation = child.prop("orientation")
                                        if float(rotation) == 0:
                                            node.setProp("rotation","0")
                                        elif float(rotation) == -90:
                                            node.setProp("rotation","1")
                                        elif float(rotation) == 90:
                                            node.setProp("rotation","2")
                                        elif float(rotation) == 180:
                                            node.setProp("rotation","3")           
                                    node.setProp(ds_xml.sX,str(xp))
                                    node.setProp(ds_xml.sY,str(yp))
                                    node.setProp(ds_xml.sHeight,str(hp))
                                    node.setProp(ds_xml.sWidth,str(wp))
                                    page.addChild(node)
        
                        child = child.next
#         except StopIteration, e:
#             traceln("=== done.")
        self.addTagProcessToMetadata(dsdom)
        return dsdom
                
if __name__ == "__main__":
    
    #command line
    traceln( "=============================================================================")
    
    cmp = primaAnalysis()

    #prepare for the parsing of the command line
    cmp.createCommandLineParser()
    cmp.add_option("", "--"+cmp.kPTTRN, dest=cmp.kPTTRN, action="store", type="string", help="REQUIRED **: File name pattern, e.g. /tmp/*/to?o*.xml"   , metavar="<pattern>")
    cmp.add_option("--dpi", dest="dpi", action="store",  help="image resolution")
    cmp.add_option("--ref", dest="ref", action="store_true", default=False, help="generate ref file")
    cmp.add_option("--reftag", dest="reftag", action="append",  help="generate ref file")
    cmp.add_option("--"+cmp.kDOCID, dest=cmp.kDOCID, action="store", type='string', help="docId in col")
    
 
    #parse the command line
    dParams, args = cmp.parseCommandLine()
    
    #Now we are back to the normal programmatic mode, we set the componenet parameters
    cmp.setParams(dParams)
    doc = cmp.run()
    cmp.writeDom(doc, True)

