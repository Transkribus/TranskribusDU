# -*- coding: utf-8 -*-
"""

    DS2PageXml.py
    some functionality to handle pageXML files (add regions,...)

    Copyright Xerox(C) 2016 H. Déjean


    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union’s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""
import sys, os
import libxml2

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version
    
from common.Component import Component
from common.trace import traceln, trace

from xml_formats.PageXml import PageXml
from ObjectModel.xmlDSDocumentClass import XMLDSDocument

from util.unitConversion import convertDot2Pixel

class DS2PageXMLConvertor(Component):
    """
        conversion from DSXML to PageXML
    """
    #DEFINE the version, usage and description of this particular component
    usage = "" 
    version = "v.01"
    description = "description: DS2PageXml conversion"

    
        
    #--- INIT -------------------------------------------------------------------------------------------------------------    
    def __init__(self):
        """
        """
        Component.__init__(self, "DS2PageXml", self.usage, self.version, self.description) 
    
        self.dpi = 300

        self.xrce_id=10000        
        
        self.storagePath = ''
        
        self.dTagNameMapping = {'PAGE':'Page','TEXT':'TextLine', 'REGION':'TextRegion','BLOCK':'TextRegion','GRAPHELT':'LineDrawingRegion','TABLE':'TableRegion','CELL':'TableCell'} 

        self.pageXmlNS = None
        
        self.bMultiPages = False
        self.bRegionOnly = False
        
    def setParams(self, dParams):
        """
        Always call first the Component setParams
        """
        Component.setParams(self, dParams)
        if dParams.has_key("bMultiPage"): self.bMultiPages =  dParams["bMultiPage"]  
        if dParams.has_key("bRegionOnly"): self.bRegionOnly =  dParams["bRegionOnly"]  
        
    
    def setDPI(self,v): self.dpi=v
    
    def setStoragePath(self,p): self.storagePath=p

    def getCoord(self,DSObject):
        """
            create Coords value (polylines) from BoundingBox
            if object has @points: return points
        """
        
        if DSObject.hasAttribute('points'):
            return DSObject.getAttribute('points')
        else:
            return self.BB2Polylines(DSObject.getX(), DSObject.getY(),DSObject.getHeight(), DSObject.getWidth())
        
    def BB2Polylines(self,x,y,h,w):
        """
            convert BB values to polylines coords
        """
        # schema does not support neg int
        if x <0: x = abs(x)  
        if y <0: y = abs(y)
        
        lx= map(lambda x:1.0*x*self.dpi/72.0, ( x,y, x+w,y, x+w,y+h, x,y+h, x,y))
        myPoints = ' '.join(["%d,%d"%(xa,ya) for xa,ya  in zip(lx[0::2], lx[1::2])])
        return myPoints    
        
    def DSPoint2PagePoints(self,sPoints):
        """    
            "52.8,47.76,128.88,44.64"
            to
            451,246 451,1094 781,1094 781,246
            
        """
        lPoints = sPoints.split(",")
        lx= map(lambda x:1.0*float(x)*self.dpi/72.0, lPoints)
        # order left right 
        xx =  zip(lx[0::2], lx[1::2])
        xx.sort(key=lambda (x,y):x)
        return ' '.join(["%d,%d"%(xa,ya) for xa,ya  in xx])
        
    def convertDSObject(self,DSObject,pageXmlParentNode):
        """
            convert DSObject and add it as child to pageXmlParentNode
            
             
             <TextLine id="line_1472550984091_215" custom="readingOrder {index:0;}">
                <Coords points="218,65 280,65 280,100 218,100"/>
                <Baseline points="218,95 280,95"/>
                <TextEquiv>
                    <Unicode>10.</Unicode>
                </TextEquiv>
            </TextLine>            
            
            
        for table:
        <TableRegion id="Table_1484215666379_5" custom="readingOrder {index:92;}">
            <Coords points="221,246 781,246 781,1094 221,1094"/>
            <TableCell row="0" col="0" colSpan="1" id="TableCell_1484215672011_8">
                <Coords points="221,246 221,1094 451,1094 451,246"/>
                <CornerPts>0 1 2 3</CornerPts>
            </TableCell>
            <TableCell row="0" col="1" colSpan="1" id="TableCell_1484215672011_7">
                <Coords points="451,246 451,1094 781,1094 781,246"/>
                <CornerPts>0 1 2 3</CornerPts>
            </TableCell>
        </TableRegion>        
        
            
            
            
            DS TEXT
             <TEXT x="52.8" y="41.04" height="6.72" width="76.08" font-size="20" y2="47.76" x2="128.88"
                points="52.8,41.04,128.88,41.04,128.88,47.76,52.8,47.76" blpoints="52.8,47.76,128.88,44.64" type="RB" id="p1_CVL-{e0004b40-06e0-4b85-97a0-2df0e1f0fe87}"/>
        """
        try:
            pageXmlName= self.dTagNameMapping[DSObject.getName()]
        except KeyError: 
#             print DSObject.getName() ," not declared"
            return 
                    
        domNode= PageXml.createPageXmlNode(pageXmlName,self.pageXmlNS)
        if DSObject.getID():
            domNode.setProp("id", "xrce_%s"%DSObject.getID())
        else: self.addXRCEID(domNode)
        pageXmlParentNode.addChild(domNode)
        
        coordsNode = libxml2.newNode('Coords')
        coordsNode.setNs(self.pageXmlNS)
        coordsNode.setProp('points', self.BB2Polylines(DSObject.getX(),DSObject.getY(), DSObject.getHeight(),DSObject.getWidth()))        
        domNode.addChild(coordsNode)            
        
        
        for attr in ['custom', 'structure','col']:
            if DSObject.hasAttribute(attr):
                domNode.setProp(attr, DSObject.getAttribute(attr))
        # if blpoints:  build Baseline        
        
        # type 
        if DSObject.hasAttribute('type'):
            domNode.setProp('type', DSObject.getAttribute('type'))
        # if blpoints:  build Baseline
        # <Baseline points="218,95 280,95"/>
        ## Baseline needs to be left-right!!
        if DSObject.hasAttribute('blpoints'):
            domBaseLine=libxml2.newNode('Baseline')
            domBaseLine.setNs(self.pageXmlNS)
            domBaseLine.setProp('points', self.DSPoint2PagePoints(DSObject.getAttribute('blpoints')))        
            domNode.addChild(domBaseLine)                
            
        
        # collect content and generate a textequiv
        
        ## specific attributes for cell
        ###  row="0" col="2" colSpan="1
        if pageXmlName == 'TableCell':
            domNode.setProp('row',str(DSObject.getIndex()[0]))
            domNode.setProp('col',str(DSObject.getIndex()[1]))
            cornerNode = libxml2.newNode('CornerPts')
            cornerNode.setContent("0 1 2 3")
            cornerNode.setNs(self.pageXmlNS)
            domNode.addChild(cornerNode)    
            coordsNode.setProp('colSpan',str(DSObject.getColSpan()))
            coordsNode.setProp('rowSpan',str(DSObject.getRowSpan()))
            
        
        #process objects
        for subobject in DSObject.getObjects():
            self.convertDSObject(subobject, domNode)
        
        
    def addXRCEID(self,node):
        node.setProp("id", "xrce_%d"%self.xrce_id)  
        self.xrce_id += 1
        
        
    
    
    def convertOnlyRegion(self,OPage,pageXmlPageNODE):
        """
            populate pageXml with OPageRegion
        """
        # get REGION elements
        lElts= OPage.getAllNamedObjects('REGION')
        for DSObject in lElts:
            self.convertDSObject(DSObject,pageXmlPageNODE)        
        
    def convertDSPage(self,OPage,pageXmlPageNODE):
        """
            populate pageXml with OPage
        """
        from ObjectModel.XMLDSGRAHPLINEClass import XMLDSGRAPHLINEClass
        from ObjectModel.XMLDSTEXTClass import XMLDSTEXTClass
        from ObjectModel.XMLDSTABLEClass import XMLDSTABLEClass

#         # TextRegion needed: create a fake one with BB zone?
#         regionNode= PageXml.createPageXmlNode("TextRegion",self.pageXmlNS)
#         pageXmlPageNODE.addChild(regionNode)
#         self.addXRCEID(regionNode)
#         coordsNode = libxml2.newNode('Coords')
#         coordsNode.setNs(self.pageXmlNS)
#         coordsNode.setProp('points', self.BB2Polylines(0,0, OPage.getHeight(),OPage.getWidth()))
#         regionNode.addChild(coordsNode)     
        
        ##get table elements
        lElts= OPage.getAllNamedObjects(XMLDSTABLEClass)
        for DSObject in lElts:
            self.convertDSObject(DSObject,pageXmlPageNODE)        
        
        # get textual elements
        lElts= OPage.getAllNamedObjects('REGION')
        for DSObject in lElts:
            self.convertDSObject(DSObject,pageXmlPageNODE)

#         # get graphelt elements
        lElts= OPage.getAllNamedObjects(XMLDSGRAPHLINEClass)
        for DSObject in lElts:
            self.convertDSObject(DSObject,pageXmlPageNODE)
        
        # get table elements
    
    def storePageXmlSetofFiles(self,lListOfDocs):
        """
            write on disc the list of dom in the PageXml format
        """
        for i,(doc,img) in enumerate(lListOfDocs):
            if self.storagePath == "":
                self.outputFileName = os.path.dirname(self.inputFileName)+os.sep+img[:-3]+"_%.4d"%(i+1) + ".xml"
            else:
                self.outputFile = self.storagePath + os.sep+img[:-3]+"_%.4d"%(i+1) + ".xml"
#             print "output: %s" % self.outputFileName
            try:self.writeDom(doc, bIndent=True)
            except IOError:return -1            
        return 0
    
    def storeMultiPageXml(self,lListDocs,outputFileName=None):
        """
            write a multipagePageXml file
        """
        from xml_formats.PageXml import MultiPageXml
        mp = MultiPageXml()
        newDoc = mp.makeMultiPageXmlMemory(map(lambda (x,y):x,lListDocs))
        if outputFileName is None:
            outputFileName = os.path.dirname(self.inputFileName) + os.sep + ".."+os.sep +"col" + os.sep + os.path.basename(self.inputFileName)[:-7] + "_du.mpxml"
        res= newDoc.saveFormatFileEnc(outputFileName, "UTF-8",True)
#         print res
#         print "output: %s" % outputFileName
        
    def run(self,domDoc):
        """
            conversion
        """
        ODoc =XMLDSDocument()
        ODoc.loadFromDom(domDoc)
        lPageXmlDoc=[]
        lPages= ODoc.getPages()   
        for page in lPages:
#             traceln("%s %s"%(page, page.getAttribute('imageFilename')))
            pageXmlDoc,pageNode = PageXml.createPageXmlDocument(creatorName='NLE', filename = os.path.basename(page.getAttribute('imageFilename')), imgW = convertDot2Pixel(self.dpi,page.getWidth()), imgH = convertDot2Pixel(self.dpi,page.getHeight()))
            self.pageXmlNS = pageXmlDoc.getRootElement().ns()
#             print "??",self.bRegionOnly
            if self.bRegionOnly:
                self.convertOnlyRegion(page, pageNode)
            else:
                self.convertDSPage(page,pageNode)
            #store pageXml
            lPageXmlDoc.append((pageXmlDoc,page.getAttribute('imageFilename')))
#             print pageXmlDoc.serialize('UTF-8', 1)
#             res= PageXml.validate(pageXmlDoc.doc)
#             print "document is valid:", res 
        
        return lPageXmlDoc
    
if __name__ == "__main__":
    
    
    docConv = DS2PageXMLConvertor()

    #prepare for the parsing of the command line
    docConv.createCommandLineParser()
    docConv.add_option("-m", "--multi", dest="bMultiPage", action="store_true", default=False, help="store as multipagePageXml", metavar="B")
    docConv.add_option("-r", "--region", dest="bRegionOnly", action="store_true", default=False, help="convert only regions", metavar="B")
      
    #parse the command line
    dParams, args = docConv.parseCommandLine()
    
    #Now we are back to the normal programmatic mode, we set the componenet parameters
    docConv.setParams(dParams)
    doc = docConv.loadDom()
    lPageXml = docConv.run(doc)
#     print docConv.bMultiPages
    if lPageXml != []:# and docM.getOutputFileName() != "-":
        if docConv.bMultiPages:
            docConv.storeMultiPageXml(lPageXml)
        else:
            docConv.storePageXmlSetofFiles(lPageXml)

    