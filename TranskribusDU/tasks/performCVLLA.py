# -*- coding: utf-8 -*-
"""


    performCVLLA.py

    create profile for nomacs (CVL LA toolkit)
    H. Déjean
    
    copyright Xerox 2017
    READ project 

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
"""



import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))


import glob
import common.Component as Component
from common.trace import traceln

from xml_formats.PageXml import PageXml
from xml_formats.PageXml import MultiPageXml

from util.Polygon import Polygon
from lxml import etree
  
class LAProcessor(Component.Component):
    """
        
    """
    usage = "" 
    version = "v.01"
    description = "description: Nomacs LA processor"

    if sys.platform == 'win32':
        cNomacs = '"C:\\Program Files\\READFramework\\bin\\nomacs.exe"'

    else:
        cNomacs = "/opt/Tools/src/tuwien-2017/nomacs/nomacs"


     
    cCVLLAProfile = """
[%%General]  
FileList="%s"
OutputDirPath=%s
FileNamePattern=<c:0>.<old>
PluginBatch\LayoutPlugin\General\drawResults=false
PluginBatch\LayoutPlugin\General\saveXml=true
PluginBatch\LayoutPlugin\General\\useTextRegions=%s
PluginBatch\LayoutPlugin\Layout Analysis Module\computeSeparators=true
PluginBatch\LayoutPlugin\Layout Analysis Module\localBlockOrientation=false
PluginBatch\LayoutPlugin\Layout Analysis Module\maxImageSide=3000
PluginBatch\LayoutPlugin\Layout Analysis Module\minSuperPixelsPerBlock=15
PluginBatch\LayoutPlugin\Layout Analysis Module\\removeWeakTextLines=true
PluginBatch\LayoutPlugin\Layout Analysis Module\scaleMode=1
PluginBatch\LayoutPlugin\Super Pixel Classification\classifierPath=
PluginBatch\LayoutPlugin\Super Pixel Labeler\\featureFilePath=
PluginBatch\LayoutPlugin\Super Pixel Labeler\labelConfigFilePath=
PluginBatch\LayoutPlugin\Super Pixel Labeler\maxNumFeaturesPerClass=10000
PluginBatch\LayoutPlugin\Super Pixel Labeler\maxNumFeaturesPerImage=1000000
PluginBatch\LayoutPlugin\Super Pixel Labeler\minNumFeaturesPerClass=10000
PluginBatch\pluginList=Layout Analysis | Layout Analysis
SaveInfo\Compression=-1
SaveInfo\DeleteOriginal=false
SaveInfo\InputDirIsOutputDir=true
SaveInfo\Mode=2
PluginBatch\LayoutPlugin\Super Pixel Labeler\\featureFilePath=
PluginBatch\LayoutPlugin\Layout Analysis Module\\removeWeakTextLines=true
    """

#PluginBatch\pluginList="Layout Analysis | Layout Analysis;Layout Analysis | Detect Lines"

    cCVLLASeparatorProfile="""
[%%General]  
FileList="%s"
OutputDirPath=%s
FileNamePattern=<c:0>.<old>
SaveInfo\Compression=-1
SaveInfo\Mode=2
SaveInfo\DeleteOriginal=false
SaveInfo\InputDirIsOutputDir=true
PluginBatch\pluginList=Layout Analysis | Detect Separator Lines
PluginBatch\LayoutPlugin\General\\useTextRegions=false
PluginBatch\LayoutPlugin\General\drawResults=false
PluginBatch\LayoutPlugin\General\saveXml=true
PluginBatch\LayoutPlugin\Super Pixel Labeler\\featureFilePath=
PluginBatch\LayoutPlugin\Super Pixel Labeler\labelConfigFilePath=
PluginBatch\LayoutPlugin\Super Pixel Labeler\maxNumFeaturesPerImage=1000000
PluginBatch\LayoutPlugin\Super Pixel Labeler\minNumFeaturesPerClass=10000
PluginBatch\LayoutPlugin\Super Pixel Labeler\maxNumFeaturesPerClass=10000
PluginBatch\LayoutPlugin\Super Pixel Classification\classifierPath=

"""
    cCVLProfileTabReg ="""
[%%General]  
FileList="%s"
OutputDirPath="%s"
FileNamePattern=<c:0>.<old>
SaveInfo\Compression=-1
SaveInfo\Mode=2
SaveInfo\DeleteOriginal=false
SaveInfo\InputDirIsOutputDir=true
PluginBatch\pluginList=Forms Analysis | Apply template (Match)
PluginBatch\FormAnalysis\FormFeatures\\formTemplate="%s"
PluginBatch\FormAnalysis\FormFeatures\distThreshold=200
PluginBatch\FormAnalysis\FormFeatures\colinearityThreshold=20
PluginBatch\FormAnalysis\FormFeatures\\variationThresholdLower=0.5
PluginBatch\FormAnalysis\FormFeatures\\variationThresholdUpper=0.55
PluginBatch\FormAnalysis\FormFeatures\saveChilds=false
     """

#     cCVLProfileTabReg ="""
# [%%General]  
# FileList="%s"
# OutputDirPath="%s"
# FileNamePattern=<c:0>.<old>
# SaveInfo\Compression=-1
# SaveInfo\Mode=2
# SaveInfo\DeleteOriginal=false
# SaveInfo\InputDirIsOutputDir=true
# PluginBatch\pluginList=Forms Analysis | Apply template (Match)
# PluginBatch\FormAnalysis\FormFeatures\\formTemplate="%s"
# PluginBatch\FormAnalysis\FormFeatures\distThreshold=200
# PluginBatch\FormAnalysis\FormFeatures\colinearityThreshold=20
# PluginBatch\FormAnalysis\FormFeatures\\variationThresholdLower=0.5
# PluginBatch\FormAnalysis\FormFeatures\\variationThresholdUpper=0.55
# PluginBatch\FormAnalysis\FormFeatures\saveChilds=false
#      """
     
        

    #--- INIT -------------------------------------------------------------------------------------------------------------    
    def __init__(self):
        """
        Always call first the Component constructor.
        """
        Component.Component.__init__(self, "tableProcessor", self.usage, self.version, self.description) 
        
        self.coldir = None
        self.docid= None
        self.bKeepRegion = False
        self.bKeepTL=False
        self.bTemplate = False
        self.bBaseLine = False
        self.bSeparator = False
        self.bRegularTextLine = False
        self.sTemplateFile = None
        
        self.xmlns='http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'

        
    def setParams(self, dParams):
        """
        Always call first the Component setParams
        Here, we set our internal attribute according to a possibly specified value (otherwise it stays at its default value)
        """
        Component.Component.setParams(self, dParams)
        if "coldir" in dParams.keys():
            self.coldir = dParams["coldir"].strip()
        if "docid" in dParams.keys():  
            self.docid = dParams["docid"].strip()
#         if dParams.has_key("bRegion"):         
#             self.bKeepRegion = dParams["bRegion"]
        if "bTL" in dParams.keys():     
            self.bKeepTL = dParams["bTL"]            
        if "bBaseline" in dParams.keys():
            self.bBaseLine = dParams["bBaseline"]
        if "bSeparator" in dParams.keys():
            self.bSeparator = dParams["bSeparator"]
        if "template" in dParams.keys():         
            self.bTemplate = dParams["template"]                        
        if "regTL" in dParams.keys():
            self.bRegularTextLine = dParams["regTL"]  

        if "templatefile" in dParams.keys():  
            self.sTemplateFile = dParams["templatefile"]
            self.bTemplate=True                  
    
    
    
    
    def reintegrateTextIntoCells(self,doc,lLTextLines=[]):
        """
        from XMLDSTABLE
        """
        def overlapX(zone1,zone2):
            [a1,a2] = zone1 #self.getX(),self.getX()+ self.getWidth()
            [b1,b2] = zone2 #zone.getX(),zone.getX()+ zone.getWidth()
            return min(a2, b2) >=   max(a1, b1) 
        
        def overlapY(zone1,zone2):
            [a1,a2] = zone1 #self.getY(),self.getY() + self.getHeight()
            [b1,b2] = zone2 #zone.getY(),zone.getY() + zone.getHeight()
            return min(a2, b2) >=  max(a1, b1)           
        
        def signedRatioOverlap(zone1,zone2):
            """
             overlap self and zone
             return surface of self in zone 
            """
            
            [x1,y1,x12,y12] = zone1 #self.getX(),self.getY(),self.getHeight(),self.getWidth()
            [x2,y2,x22,y22] = zone2 #zone.getX(),zone.getY(),zone.getHeight(),zone.getWidth()
            
            w1,h1 = x12-x1,y12-y1 
            w2,h2 = x22-x2,y22-y2 
            fOverlap = 0.0
            
#             print (x1,x12),(x2,x22)
#             print overlapX((x1,x12),(x2,x22)) 
#             print (y1,y12),(y2,y22)
#             print overlapY((y1,y12),(y2,y22))
#             if overlapX((x1,w1),(x2,w2)) and overlapY((y1,h1),(y2,h2)):
            if overlapX((x1,x12),(x2,x22)) and overlapY((y1,y12),(y2,y22)):
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
            
            return fOverlap
            
        def bestRegionsAssignment(plgtl,lRegions):
            """
                find the best (max overlap for self) region  for self
            """
    
            lOverlap=[]        
            for _,plg in lRegions:
                lOverlap.append(signedRatioOverlap(plgtl.getBoundingBox(),plg.getBoundingBox()))
#             print plgtl.getBoundingBox(), lOverlap
            if max(lOverlap) == 0: return None
            return lRegions[lOverlap.index(max(lOverlap))]
        
        
        lPages = PageXml.getChildByName(doc.getroot(),'Page')
        lRegionsToBeDeleted = []
        for i, page in enumerate(lPages):
            if lLTextLines == []:
                lTextLines = PageXml.getChildByName(page,'TextLine')
            else: lTextLines =lLTextLines[i]
                
            lCells = PageXml.getChildByName(page,'TableCell')
    
#             print len(lCells),len(lTextLines)
            lOCells=[]  
            for cell in lCells:
                #get Coords
                xpath  = "./a:%s" % ("Coords")
                lCoords = cell.xpath(xpath,namespaces={"a": self.xmlns})            
                coord= lCoords[0]
                sPoints=coord.get('points')
                lsPair = sPoints.split(' ')
                lXY = list()
                for sPair in lsPair:
                    (sx,sy) = sPair.split(',')
                    lXY.append( (int(sx), int(sy)) )
                plg = Polygon(lXY)
                lOCells.append((cell,plg))
    
            # find the best assignment of each text
            for tl in lTextLines:
                #get Coords
                xpath  = "./a:%s" % ("Coords")
                lCoords = tl.xpath(xpath,namespaces={"a": self.xmlns})            
                coord= lCoords[0]
                sPoints=coord.get('points')
                lsPair = sPoints.split(' ')
                lXY = list()
                for sPair in lsPair:
                    (sx,sy) = sPair.split(',')
                    lXY.append( (int(sx), int(sy)) )
                plg = Polygon(lXY)
                cell = bestRegionsAssignment(plg,lOCells)
                if cell:
                    c,_=cell
                    lRegionsToBeDeleted.append(c.parent)
                    ## what about parent  TextRegion  delete at least TextRegion/TextEquiv
#                     tl.unlinkNode()
                    tlcp = tl.docCopyNode(c.doc,True)
#                     tlcp.unlinkNode()
                    c.append(tlcp)
#                     print c
            
        for region in lRegionsToBeDeleted:
            region.getParent().remove(region)
#             region.unlinkNode()
#             region.freeNode()
            
            
            
    def reinitPage(self,doc):
        """
         empty page 
        """
        lNodes = PageXml.getChildByName(doc.getroot(),'Page')

        for node in lNodes:
            node.unlinkNode()
            
    def findTemplate(self,doc):
        """
            find the page where the first TableRegion occurs and extract it
            
        """
        from copy import deepcopy
        
        lT = PageXml.getChildByName(doc.getroot(),'TableRegion')
        if lT == []:
            return None
        firstTable=lT[0]
        # lazy guy!
        newDoc,fakepage = PageXml.createPageXmlDocument('NLE', '', 0,0)
        
        
        page=firstTable.getparent()
        fakepage.set("imageFilename",page.get('imageFilename'))
        fakepage.set("imageWidth",page.get('imageWidth'))
        fakepage.set("imageHeight",page.get('imageHeight'))
        page.getparent().remove(page)
        # add table
        
        xx =deepcopy(firstTable)
        fakepage.append(xx)
        return newDoc
        

            
    def createRegistrationProfile(self,sTemplatefile):
        # get all images files
        localpath =  os.path.abspath("./%s/col/%s"%(self.coldir,self.docid))
        l =      glob.glob(os.path.join(localpath, "*.jpg"))
        l.sort()
        listfile = ";".join(l)
        listfile  = listfile.replace(os.sep,"/")
        txt=  LAProcessor.cCVLProfileTabReg % (listfile,localpath.replace(os.sep,"/"),os.path.abspath("%s"%(sTemplatefile)).replace(os.sep,"/"))

        # wb mandatory for crlf in windows
        prnfilename = "%s%s%s_reg.prn"%(self.coldir,os.sep,self.docid)
        f=open(prnfilename,'w', encoding="utf-8")
        f.write(txt)
        return prnfilename
    

    
    def createLinesProfile(self):
        """
             OutputDirPath mandatory
        """
        # get all images files
        localpath =  os.path.abspath("./%s/col/%s"%(self.coldir,self.docid))
        l =      glob.glob(os.path.join(localpath, "*.jpg"))
        l.sort()
        listfile = ";".join(l)
        listfile  = listfile.replace(os.sep,"/")
        localpath = localpath.replace(os.sep,'/')
        txt =  LAProcessor.cCVLLASeparatorProfile % (listfile,localpath)
        # wb mandatory for crlf in windows
        prnfilename = "%s%s%s_gl.prn"%(self.coldir,os.sep,self.docid)
        f=open(prnfilename,'wb')
        f.write(txt)
        return prnfilename
    
    def createLAProfile(self):
        """
             OutputDirPath mandatory
        """
        # get all images files
        localpath =  os.path.abspath("./%s/col/%s"%(self.coldir,self.docid))
        l =      glob.glob(os.path.join(localpath, "*.jpg"))
        l.sort()
        listfile = ";".join(l)
        listfile  = listfile.replace(os.sep,"/")
        localpath = localpath.replace(os.sep,'/')
        txt =  LAProcessor.cCVLLAProfile % (listfile,localpath,self.bKeepRegion)
#         print txt
        # wb mandatory for crlf in windows
        prnfilename = "%s%s%s_la.prn"%(self.coldir,os.sep,self.docid)
        f=open(prnfilename,'wb')
        f.write(txt)
        return prnfilename
    
    
    def storeMPXML(self,lFiles):
        """
            store files in lFiles as mpxml
        """
        docDir = os.path.join(self.coldir+os.sep+'col',self.docid)
        
        doc = MultiPageXml.makeMultiPageXml(lFiles)

        sMPXML  = docDir+".mpxml"
#         print sMPXML
        doc.write(sMPXML,encoding="UTF-8",pretty_print=True,xml_declaration=True)       
        
#         trace("\t\t- validating the MultiPageXml ...")
#         if not MultiPageXml.validate(doc): 
#                 traceln("   *** WARNING: XML file is invalid against the schema: '%s'"%self.outputFileName)
#         traceln(" Ok!")        
    
        return doc, sMPXML
    
    
    def extractFileNamesFromMPXML(self,doc):
        """
            to insure correct file order !
        """
        xmlpath=os.path.abspath("%s%s%s%s%s" % (self.coldir,os.sep,'col',os.sep,self.docid))

        lNd = PageXml.getChildByName(doc.getroot(), 'Page')
#         for i in lNd:print i
        return list(map(lambda x:"%s%s%s.xml"%(xmlpath,os.sep,x.get('imageFilename')[:-4]), lNd))
    
    
    
    def performLA(self,doc):
        """
            # for document doc 
            ## find the page where the template is
            ## store it as template (check borders))
            ## generate profile for table registration
            ## (execution)
            ## create profile for lA
            ## (execution)    
        """
        
#         lNumPages = []
        if self.bTemplate or self.bBaseLine or self.bSeparator:
            # extract list of files sorted as in MPXML
            lFullPathXMLNames = self.extractFileNamesFromMPXML(doc)
            nbPages = len(lFullPathXMLNames) 
            ## 1 generate xml files if only pxml are there
            
            xmlpath=os.path.abspath(os.path.join (self.coldir,'col',self.docid))
            
            lXMLNames = [ "%s%s%s"%(xmlpath,os.sep,name) for name in os.listdir(xmlpath) if os.path.basename(name)[-4:] =='.xml']
            isXml = [] != lXMLNames        
            if isXml:
                [ os.remove("%s%s%s"%(xmlpath,os.sep,name)) for name in os.listdir(xmlpath) if os.path.basename(name)[-4:] =='.xml']    
                isXml = False        
            isPXml = [] != [ name for name in os.listdir(xmlpath) if os.path.basename(name)[-5:] =='.pxml']              
            assert not isXml and isPXml
    
            # recreate doc?  (mpxml)
            
            lPXMLNames = [ name for name in os.listdir(xmlpath) if os.path.basename(name)[-5:] =='.pxml']
            if not isXml:
                # copy pxml in xml
                for name in lPXMLNames: 
                    oldname = "%s%s%s" %(xmlpath,os.sep,name)
                    newname = "%s%s%s" % (xmlpath,os.sep,name)
                    newname = newname[:-5]+'.xml' 
                    tmpdoc =  etree.parse(oldname)
                    tmpdoc.write(newname,encoding="UTF-8", pretty_print=True,xml_declaration=True)

        if self.bKeepTL:
            # keep ltextLione
            lTextLines=[]
            lPages = PageXml.getChildByName(doc.getroot(),'Page')
            for page in lPages:
                lTextLines.append(PageXml.getChildByName(page,'TextLine'))
        ## Table registration 
        if self.bTemplate:
            if self.sTemplateFile is None:
                templatePage = self.findTemplate(doc)
                if templatePage is None:
                    traceln("No table found in this document: %s" % self.docid)
                else:
                    oldOut=  self.outputFileName
                    self.outputFileName = "%s%s%s.templ.xml" % (self.coldir,os.sep,self.docid)
                    stemplatefile = "%s%s%s.templ.xml" % (self.coldir,os.sep,self.docid)
                    print (stemplatefile)
                    self.writeDom(templatePage, True)
                    self.outputFileName = oldOut
                    prnregfilename= self.createRegistrationProfile(stemplatefile)
            
            else:
#                 raise Exception, 'file template stuff: to be done'
                prnregfilename= self.createRegistrationProfile(self.sTemplateFile)
            
            job = LAProcessor.cNomacs+ " --batch %s"%(prnregfilename)
            os.system(job)
            traceln('table registration done: %s'% prnregfilename)            
        
        
        ## separator detection
        if self.bSeparator:
            prnglfilename = self.createLinesProfile()
            job = LAProcessor.cNomacs+ " --batch %s"%(prnglfilename)
            os.system(job)
            traceln( 'GL done: %s' % prnglfilename)    
        
        ## baseline detection
        if self.bBaseLine:
            prnlafilename = self.createLAProfile()
#             job = LAProcessor.cNomacs+ " --batch %s"%(prnlafilename)
            job = LAProcessor.cNomacsold+ " --batch %s"%(prnlafilename)

            os.system(job)
            traceln('LA done: %s' % prnlafilename)        
        
        if self.bTemplate or self.bBaseLine or self.bSeparator:
            doc, sMPXML= self.storeMPXML(lFullPathXMLNames)     
        
        # Does not work with URO LA! 
        if self.bKeepTL:
            self.reintegrateTextIntoCells(doc,lTextLines)
        
        ## text rectangles as textline region 
        if self.bRegularTextLine:
            self.regularTextLines(doc)
        doc.write(sMPXML,encoding="UTF-8",pretty_print=True,xml_declaration=True)       

        return doc, nbPages
    
    def regularTextLinesold(self,doc):
        """
            from a baseline: create a regular TextLine:
            
            also: for slanted baseline: 
                
        """
        from shapely.geometry import LineString
        from shapely.affinity import  translate        
        self.xmlns='http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'

        lTextLines = PageXml.getChildByName(doc.getroot(),'TextLine')
        for tl in lTextLines:
            #get Coords
            xpath  = "./a:%s" % ("Coords")
            lCoords = tl.xpath(xpath,namespaces={"a": self.xmlns})            
            coord= lCoords[0]
            xpath  = "./a:%s" % ("Baseline")
            lBL = tl.xpath(xpath,namespaces={"a": self.xmlns})      
            baseline = lBL[0]
            
            sPoints=baseline.get('points')
            lsPair = sPoints.split(' ')
            lXY = list()
            for sPair in lsPair:
                try:
                    (sx,sy) = sPair.split(',')
                    lXY.append( (int(sx), int(sy)) )
                except ValueError:print (tl)
            plg = Polygon(lXY)
            line=LineString(lXY)
            # 50 seems to large: the manual GT is 30  ? not always!
            iHeight = 30   # in pixel
            x1,y1, x2,y2 = plg.getBoundingBox()
            if coord is not None: 
                coord.set('points',"%d,%d %d,%d %d,%d %d,%d" % (x1,y1-iHeight,x2,y1-iHeight,x2,y2,x1,y2))
            else:
                print (tl)                     

    def regularTextLines(self,doc):
        """
            from a baseline: create a regular TextLine:
        """
        from shapely.geometry import LineString
        from shapely.affinity import  translate   
        self.xmlns='http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'

        lTextLines = PageXml.getChildByName(doc.getroot(),'TextLine')
        for tl in lTextLines:
            #get Coords
            xpath  = "./a:%s" % ("Coords")
            lCoords = tl.xpath(xpath,namespaces={"a": self.xmlns})            
            coord= lCoords[0]
            xpath  = "./a:%s" % ("Baseline")
            lBL = tl.xpath(xpath,namespaces={"a": self.xmlns})      
            try:baseline = lBL[0]
            except IndexError:continue
 
            sPoints=baseline.get('points')
            lsPair = sPoints.split(' ')
            lXY = list()
            for sPair in lsPair:
                try:
                    (sx,sy) = sPair.split(',')
                    lXY.append( (int(sx), int(sy)) )
                except ValueError:print (tl)
            #plg = Polygon(lXY)
            try: line=LineString(lXY)
            except ValueError: continue  # LineStrings must have at least 2 coordinate tuples
            topline=translate(line,yoff=-20)
            #iHeight = 20   # in pixel
            #x1,y1, x2,y2 = topline.getBoundingBox()
            if coord is not None: 
                spoints = ' '.join("%s,%s"%(int(x[0]),int(x[1])) for x in line.coords)
                lp=list(topline.coords)
                lp.reverse()
                spoints =spoints+ ' ' +' '.join("%s,%s"%(int(x[0]),int(x[1])) for x in lp) 
                #spoints = ' '.join("%s,%s"%(x[0],x[1]) for x in pp.coords)
                #coord.set('points',"%d,%d %d,%d %d,%d %d,%d" % (x1,y1-iHeight,x2,y1-iHeight,x2,y2,x1,y2))
                coord.set('points',spoints)
            else:
                print (tl)                     
#             print tl    
    
    def run(self,doc):
        """
            GT from TextRegion
            or GT from Table
            
            input mpxml (GT)
            delete TextLine
            
        """
        if not (self.bTemplate or self.bBaseLine or self.bSeparator) and self.bRegularTextLine:
            self.regularTextLines(doc)
            self.writeDom(doc, True)  
        else:
            doc,nbpages =  self.performLA(doc)
            return doc
    

if __name__ == "__main__":

    # for each document 
    ## find the page where the template is
    ## store it as template (check borders))
    ## generate profile for table registration
    ## (execution)
    ## create profile for lA
    ## (execution)
    
    tp = LAProcessor()
    #prepare for the parsing of the command line
    tp.createCommandLineParser()
    tp.add_option("--coldir", dest="coldir", action="store", type="string", help="collection folder")
    tp.add_option("--docid", dest="docid", action="store", type="string", help="document id")
    tp.add_option("--bl", dest="bBaseline", action="store_true", default=False, help="detect baselines")
#     tp.add_option("--region", dest="bRegion", action="store_true", default=False, help="keep Region")
    tp.add_option("--tl", dest="bTL", action="store_true", default=False, help="keep textlines")
    tp.add_option("--sep", dest="bSeparator", action="store_true", default=False, help="detect separator (graphical lines)")
    tp.add_option("--regTL", dest="regTL", action="store_true", default=False, help="generate regular TextLines")
    tp.add_option("--form", dest="template", action="store_true", default=False, help="perform template registration")
    tp.add_option("--formfile", dest="templatefile", action="store", type='string', default=None,help="use this template file (pagexml) for registration")

    #tp.add_option("--form", dest="template", action="store", type="string", help="perform template registration")
        
    #parse the command line
    dParams, args = tp.parseCommandLine()
    
    #Now we are back to the normal programmatic mode, we set the componenet parameters
    tp.setParams(dParams)
    
    doc = tp.loadDom()
    tp.run(doc)
    


