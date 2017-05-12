# -*- coding: utf-8 -*-
"""


    ABP_LA.py

    create profile for  template registration and la analysis (CVL)
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
from common.trace import traceln, trace

from xml_formats.PageXml import PageXml
from xml_formats.Page2DS import primaAnalysis
from xml_formats.PageXml import MultiPageXml

import libxml2

class TableProcessor(Component.Component):
    """
        
    """
    usage = "" 
    version = "v.01"
    description = "description: table template processor"

    
    cCVLLAProfileold="""
[%%General]  
FileList="%s"
OutputDirPath=%s
FileNamePattern=<c:0>.<old>
SaveInfo\Compression=-1
SaveInfo\Mode=2
SaveInfo\DeleteOriginal=false
SaveInfo\InputDirIsOutputDir=true
PluginBatch\pluginList=Layout Analysis | Layout Analysis
PluginBatch\LayoutPlugin\General\useTextRegions=false
PluginBatch\LayoutPlugin\General\drawResults=false
PluginBatch\LayoutPlugin\General\saveXml=true
PluginBatch\LayoutPlugin\Super Pixel Labeler\\featureFilePath=
PluginBatch\LayoutPlugin\Super Pixel Labeler\labelConfigFilePath=
PluginBatch\LayoutPlugin\Super Pixel Labeler\maxNumFeaturesPerImage=1000000
PluginBatch\LayoutPlugin\Super Pixel Labeler\minNumFeaturesPerClass=10000
PluginBatch\LayoutPlugin\Super Pixel Labeler\maxNumFeaturesPerClass=10000
PluginBatch\LayoutPlugin\Super Pixel Classification\classifierPath=

"""

    cCVLLAProfile = """
[%%General]  
FileList="%s"
OutputDirPath=%s
FileNamePattern=<c:0>.<old>
PluginBatch\LayoutPlugin\General\drawResults=false
PluginBatch\LayoutPlugin\General\saveXml=true
PluginBatch\LayoutPlugin\General\useTextRegions=true
PluginBatch\LayoutPlugin\Layout Analysis Module\computeSeparators=true
PluginBatch\LayoutPlugin\Layout Analysis Module\localBlockOrientation=false
PluginBatch\LayoutPlugin\Layout Analysis Module\maxImageSide=3000
PluginBatch\LayoutPlugin\Layout Analysis Module\minSuperPixelsPerBlock=15
PluginBatch\LayoutPlugin\Layout Analysis Module\removeWeakTextLines=true
PluginBatch\LayoutPlugin\Layout Analysis Module\scaleMode=1
PluginBatch\LayoutPlugin\Super Pixel Classification\classifierPath=
PluginBatch\LayoutPlugin\Super Pixel Labeler\featureFilePath=
PluginBatch\LayoutPlugin\Super Pixel Labeler\labelConfigFilePath=
PluginBatch\LayoutPlugin\Super Pixel Labeler\maxNumFeaturesPerClass=10000
PluginBatch\LayoutPlugin\Super Pixel Labeler\maxNumFeaturesPerImage=1000000
PluginBatch\LayoutPlugin\Super Pixel Labeler\minNumFeaturesPerClass=10000
PluginBatch\pluginList=Layout Analysis | Layout Analysis
SaveInfo\Compression=-1
SaveInfo\DeleteOriginal=false
SaveInfo\InputDirIsOutputDir=true
SaveInfo\Mode=2
PluginBatch\LayoutPlugin\Super Pixel Labeler\featureFilePath=
PluginBatch\LayoutPlugin\Layout Analysis Module\removeWeakTextLines=true
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
PluginBatch\LayoutPlugin\General\useTextRegions=false
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
PluginBatch\pluginList=Forms Analysis | Apply Template (Single)
PluginBatch\FormAnalysis\FormFeatures\threshLineLenRatio=0.6
PluginBatch\FormAnalysis\FormFeatures\distThreshold=30
PluginBatch\FormAnalysis\FormFeatures\errorThr=15
PluginBatch\FormAnalysis\FormFeatures\\formTemplate="%s"
PluginBatch\FormAnalysis\FormFeatures\saveChilds=false
     """
     
    if sys.platform == 'win32':
        cNomacs = '"C:\\Program Files\\READFramework\\nomacs-x64\\nomacs.exe"'
    else:
        cNomacs = "/opt/Tools/src/tuwien-2017/nomacs/nomacs"
        
    
    #--- INIT -------------------------------------------------------------------------------------------------------------    
    def __init__(self):
        """
        Always call first the Component constructor.
        """
        Component.Component.__init__(self, "tableProcessor", self.usage, self.version, self.description) 
        
        self.coldir = None
        self.docid= None
        self.bTextRegion = False
        
    def setParams(self, dParams):
        """
        Always call first the Component setParams
        Here, we set our internal attribute according to a possibly specified value (otherwise it stays at its default value)
        """
        Component.Component.setParams(self, dParams)
        if dParams.has_key("coldir"): 
            self.coldir = dParams["coldir"]
        if dParams.has_key("docid"):         
            self.docid = dParams["docid"]
        if dParams.has_key("bTextRegion"):         
            self.bTextRegion = dParams["bTextRegion"]




                
                
    def unLinkTextLines(self,doc):
        """
            delete textlines and baselines
        """
        lT = PageXml.getChildByName(doc.getRootElement(),'TextLine')
        if lT == []:
            return doc
        
        for text in lT:
            text.unlinkNode()
            text.freeNode()
        return doc
    
    def unLinkTable(self,doc):
        """
            delete table
        """
        lT = PageXml.getChildByName(doc.getRootElement(),'TableRegion')
        if lT == []:
            return doc
        
        for table in lT:
            table.unlinkNode()
            table.freeNode()
        return doc
      
    def findTemplate(self,doc):
        """
            find the page where the first TableRegion occurs and extract it
        """
        lT = PageXml.getChildByName(doc.getRootElement(),'TableRegion')
        if lT == []:
            return None
        firstTable=lT[0]
        # lazy guy!
        page = firstTable.parent
        newDoc,_ = PageXml.createPageXmlDocument('XRCE', '', 0,0)
        page.unlinkNode()
        newDoc.setRootElement(page)
        ### need to add the ns!!
#         print newDoc.serialize('utf-8',True)
        # Borders must be visible: 
        #leftBorderVisible="false" rightBorderVisible="false" topBorderVisible="false
        lcells = PageXml.getChildByName(newDoc.getRootElement(),'TableCell')
        for cell in lcells:
            cell.setProp("leftBorderVisible",'true')
            cell.setProp("rightBorderVisible",'true')
            cell.setProp("topBorderVisible",'true')
            cell.setProp("bottomBorderVisible",'true')

        return newDoc
        
    def createRegistrationProfile(self):
        # get all images files
        localpath =  os.path.abspath("./%s/col/%s"%(self.coldir,self.docid))
        l =      glob.glob(os.path.join(localpath, "*.jpg"))
        listfile = ";".join(l)
        listfile  = listfile.replace(os.sep,"/")
        txt=  TableProcessor.cCVLProfileTabReg % (listfile,"%s/col/%s"%(self.coldir,self.docid),os.path.abspath("%s/%s.templ.xml"%(self.coldir,self.docid)).replace(os.sep,"/"))
        # wb mandatory for crlf in windows
        prnfilename = "%s%s%s_reg.prn"%(self.coldir,os.sep,self.docid)
        f=open(prnfilename,'wb')
        f.write(txt)
        return prnfilename
    
    


    
    def createLinesProfile(self):
        """
             OutputDirPath mandatory
        """
        # get all images files
        localpath =  os.path.abspath("./%s/col/%s"%(self.coldir,self.docid))
        l =      glob.glob(os.path.join(localpath, "*.jpg"))
        listfile = ";".join(l)
        listfile  = listfile.replace(os.sep,"/")
        localpath = localpath.replace(os.sep,'/')
        txt =  TableProcessor.cCVLLASeparatorProfile % (listfile,localpath)

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
        listfile = ";".join(l)
        listfile  = listfile.replace(os.sep,"/")
        localpath = localpath.replace(os.sep,'/')
        txt =  TableProcessor.cCVLLAProfile % (listfile,localpath)

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
        print sMPXML
        doc.saveFormatFileEnc(sMPXML,"UTF-8",True)       
        
#         trace("\t\t- validating the MultiPageXml ...")
#         if not MultiPageXml.validate(doc): 
#                 traceln("   *** WARNING: XML file is invalid against the schema: '%s'"%self.outputFileName)
#         traceln(" Ok!")        
    
    
    def creaGTWithTableColumn(self,doc):
        """
            # for document doc 
            ## find the page where the template is
            ## store it as template (check borders))
            ## generate profile for table registration
            ## (execution)
            ## create profile for lA
            ## (execution)    
        """
        templatePage = self.findTemplate(doc)


        ## RM  previous *.xml
        xmlpath="%s%s%s%s%s" % (self.coldir,os.sep,'col',os.sep,self.docid)
        [ os.remove("%s%s%s"%(xmlpath,os.sep,name)) for name in os.listdir(xmlpath) if os.path.basename(name)[-4:] =='.xml']
              
        if templatePage is None:
            traceln("No table found in this document: %s" % self.docid)
            
        else:
            oldOut=  self.outputFileName
            self.outputFileName = "%s%s%s.templ.xml" % (self.coldir,os.sep,self.docid)
            print self.outputFileName 
            self.writeDom(templatePage, True)
            self.outputFileName = oldOut
            prnregfilename= self.createRegistrationProfile()
        

            job = TableProcessor.cNomacs+ " --batch %s"%(prnregfilename)
            os.system(job)
            traceln('table registration done: %s'% prnregfilename)            
        
        prnglfilename = self.createLinesProfile()
        prnlafilename = self.createLAProfile()

        job = TableProcessor.cNomacs+ " --batch %s"%(prnglfilename)
        os.system(job)
        traceln( 'GL done: %s' % prnglfilename)    
                
        job = TableProcessor.cNomacs+ " --batch %s"%(prnlafilename)
        os.system(job)
        traceln('LA done: %s' % prnlafilename)        
        
        lFullPathXMLNames = [ "%s%s%s" % (xmlpath,os.sep,name) for name in os.listdir(xmlpath) if os.path.basename(name)[-4:] =='.xml']

        self.storeMPXML(lFullPathXMLNames)        
    
    def createGTWithtextRegion(self):
        """
            85 pages 2559/7048
            
            - if xml : destroy them?: oui
            - convert pxml into xml
            - apply LA only
            -create mpxml 
        """
        # any xml ?
        xmlpath="%s%s%s%s%s" % (self.coldir,os.sep,'col',os.sep,self.docid)
        
        lXMLNames = [ "%s%s%s"%(xmlpath,os.sep,name) for name in os.listdir(xmlpath) if os.path.basename(name)[-4:] =='.xml']
        isXml = [] != lXMLNames        
        if isXml:
            [ os.remove("%s%s%s"%(xmlpath,os.sep,name)) for name in os.listdir(xmlpath) if os.path.basename(name)[-4:] =='.xml']    
            isXml = False
            
        isPXml = [] != [ name for name in os.listdir(xmlpath) if os.path.basename(name)[-5:] =='.pxml']              
        
        assert not isXml and isPXml
        
        
        lPXMLNames = [ name for name in os.listdir(xmlpath) if os.path.basename(name)[-5:] =='.pxml']
        if not isXml:
            # copy pxml in xml
            for name in lPXMLNames: 
                oldname = "%s%s%s" %(xmlpath,os.sep,name)
                newname = "%s%s%s" % (xmlpath,os.sep,name)
                newname = newname[:-5]+'.xml' 
                
                doc =  libxml2.parseFile(oldname)
                self.unLinkTable(doc)
                doc.saveFileEnc(newname,"UTF-8")                     
                
                #shutil.copyfile(oldname, newname)

        prnglfilename = self.createLinesProfile()

        prnlafilename = self.createLAProfile()

        job = TableProcessor.cNomacs+ " --batch %s"%(prnglfilename)
        os.system(job)
        print 'GL done', prnlafilename   
        
        job = TableProcessor.cNomacs+ " --batch %s"%(prnlafilename)
        os.system(job)
        print 'LA done', prnlafilename 

        lFullPathXMLNames = [ "%s%s%s" % (xmlpath,os.sep,name) for name in os.listdir(xmlpath) if os.path.basename(name)[-4:] =='.xml']
        
        
        

        self.storeMPXML(lFullPathXMLNames)
        
        ## convert xml into mpxml
        
        
        
    def run(self,doc):
        """
            GT from TextRegion
            or GT from Table
            
            input mpxml (GT)
            delete TextLine
            
        """
        self.unLinkTextLines(doc)

        if self.bTextRegion:
            self.createGTWithtextRegion()
        else:
            self.creaGTWithTableColumn(doc)
        
        
        return 
        
        ## convert *.xml into DS
        dsconv =primaAnalysis()
        dsconv.dpi=300
        dsconv.sPttrn = "%scol%s%s%s%s" % (self.coldir,os.sep,self.docid,os.sep,"*.xml")
        print dsconv.sPttrn
        dsconv.sDocID = self.docid
        doc = dsconv.run()
        dsconv.outputFileName = self.outputFileName
        print dsconv.outputFileName
        dsconv.writeDom(doc, True)

        ## convert ds into .mpxml
        ## upload  (option)
        
        # row detection

    

if __name__ == "__main__":

    # for each document 
    ## find the page where the template is
    ## store it as template (check borders))
    ## generate profile for table registration
    ## (execution)
    ## create profile for lA
    ## (execution)
    
    tp = TableProcessor()
    #prepare for the parsing of the command line
    tp.createCommandLineParser()
    tp.add_option("--coldir", dest="coldir", action="store", type="string", help="collection folder")
    tp.add_option("--docid", dest="docid", action="store", type="string", help="document id")
    tp.add_option("--useTR", dest="bTextRegion", action="store_true", default=False, help="use TextRegion forGT")
        
    #parse the command line
    dParams, args = tp.parseCommandLine()
    
    #Now we are back to the normal programmatic mode, we set the componenet parameters
    tp.setParams(dParams)
    
    doc = tp.loadDom()
    tp.run(doc)
    