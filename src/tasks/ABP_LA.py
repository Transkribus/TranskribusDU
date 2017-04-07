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

import sys, os.path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))


import glob
import common.Component as Component

from xml_formats.PageXml import PageXml
from xml_formats import DS2PageXml, Page2DS




class TableProcessor(Component.Component):
    """
        
    """
    usage = "" 
    version = "v.01"
    description = "description: table template processor"

    
    cCVLLAProfile="""
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
    cCVLProfile ="""
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
    
    cNomacs = '"C:\\Program Files\\READFramework\\nomacs-x64\\nomacs.exe"'
    
    #--- INIT -------------------------------------------------------------------------------------------------------------    
    def __init__(self):
        """
        Always call first the Component constructor.
        """
        Component.Component.__init__(self, "tableProcessor", self.usage, self.version, self.description) 
        
        self.colname = None
        self.docid= None
        
    def setParams(self, dParams):
        """
        Always call first the Component setParams
        Here, we set our internal attribute according to a possibly specified value (otherwise it stays at its default value)
        """
        Component.Component.setParams(self, dParams)
        if dParams.has_key("coldir"): 
            self.colname = dParams["coldir"]
        if dParams.has_key("docid"):         
            self.docid = dParams["docid"]

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
        return newDoc
        
    def createRegistrationProfile(self):
        # get all images files
        localpath =  os.path.abspath("./%s/col/%s"%(self.colname,self.docid))
        l =      glob.glob(os.path.join(localpath, "*.jpg"))
        listfile = ";".join(l)
        listfile  = listfile.replace(os.sep,"/")
        txt=  TableProcessor.cCVLProfile % (listfile,"%s/col/%s"%(self.colname,self.docid),os.path.abspath("%s/%s.templ.xml"%(self.colname,self.docid)).replace(os.sep,"/"))
        # wb mandatory for crlf in windows
        prnfilename = "%s%s%s_reg.prn"%(self.colname,os.sep,self.docid)
        f=open(prnfilename,'wb')
        f.write(txt)
        return prnfilename
    
    def createLAProfile(self):
        """
             OutputDirPath mandatory
        """
        # get all images files
        localpath =  os.path.abspath("./%s/col/%s"%(self.colname,self.docid))
        l =      glob.glob(os.path.join(localpath, "*.jpg"))
        listfile = ";".join(l)
        listfile  = listfile.replace(os.sep,"/")
        localpath = localpath.replace(os.sep,'/')
        txt =  TableProcessor.cCVLLAProfile % (listfile,localpath)
        # wb mandatory for crlf in windows
        prnfilename = "%s%s%s_la.prn"%(self.colname,os.sep,self.docid)
        f=open(prnfilename,'wb')
        f.write(txt)
        return prnfilename
    
    def run(self,doc):
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

        if templatePage is None:
            print "No table found in this document:%d"%self.docid
            return
        self.outputFileName = "%s%s%s.templ.xml"%(self.colname,os.sep,self.docid)
        print self.outputFileName 
        self.writeDom(templatePage, True)
        prnregfilename= self.createRegistrationProfile()
        prnlafilename = self.createLAProfile()
#         import subprocess
        job = TableProcessor.cNomacs+ " --batch %s"%(prnregfilename)
#         subprocess.call([job])
        os.system(job)
        print 'job done', prnregfilename
        job = TableProcessor.cNomacs+ " --batch %s"%(prnlafilename)
        os.system(job)
        print 'job done', prnlafilename
        
        
        ## convert *.xml into DS
        ## convert ds int ods.mpxml
        ## upload  (option)
        

    

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
        
    #parse the command line
    dParams, args = tp.parseCommandLine()
    
    #Now we are back to the normal programmatic mode, we set the componenet parameters
    tp.setParams(dParams)
    
    doc = tp.loadDom()
    tp.run(doc)
    