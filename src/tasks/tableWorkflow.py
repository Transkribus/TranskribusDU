# -*- coding: utf-8 -*-
"""


    IETableWorkflow.py

    Process a full collection with table analysis and IE extraction
    Based on template
        
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
import logging 
import json
import glob
from optparse import OptionParser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))

try:
    import TranskribusPyClient_version
except ImportError:
    sys.path.append( os.path.join( os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) ))))
                                    , "TranskribusPyClient", "src" ))
    sys.path.append( os.path.join( os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) ))))
                                    , "TranskribusPyClient", "src" ))    
    import TranskribusPyClient_version

from TranskribusPyClient.client import TranskribusClient
from TranskribusCommands.TranskribusDU_transcriptUploader import TranskribusTranscriptUploader
from TranskribusCommands.Transkribus_downloader import TranskribusDownloader
from TranskribusCommands import  sCOL,  __Trnskrbs_basic_options

import common.Component as Component
from common.trace import traceln, trace

from xml_formats.PageXml import PageXml, MultiPageXml
from tasks.performCVLLA import LAProcessor
from tasks.DU_ABPTable_T import DU_ABPTable_TypedCRF
from xml_formats.Page2DS import primaAnalysis
from xml_formats.DS2PageXml import DS2PageXMLConvertor
from tasks.rowDetection import RowDetection
from ObjectModel.xmlDSDocumentClass import XMLDSDocument

class TableProcessing(Component.Component):
    usage = "" 
    version = "v.01"
    description = "description: table layout analysis based on template"


    sCOL = "col"
    sMPXMLExtension = ".mpxml"
    
    def __init__(self):
        """
        Always call first the Component constructor.
        """
        Component.Component.__init__(self, "TableProcessing", self.usage, self.version, self.description) 
        
        self.colid = None
        self.docid= None

        self.bFullCol = False
        # generate MPXML using Ext        
        self.useExtForMPXML = None
        
        self.sRowModelName = None
        self.sRowModelDir = None
        
        
    def setParams(self, dParams):
        """
        Always call first the Component setParams
        Here, we set our internal attribute according to a possibly specified value (otherwise it stays at its default value)
        """
        Component.Component.setParams(self, dParams)
        if dParams.has_key("coldir"): 
            self.coldir = dParams["coldir"]
        if dParams.has_key("colid"): 
            self.colid = dParams["colid"]
        if dParams.has_key("docid"):         
            self.docid = dParams["docid"]
        if dParams.has_key("useExt"):         
            self.useExtForMPXML  = dParams["useExt"]            
        
        if dParams.has_key('regMPXML'):
            self.bRegenerateMPXML=True
            
        if dParams.has_key("rowmodelname"):         
            self.sRowModelName = dParams["rowmodelname"]
        if dParams.has_key("rowmodeldir"):         
            self.sRowModelDir = dParams["rowmodeldir"]

        if dParams.has_key("htrmodel"): 
            self.sHTRmodel = dParams["htrmodel"]
        if dParams.has_key("dictname"): 
            self.sDictName = dParams["dictname"]
            
            
        # Connection to Transkribus
        self.myTrKCient = None
        self.persist = False
        self.loginInfo = False
        if dParams.has_key("server"):         
            self.server = dParams["server"]
        if dParams.has_key("persist"):         
            self.persist = dParams["persist"]        
        if dParams.has_key("login"):         
            self.loginInfo = dParams["login"]   

    def login(self,trnskrbs_client,  trace=None, traceln=None):
        """
        deal with the complicated login variants...
            -trace and traceln are optional print methods 
        return True or raises an exception
        """  
        DEBUG=True
        bOk = False
        if self.persist:
            #try getting some persistent session token
            if DEBUG and trace: trace("  ---login--- Try reusing persistent session ... ")
            try:
                bOk = trnskrbs_client.reusePersistentSession()
                if DEBUG and traceln: traceln("OK!")
            except:
                if DEBUG and traceln: traceln("Failed")
              
        if not bOk:
            if self.loginInfo:
                login, pwd = self.loginInfo, self.pwd
            else:
                if trace: DEBUG and trace("  ---login--- no login provided, looking for stored credentials... ")
                login, pwd = trnskrbs_client.getStoredCredentials(bAsk=False)
                if DEBUG and traceln: traceln("OK")    
    
            if DEBUG and traceln: trace("  ---login--- logging onto Transkribus as %s "%login)
            trnskrbs_client.auth_login(login, pwd)
            if DEBUG and traceln: traceln("OK")
            bOk = True
    
        return bOk

    def downloadCollection(self,colid,destDir,docid,bForce=False):
        """
            download colID
            
            replace destDir by '.'  ?
        """
        
#         options.server, proxies, loggingLevel=logging.WARN)
        #download
        downloader = TranskribusDownloader(self.myTrKCient.getServerUrl(),self.myTrKCient.getProxies())
        downloader.setSessionId(self.myTrKCient.getSessionId())
        traceln("- Downloading collection %s to folder %s"%(colid, os.path.abspath(destDir)))
#         col_ts, colDir = downloader.downloadCollection(colid, destDir, bForce=options.bForce, bNoImage=options.bNoImage)
        col_ts, colDir,ldocids,dFileListPerDoc = downloader.downloadCollection(colid, destDir, bForce=True, bNoImage=True,sDocId=docid)
        traceln("- Done")
    
        with open(os.path.join(colDir, "config.txt"), "w") as fd: fd.write("server=%s\nforce=%s\nstrict=%s\n"%(self.server, True, False))
    
        downloader.generateCollectionMultiPageXml(os.path.join(colDir, TableProcessing.sCOL),dFileListPerDoc,False)
    
        traceln('- Done, see in %s'%colDir)        
        
        return ldocids
    
    def upLoadDocument(self,colid,coldir,docid,sNote="",sTranscripExt='.mpxml'):
        """
            download colID
        """
        
#         options.server, proxies, loggingLevel=logging.WARN)
        #download
#         uploader = TranskribusTranscriptUploader(self.server,self.proxies)
        uploader = TranskribusTranscriptUploader(self.myTrKCient.getServerUrl(),self.myTrKCient.getProxies())
        uploader.setSessionId(self.myTrKCient.getSessionId())
        traceln("- uploading document %s to collection %s" % (docid,colid))
        uploader.uploadDocumentTranscript(colid, docid, os.path.join(coldir,sCOL), sNote, 'NLE Table', sTranscripExt, iVerbose=False)
        traceln("- Done")
        return 

    def applyHTR(self,colid,docid,nbPages,modelname,dictionary):
        """
            apply HTR on docid
            
            htr id is needed: we have htrmodename
        """
        
        sPages= "1-%d"%(nbPages)
        sModelID = None
        # get modelID
        lColModels =  self.myTrKCient.listRnns(colid)
        for model in lColModels:
            if model['name'] == modelname:
                sModelID  = model['htrId']
                traceln('model id = %s'%sModelID)
                #some old? models do not have params field
#             try: traceln("%s\t%s\t%s" % (model['htrId'],model['name'],model['params']))
#             except KeyError: traceln("%s\t%s\tno params" % (model['htrId'],model['name']))
        raise sModelID != None, "no model ID found for %s" %(modelname)
        jobid = self.myTrKCient.htrRnnDecode(colid, sModelID, dictionary, docid, sPages)
        traceln(jobid)        
        return jobid
               
    
    def extractFileNamesFromMPXML(self,mpxmldoc):
        """
            to insure correct file order !
            
            duplicated form performCVLLA.py
        """
        xmlpath=os.path.abspath(os.path.join(self.coldir,sCOL,self.docid))

        lNd = PageXml.getChildByName(mpxmldoc.getRootElement(), 'Page')
#         for i in lNd:print i
        return map(lambda x:"%s%s%s.xml"%(xmlpath,os.sep,x.prop('imageFilename')[:-4]), lNd)
        
        
    def processDocument(self,coldir,colid,docid,dom=None):
        """
            process a single document
            
            1 python ../../src/xml_formats/PageXml.py trnskrbs_5400/col/17442 --ext=pxml
            2 python ../../src/tasks/performCVLLA.py  --coldir=trnskrbs_5400/  --docid=17442 -i trnskrbs_5400/col/17442.mpxml  --bl --regTL --form
            3 python ../../src/tasks/DU_ABPTable_T.py modelMultiType tableRow2 --run=trnskrbs_5400
            4 python ../../src/xml_formats/Page2DS.py --pattern=trnskrbs_5400/col/17442_du.mpxml -o trnskrbs_5400/xml/17442.ds_xml  --docid=17442
            5 python src/IE_test.py -i trnskrbs_5400/xml/17442.ds_xml -o trnskrbs_5400/out/17442.ds_xml
            6 python ../../../TranskribusPyClient/src/TranskribusCommands/TranskribusDU_transcriptUploader.py --nodu trnskrbs_5400 5400 17442
            7 python ../../../TranskribusPyClient/src/TranskribusCommands/do_htrRnn.py  <model-name> <dictionary-name> 5400 17442
            
            wait
            8  python ../../../TranskribusPyClient/src/TranskribusCommands/Transkribus_downloader.py 5400 --force 
             #covnert to ds
            9  python ../../src/xml_formats/Page2DS.py --pattern=trnskrbs_5400/col/17442.mpxml -o trnskrbs_5400/xml/17442.ds_xml  --docid=17442
            10 python src/IE_test.py -i trnskrbs_5400/xml/17442.ds_xml -o trnskrbs_5400/out/17442.ds_xml  --doie --usetemplate  
            
        """
        
        #regenerate  mpxml from pxml (optional)      
        #python ../../src/xml_formats/PageXml.py trnskrbs_5400/col/17440 --ext=pxml
        # which entry?  see PageXml mainb :        
        
        ## load dom
        if dom is None:
            self.inputFileName =  os.path.abspath(os.path.join(coldir,TableProcessing.sCOL,docid+TableProcessing.sMPXMLExtension))
            mpxml_doc = self.loadDom()
 
        else:
            # load provided mpxml
            mpxml_doc = dom
                 
        # perform LA  separator, table registration, baseline with normalization  
        #python ../../src/tasks/performCVLLA.py  --coldir=trnskrbs_5400/  --docid=17442 -i trnskrbs_5400/col/17442.mpxml  --bl --regTL --form
        latool= LAProcessor()
#         latool.setParams(dParams)
        latool.coldir = coldir
        latool.docid = docid
        latool.bTemplate, latool.bSeparator , latool.bBaseLine , latool.bRegularTextLine = True,True,True,True
        # creates xml and a new mpxml 
        mpxml_doc,nbPages = latool.performLA(mpxml_doc)
         
        # tag text for BIES cell
        #python ../../src/tasks/DU_ABPTable_T.py modelMultiType tableRow2 --run=trnskrbs_5400
        """ 
            needed : doer = DU_ABPTable_TypedCRF(sModelName, sModelDir,
        """
        doer = DU_ABPTable_TypedCRF(self.sRowModelName, self.sRowModelDir)
        doer.load()
        ## needed predict at file level, and do not store dom, but return it
        BIESFiles  = doer.predict([coldir],docid)
        BIESDom = self.loadDom(BIESFiles[0])
#         res= BIESDom.saveFormatFileEnc('test.mpxml', "UTF-8",True)
        
        # MPXML2DS
        #python ../../src/xml_formats/Page2DS.py --pattern=trnskrbs_5400/col/17442_du.mpxml -o trnskrbs_5400/xml/17442.ds_xml  --docid=17442
        dsconv = primaAnalysis()
        DSBIESdoc = dsconv.convert2DS(BIESDom,self.docid)
         
        # create XMLDOC objcy
        self.ODoc = XMLDSDocument()
        self.ODoc.loadFromDom(DSBIESdoc) #,listPages = range(self.firstPage,self.lastPage+1))        
        # create row
        #python src/IE_test.py -i trnskrbs_5400/xml/17442.ds_xml -o trnskrbs_5400/out/17442.ds_xml
        rdc = RowDetection()
        rdc.findRowsInDoc(self.ODoc)
 
 
        #python ../../src/xml_formats/DS2PageXml.py -i trnskrbs_5400/out/17442.ds_xml --multi
        # DS2MPXML
        DS2MPXML = DS2PageXMLConvertor()
        lPageXml = DS2MPXML.run(self.ODoc.getDom())
        if lPageXml != []:
#             if DS2MPXML.bMultiPages:
            newDoc = MultiPageXml.makeMultiPageXmlMemory(map(lambda (x,y):x,lPageXml))
            outputFileName = os.path.join(self.coldir, sCOL, self.docid+TableProcessing.sMPXMLExtension)
            res= newDoc.saveFormatFileEnc(outputFileName, "UTF-8",True)
#             else:
#                 DS2MPXML.storePageXmlSetofFiles(lPageXml)

        
        #create Transkribus client
        self.myTrKCient = TranskribusClient(sServerUrl=self.server,proxies={'https':'http://cornillon:8000'},loggingLevel=logging.WARN)
        #login
#         res= self.myTrKCient.login(self.myTrKCient, self.options,trace=trace, traceln=traceln)
        res= self.login(self.myTrKCient,trace=trace, traceln=traceln)

        traceln('login: ',res)
        
        #upload
        # python ../../../TranskribusPyClient/src/TranskribusCommands/TranskribusDU_transcriptUploader.py --nodu trnskrbs_5400 5400 17442
        self.upLoadDocument(colid, coldir,docid,sNote='test')
        
        ## apply HTR
        jobid = self.applyHTR(colid,docid, nbPages,self.sHTRmodel,self.sDictName)
        bWait=True
        traceln("waiting for job %s"%jobid)
        while bWait:
            dInfo = self.myTrKCient.getJobStatus(jobid)
            bWait = dInfo['state'] not in [ 'FINISHED', 'FAILED' ]        
     
        
        # download  where???
        # python ../../../TranskribusPyClient/src/TranskribusCommands/Transkribus_downloader.py 5400 --force
        #   coldir is not right!! coldir must refer to the parent folder! 
        self.downloadCollection(colid,coldir,docid,bForce=True)
        
        #done!!
        
        # IE extr
        ## not here: specific to a usecas 
        #python src/IE_test.py -i trnskrbs_5400/xml/17442.ds_xml -o trnskrbs_5400/out/17442.ds_xml  --doie --usetemplate        
    
    
    def processCollection(self,coldir):
        """
            process all files in a colelction
            need mpxml files
        """
        lsDocFilename = sorted(glob.iglob(os.path.join(coldir, "*"+TableProcessing.sMPXMLExtension)))
        lDocId = []
        for sDocFilename in lsDocFilename:
            sDocId = os.path.basename(sDocFilename)[:-len(TableProcessing.sMPXMLExtension)]
            try:
                docid = int(sDocId)
                lDocId.append(docid)
            except ValueError:
                traceln("Warning: folder %s : %s invalid docid, IGNORING IT"%(self.coldir, sDocId))
                continue        
        
        # process each document
        for docid in lDocId:
            traceln("Processing %s : %s "%(self.coldir, sDocId))
            self.processDocument(self.colid, docid)
            traceln("\tProcessing done for %s "%(self.coldir, sDocId))


    def processParameters(self):
        """
            what to do with the parameters provided by the command line
        """
        if self.colid is None:
            print 'collection id missing!'
            sys.exit(1)

        self.bFullCol = self.docid != None

        if self.bRegenerateMPXML and self.docid is not None:
            l = glob.glob(os.path.join(self.coldir,sCOL,self.docid, "*.pxml"))
            doc = MultiPageXml.makeMultiPageXml(l)
            outputFileName = os.path.join(self.coldir, sCOL, self.docid+TableProcessing.sMPXMLExtension)
            res= doc.saveFormatFileEnc(outputFileName, "UTF-8",True)       
            return doc
        return None
        
    def run(self):
        """
            process at colllection level or document level
        """
        newMPXML = self.processParameters()
        if self.bFullCol is None:
            self.processCollection(self.colid)
        else:
            self.processDocument(self.coldir,self.colid, self.docid,newMPXML)
        
if __name__ == "__main__":
          
    
    ## parser for cloud connection
    parser = OptionParser()
    
    
    tableprocessing = TableProcessing()
    tableprocessing.createCommandLineParser()

    tableprocessing.parser.add_option("-s", "--server"  , dest='server', action="store", type="string", default="https://transkribus.eu/TrpServer", help="Transkribus server URL")
    
    tableprocessing.parser.add_option("-l", "--login"   , dest='login' , action="store", type="string", help="Transkribus login (consider storing your credentials in 'transkribus_credentials.py')")    
    tableprocessing.parser.add_option("-p", "--pwd"     , dest='pwd'   , action="store", type="string", help="Transkribus password")

    tableprocessing.parser.add_option("--persist"       , dest='persist', action="store_true", help="Try using an existing persistent session, or log-in and persists the session.")
    
    tableprocessing.parser.add_option("--https_proxy"   , dest='https_proxy'  , action="store", type="string", help="proxy, e.g. http://cornillon:8000")    
    
    tableprocessing.parser.add_option("--pxml", dest="regMPXML", action="store_true",  help="recreate MPXML frol PXML")

    tableprocessing.parser.add_option("--coldir", dest="coldir", action="store", type="string", help="collection folder")
    tableprocessing.parser.add_option("--colid", dest="colid", action="store", type="string", help="collection id")

    tableprocessing.parser.add_option("--docid", dest="docid", action="store", type="string", help="document id")
    tableprocessing.parser.add_option("--useExt", dest="useExt", action="store", type="string", help="generate mpxml using page file .ext")

    ## ROW
    tableprocessing.parser.add_option("--rowmodel", dest="rowmodelname", action="store", type="string", help="row model name")
    tableprocessing.parser.add_option("--rowmodeldir", dest="rowmodeldir", action="store", type="string", help="row model directory")
    ## HTR
    tableprocessing.parser.add_option("--htrmodel", dest="htrmodel", action="store", type="string", help="HTR mode")
    tableprocessing.parser.add_option("--dictname", dest="dictname", action="store", type="string", help="dictionary for HTR")

#    tableprocessing.add_option('-f',"--first", dest="first", action="store", type="int", help="first page to be processed")
#    tableprocessing.add_option('-l',"--last", dest="last", action="store", type="int", help="last page to be processed")    
            
    #parse the command line
    dParams, args = tableprocessing.parseCommandLine()
    #Now we are back to the normal programmatic mode, we set the componenet parameters
    tableprocessing.setParams(dParams)
    
    tableprocessing.run()
    
    