# -*- coding: utf-8 -*-
"""


    IETableWorkflow.py

    Process a full collection with table analysis and IE extraction
    Based on template
        
    H. Déjean
    
    copyright Xerox 2017
    READ project 


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
"""




import sys, os
import logging 
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
from TranskribusCommands.TranskribusDU_transcriptUploader import  TranskribusDUTranscriptUploader
from TranskribusCommands.Transkribus_downloader import TranskribusDownloader
from TranskribusCommands.do_analyzeLayoutNew import DoLAbatch
from TranskribusCommands.do_htrRnn import DoHtrRnn
from TranskribusCommands import  sCOL

import common.Component as Component
from TranskribusPyClient.common.trace import traceln, trace

from xml_formats.PageXml import PageXml
from tasks.DU_ABPTable_T import DU_ABPTable_TypedCRF
from xml_formats.PageXml import MultiPageXml
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
        self.useExtForMPXML = False

        self.bRegenerateMPXML = False
                
        self.sRowModelName = None
        self.sRowModelDir = None
        
        self.sHTRmodel = None
        self.sDictName = None
        
    def setParams(self, dParams):
        """
        Always call first the Component setParams
        Here, we set our internal attribute according to a possibly specified value (otherwise it stays at its default value)
        """
        Component.Component.setParams(self, dParams)
        if "coldir" in dParams: 
            self.coldir = dParams["coldir"]
        if "colid" in dParams: 
            self.colid = dParams["colid"]
        if "colid" in dParams:         
            self.docid = dParams["docid"]
        if "useExt" in dParams:         
            self.useExtForMPXML  = dParams["useExt"]            
        
        if 'regMPXML' in dParams:
            self.bRegenerateMPXML=True
            
        if "rowmodelname" in dParams:         
            self.sRowModelName = dParams["rowmodelname"]
        if "rowmodeldir" in dParams:         
            self.sRowModelDir = dParams["rowmodeldir"]

        if "htrmodel" in dParams: 
            self.sHTRmodel = dParams["htrmodel"]
        if "dictname" in dParams: 
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

    def downloadCollection(self,colid,destDir,docid,bNoImg=True,bForce=False):
        """
            download colID
            
            replace destDir by '.'  ?
        """
        destDir="."
#         options.server, proxies, loggingLevel=logging.WARN)
        #download
        downloader = TranskribusDownloader(self.myTrKCient.getServerUrl(),self.myTrKCient.getProxies())
        downloader.setSessionId(self.myTrKCient.getSessionId())
        traceln("- Downloading collection %s to folder %s"%(colid, os.path.abspath(destDir)))
#         col_ts, colDir = downloader.downloadCollection(colid, destDir, bForce=options.bForce, bNoImage=options.bNoImage)
        col_ts, colDir, ldocids, dFileListPerDoc = downloader.downloadCollection(colid, destDir, bForce = bForce, bNoImage=bNoImg,sDocId=docid)
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
        uploader = TranskribusDUTranscriptUploader(self.myTrKCient.getServerUrl(),self.myTrKCient.getProxies())
        uploader.setSessionId(self.myTrKCient.getSessionId())
        traceln("- uploading document %s to collection %s" % (docid,colid))
        uploader.uploadDocumentTranscript(colid, docid, os.path.join(coldir,sCOL), sNote, 'NLE Table', sTranscripExt, iVerbose=False)
        traceln("- Done")
        return 

    def applyLA_URO(self,colid,docid,nbpages):
        """
        apply textline finder 
        """
        # do the job...
#         if options.trp_doc:
#             trpdoc =  json.load(codecs.open(options.trp_doc, "rb",'utf-8'))
#             docId,sPageDesc = doer.buildDescription(colId,options.docid,trpdoc)

        traceln('process %s pages...'%nbpages)
        lretJobIDs = []
        for i in range(1,nbpages+1):
            LA = DoLAbatch(self.myTrKCient.getServerUrl(),self.myTrKCient.getProxies())
            LA._trpMng.setSessionId(self.myTrKCient.getSessionId())
            LA.setSessionId(self.myTrKCient.getSessionId())
            _,sPageDesc = LA.buildDescription(colid,"%s/%s"%(docid,i))    
            sPageDesc = LA.jsonToXMLDescription(sPageDesc)
            _,lJobIDs = LA.run(colid, sPageDesc,"CITlabAdvancedLaJob",False)
            traceln(lJobIDs)
            lretJobIDs.extend(lJobIDs)
            traceln("- LA running for page %d job:%s"%(i,lJobIDs))
        return lretJobIDs
                
                
    def applyHTRForRegions(self,colid,docid,nbpages,modelname,dictionary):
        """
            apply an htr model at region level 
        """
        
        htrComp = DoHtrRnn(self.myTrKCient.getServerUrl(),self.myTrKCient.getProxies())
        htrComp._trpMng.setSessionId(self.myTrKCient.getSessionId())
        htrComp.setSessionId(self.myTrKCient.getSessionId())
        
        _,sPageDesc = htrComp.buildDescription(colid,"%s/%s"%(docid,nbpages))    
         
        sPages= "1-%d"%(nbpages)
        sModelID = None
        # get modelID
        lColModels =  self.myTrKCient.listRnns(colid)
        for model in lColModels:
#             print model['htrId'], type(model['htrId']), modelname,type(modelname)
            if str(model['htrId']) == str(modelname):
                sModelID  = model['htrId']
                traceln('model id = %s'%sModelID)
                #some old? models do not have params field
#             try: traceln("%s\t%s\t%s" % (model['htrId'],model['name'],model['params']))
#             except KeyError: traceln("%s\t%s\tno params" % (model['htrId'],model['name']))
        if  sModelID == None: raise Exception, "no model ID found for %s" %(modelname)
        ret = htrComp.htrRnnDecode(colid, sModelID, dictionary, docid, sPageDesc,bDictTemp=False)
        traceln(ret)
        return ret
        
    def applyHTR(self,colid,docid,nbpages,modelname,dictionary):
        """
            apply HTR on docid
            
            htr id is needed: we have htrmodename
        """
        htrComp = DoHtrRnn(self.myTrKCient.getServerUrl(),self.myTrKCient.getProxies())
        htrComp._trpMng.setSessionId(self.myTrKCient.getSessionId())
        htrComp.setSessionId(self.myTrKCient.getSessionId())
        
        _,sPageDesc = htrComp.buildDescription(colid,"%s/%s"%(docid,nbpages))    
         
        sPages= "1-%d"%(nbpages)
        sModelID = None
        # get modelID
        lColModels =  self.myTrKCient.listRnns(colid)
        for model in lColModels:
#             print model['htrId'], type(model['htrId']), modelname,type(modelname)
            if str(model['htrId']) == str(modelname):
                sModelID  = model['htrId']
                traceln('model id = %s'%sModelID)
                #some old? models do not have params field
#             try: traceln("%s\t%s\t%s" % (model['htrId'],model['name'],model['params']))
#             except KeyError: traceln("%s\t%s\tno params" % (model['htrId'],model['name']))
        if  sModelID == None: raise Exception, "no model ID found for %s" %(modelname)
        ret = htrComp.htrRnnDecode(colid, sModelID, dictionary, docid, sPageDesc,bDictTemp=False)
        traceln(ret)
        return ret
               
    
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

        #create Transkribus client
        self.myTrKCient = TranskribusClient(sServerUrl=self.server,proxies={},loggingLevel=logging.WARN)
        #login
        _ = self.login(self.myTrKCient,trace=trace, traceln=traceln)
        
#         self.downloadCollection(colid,coldir,docid,bNoImg=False,bForce=True)

        ## load dom
        if dom is None:
            self.inputFileName =  os.path.abspath(os.path.join(coldir,TableProcessing.sCOL,docid+TableProcessing.sMPXMLExtension))
            mpxml_doc = self.loadDom()
            nbPages = MultiPageXml.getNBPages(mpxml_doc)
        else:
            # load provided mpxml
            mpxml_doc = dom
            nbPages = MultiPageXml.getNBPages(mpxml_doc)
                    
#         ### table registration: need to compute/select???   the template
#         # perform LA  separator, table registration, baseline with normalization  
#         #python ../../src/tasks/performCVLLA.py  --coldir=trnskrbs_5400/  --docid=17442 -i trnskrbs_5400/col/17442.mpxml  --bl --regTL --form
#         tableregtool= LAProcessor()
# #         latool.setParams(dParams)
#         tableregtool.coldir = coldir
#         tableregtool.docid = docid
#         tableregtool.bTemplate, tableregtool.bSeparator , tableregtool.bBaseLine , tableregtool.bRegularTextLine = True,False,False,False
#         # creates xml and a new mpxml 
#         mpxml_doc,nbPages = tableregtool.performLA(mpxml_doc)
#          
#          

#         self.upLoadDocument(colid, coldir,docid,sNote='NLE workflow;table reg done')        
         
        lJobIDs = self.applyLA_URO(colid, docid, nbPages)
        return 
    
        bWait=True
        assert  lJobIDs  != []
        jobid=lJobIDs[-1]
        traceln("waiting for job %s"%jobid)
        while bWait:
            dInfo = self.myTrKCient.getJobStatus(jobid)
            bWait = dInfo['state'] not in [ 'FINISHED', 'FAILED' ]   
         
         
        ## coldir???
        self.downloadCollection(colid,coldir,docid,bNoImg=True,bForce=True)
 
        ##STOP HERE FOR DAS newx testset:
        return 
          
        # tag text for BIES cell
        #python ../../src/tasks/DU_ABPTable_T.py modelMultiType tableRow2 --run=trnskrbs_5400
        """ 
            needed : doer = DU_ABPTable_TypedCRF(sModelName, sModelDir,
        """
        doer = DU_ABPTable_TypedCRF(self.sRowModelName, self.sRowModelDir)
        doer.load()
        ## needed predict at file level, and do not store dom, but return it
        rowpath=os.path.join(coldir,"col")
        BIESFiles  = doer.predict([rowpath],docid)
        BIESDom = self.loadDom(BIESFiles[0])
#         res= BIESDom.saveFormatFileEnc('test.mpxml', "UTF-8",True)
          
        # MPXML2DS
        #python ../../src/xml_formats/Page2DS.py --pattern=trnskrbs_5400/col/17442_du.mpxml -o trnskrbs_5400/xml/17442.ds_xml  --docid=17442
        dsconv = primaAnalysis()
        DSBIESdoc = dsconv.convert2DS(BIESDom,self.docid)
           
        # create XMLDOC object
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
            newDoc = MultiPageXml.makeMultiPageXmlMemory(map(lambda xy:xy[0],lPageXml))
            outputFileName = os.path.join(self.coldir, sCOL, self.docid+TableProcessing.sMPXMLExtension)
            newDoc.write(outputFileName, xml_declaration=True,encoding="UTF-8",pretty_print=True)
#             else:
#                 DS2MPXML.storePageXmlSetofFiles(lPageXml)
 
        return 
     
        #upload
        # python ../../../TranskribusPyClient/src/TranskribusCommands/TranskribusDU_transcriptUploader.py --nodu trnskrbs_5400 5400 17442
        self.upLoadDocument(colid, coldir,docid,sNote='NLE workflow;table row done')
         
         
        ## apply HTR
        ## how to deal with specific dictionaries?
         
        ## here need to know the ontology and the template
        
        nbPages=1
        jobid = self.applyHTR(colid,docid, nbPages,self.sHTRmodel,self.sDictName)
        bWait=True
        traceln("waiting for job %s"%jobid)
        while bWait:
            dInfo = self.myTrKCient.getJobStatus(jobid)
            bWait = dInfo['state'] not in [ 'FINISHED', 'FAILED' ,'CANCELED']        
     
        
        # download  where???
        # python ../../../TranskribusPyClient/src/TranskribusCommands/Transkribus_downloader.py 5400 --force
        #   coldir is not right!! coldir must refer to the parent folder! 
        self.downloadCollection(colid,coldir,docid,bNoImg=True,bForce=True)
        
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
            print('collection id missing!')
            sys.exit(1)

        self.bFullCol = self.docid != None

        if self.bRegenerateMPXML and self.docid is not None:
            l = glob.glob(os.path.join(self.coldir,sCOL,self.docid, "*.pxml"))
            doc = MultiPageXml.makeMultiPageXml(l)
            outputFileName = os.path.join(self.coldir, sCOL, self.docid+TableProcessing.sMPXMLExtension)
            doc.write(outputFileName, xml_declaration=True,encoding="UTF-8",pretty_print=True)       
            return doc
        return None
        
    def run(self):
        """
            process at collection level or document level
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
    tableprocessing.parser.add_option("--htrid", dest="htrmodel", action="store", type="string", help="HTR mode")
    tableprocessing.parser.add_option("--dictname", dest="dictname", action="store", type="string", help="dictionary for HTR")

#    tableprocessing.add_option('-f',"--first", dest="first", action="store", type="int", help="first page to be processed")
#    tableprocessing.add_option('-l',"--last", dest="last", action="store", type="int", help="last page to be processed")    
            
    #parse the command line
    dParams, args = tableprocessing.parseCommandLine()
    #Now we are back to the normal programmatic mode, we set the componenet parameters
    tableprocessing.setParams(dParams)
    
    tableprocessing.run()
    
    