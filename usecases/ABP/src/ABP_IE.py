# -*- coding: utf-8 -*-
"""


    IE module: for test

     H. Déjean
    
    copyright Xerox 2017
    copyright Naver Labs Europe 2017,2018

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
from __future__ import absolute_import
from __future__ import  print_function
from __future__ import unicode_literals
from io import open

import sys, os.path
from lxml import etree
sys.path.append (os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))) + os.sep+'TranskribusDU')


import common.Component as Component
import config.ds_xml_def as ds_xml
from ObjectModel.xmlDSDocumentClass import XMLDSDocument
from ObjectModel.XMLDSTEXTClass  import XMLDSTEXTClass
from ObjectModel.treeTemplateClass import treeTemplateClass
from ObjectModel.XMLDSGRAHPLINEClass import XMLDSGRAPHLINEClass
from ObjectModel.XMLDSTABLEClass import XMLDSTABLEClass
from ObjectModel.XMLDSTableRowClass import XMLDSTABLEROWClass
from ObjectModel.XMLDSCELLClass import XMLDSTABLECELLClass

from spm.spmTableRow import tableRowMiner

from ObjectModel.tableTemplateClass import tableTemplateClass

from  ABPIEOntology import *

from util.lcs import matchLCS 

class IETest(Component.Component):
    """
        
    """
    usage = "" 
    version = "v.01"
    description = "description: Information Extraction Tool for the ABP collection (READ project)"

    #--- INIT -------------------------------------------------------------------------------------------------------------    
    def __init__(self):
        """
        Always call first the Component constructor.
        """
        Component.Component.__init__(self, "ABP_IE", self.usage, self.version, self.description) 
        self.usage=self.usage = "python %prog" + self.usageComponent
        self.colname = None
        self.docid= None
        
        self.sTemplate = None
        self.BuseStoredTemplate = False
        
        # HTR model id
        self.htrModelID = None
        
        # IE model
        self.sModelDir = None
        self.sModelName = None
        
        # for --test
        self.evalData = None
        
    def setParams(self, dParams):
        """
        Always call first the Component setParams
        Here, we set our internal attribute according to a possibly specified value (otherwise it stays at its default value)
        """
        Component.Component.setParams(self, dParams)
        if "coldir" in dParams: 
            self.colname = dParams["coldir"]
            
        if "docid" in dParams:         
            self.docid = dParams["docid"]

        if "htrid" in dParams:         
            self.htrModelID = dParams["htrid"]

        if "template" in dParams:         
            self.sTemplate = dParams["template"]

        if "UseStoredTemplate" in dParams:         
            self.BuseStoredTemplate = dParams["UseStoredTemplate"]            
            
        if 'modelName' in dParams:
            self.sModelName = dParams['modelName']
        if 'modelDir' in dParams:
            self.sModelDir = dParams['modelDir']            
            
        
    def labelTable(self,table):
        """
            toy example
            label columns with tags 
        """
        table.getColumns()[0].label()
        


    def extractData(self,table,myRecord, lTemplate):
        """
            layout 
            tag content
            [use scoping for propagating 
                scoping: for tagging and for data
                scope fieldname    scope (fiedlname, fieldvalue)]   
            
            find if possible a contiguous repetition of records
            
            
            find layout level for record completion
            extract data/record
              -inference if IEOnto
              
              
            
        """
        self.bDebug = True
        table.buildNDARRAY()
        if lTemplate is not None:
            # convert string to tableTemplateObject
            template = tableTemplateClass()
            template.buildFromPattern(lTemplate)
            template.labelTable(table)
        else: return None
        
        #tag fields with template
        for cell in table.getCells():
            if cell.getFields() != []:
                if self.bDebug:print(table.getPage(),cell.getIndex(), cell.getFields(), cell.getContent())
            for field in cell.getFields():
                if field is not None:
                    res = field.applyTaggers(cell)
                    # res [ (token,label,score) ...]
                    extractedValues = field.extractLabel(res)
                    if extractedValues != []:
#                         extractedValues = map(lambda offset,value,label,score:(value,score),extractedValues)
                        extractedValues = list(map(lambda x:(x[1],x[3]),extractedValues))

                        field.setValue(extractedValues)
                        if self.bDebug: print ('found:',field, field.getValue())
        

        ### now at record level ?
        ### scope = propagation using only docObject (hardcoded ?)
        ### where to put the propagation mechanism?
#         myRecord.propagate(table)
        
        ## 'backpropagation:  select the rows, and collection subobjects with fields  (cells)
        
        for row in table.getRows():
            #if not row.isHeaders():
            myRecord.addCandidate(row)
        
#         #for each cell: take the record and
#         ### FR NOW: TAKE THE FIRST COLUMN
#         firstCol = table.getColumns()[0]
#         for cell in firstCol.getCells():
#             myRecord.addCandidate(cell)
        
        
        myRecord.rankCandidates()
        
        lcand = myRecord.getCandidates()
#         print lcand
#         myRecord.display()        
        
    def mergeLineAndCells(self,lPages):
        """
            assign lines(TEXT) to cells
        """
        
        for page in lPages:
            lLines = page.getAllNamedObjects(XMLDSTEXTClass)
            lCells = page.getAllNamedObjects(XMLDSTABLECELLClass)
            dAssign={}
            for line in lLines:
                bestscore= 0.0
                for cell in lCells:
                    ratio = line.ratioOverlap(cell)
                    if ratio > bestscore:
                        bestscore=ratio
                        dAssign[line]=cell
                
            [dAssign[line].addObject(line) for line in dAssign.keys()]
            [cell.getObjects().sort(key=lambda x:x.getY()) for cell in lCells]
                
       
    def testGTText(self,page):
        """
            extract region text and parse it 
        """         
        from contentProcessing.taggerTrainKeras import DeepTagger
        
        myTagger = DeepTagger()
        myTagger.bPredict = True
        myTagger.sModelName = '2mix_cm'
        myTagger.dirName = 'IEdata/model/'
        myTagger.loadModels()
        
        for region in page.getObjects():
#             print region.getContent().encode('utf-8')
            res= myTagger.predict([region.getContent()])
            try:
                res= myTagger.predict([region.getContent()])
#                 print res
            except: print ('SENT WITH ISSUES : [%s]' % (region.getContent().encode('utf-8')))



    def learnTemplate(self):
        """
            for handwritten template: categorize a set of columns
            using lstm? -> which GT?  introduce variation, which noise??
            always some differences to be expected!
            htr a large scale of pages from different docs
            assuming some templates: assign columns correctly?
        """
    def loadTemplates(self):
        """
            in ABPIEOntology
        """

    def htrWithTemplate(self,table,template,htrModelId):
        """
            perform an HTR with dictionaries specific to each column
            
            need: docid, pageid
        """
        
        # for the current column: need to get tablecells ids
        # more efficient(?why more efficient?) to have it at column level: not cell ; so just after table template tool
        for col in table.getColumns():
            lCellsID = map(lambda x:x.getID(),col.getCells())
            for id in lCellsID: print(id)
            
        
        
    def selectTemplat(self,lTemplates):
        """
         if a list of templates is available: 
             take a couple pages and perform IE: simply sum up the score of the tagger
             
        """
    def     processWithTemplate(self,table,dr):
        """
            according to the # of columns, apply the corresponding template 
        """
        # selection of the dictionaries per columns
        # template 5,10: first col = numbering
#         lTemplateHTR = [  
#              ((slice(1,None),slice(0,1))  ,[ 'abp_names', 'names_aux','numbering'])
#             , ((slice(1,None),slice(1,2)) ,[ 'abp_profession','religion' ])
#             , ((slice(1,None),slice(2,3)) ,[ 'abp_location' ]) 
#             , ((slice(1,None),slice(3,4)) ,[ 'abp_family' ])
#              ,((slice(1,None),slice(4,5)) ,[ 'deathreason','artz'])  
#             , ((slice(1,None),slice(5,6)) ,[ 'abp_dates' ])
#             , ((slice(1,None),slice(6,7)) ,[ 'abp_dates','abp_location' ])
#             , ((slice(1,None),slice(7,8)) ,[ 'abp_age'])
# #            , ((slice(1,None),slice(8,9)) ,[ dr.getFieldByName('priester')])
# #            , ((slice(1,None),slice(9,10)),[ dr.getFieldByName('notes')])
#            ]
        
        lTemplateIE2 = [
             ((slice(1,None),slice(0,1))  ,[ 'numbering'],[ dr.getFieldByName('numbering') ])
            , ((slice(1,None),slice(1,2))  ,[ 'abp_names', 'names_aux','numbering','religion'],[ dr.getFieldByName('lastname'), dr.getFieldByName('firstname'),dr.getFieldByName('religion')  ])
            , ((slice(1,None),slice(2,3)) ,[ 'abp_profession','religion' ]        ,[ dr.getFieldByName('occupation'), dr.getFieldByName('religion') ])
            , ((slice(1,None),slice(3,4))  ,[ 'abp_location' ]                    ,[ dr.getFieldByName('location') ]) 
            , ((slice(1,None),slice(4,5)) ,[ 'abp_family' ]                       ,[ dr.getFieldByName('situation') ])
             ,((slice(1,None),slice(5,6)) ,[ 'deathreason','artz']                ,[ dr.getFieldByName('deathreason'),dr.getFieldByName('doktor')])
            , ((slice(1,None),slice(6,7)) ,[]                                     , [ ])  #binding
            , ((slice(1,None),slice(7,8)) ,[ 'abp_dates' ]                        ,[ dr.getFieldByName('deathDate') ])
            , ((slice(1,None),slice(8,9)) ,[ 'abp_dates','abp_location' ]         ,[ dr.getFieldByName('burialDate'),dr.getFieldByName('burialLocation') ])
            , ((slice(1,None),slice(9,10)) ,[ 'abp_age']                           ,[ dr.getFieldByName('age')])
#            , ((slice(1,None),slice(9,10)) ,[ dr.getFieldByName('priester')])
#            , ((slice(1,None),slice(10,11)),[ dr.getFieldByName('notes')])
           ]        
        
        lTemplateIE = [  
             ((slice(1,None),slice(0,1))  ,[ 'abp_names', 'names_aux','numbering','religion'],[ dr.getFieldByName('lastname'), dr.getFieldByName('firstname') ,dr.getFieldByName('religion')])
            , ((slice(1,None),slice(1,2)) ,[ 'abp_profession','religion' ]        ,[ dr.getFieldByName('occupation'), dr.getFieldByName('religion') ])
            , ((slice(1,None),slice(2,3))  ,[ 'abp_location' ]                    ,[ dr.getFieldByName('location') ]) 
            , ((slice(1,None),slice(3,4)) ,[ 'abp_family' ]                       ,[ dr.getFieldByName('situation') ])
            #[] binding
            , ((slice(1,None),slice(4,6)) ,[ 'deathreason','artz']                ,[ dr.getFieldByName('deathreason'),dr.getFieldByName('doktor')])
            , ((slice(1,None),slice(6,7)) ,[ 'abp_dates' ]                        ,[ dr.getFieldByName('deathDate') ])
            , ((slice(1,None),slice(7,8)) ,[ 'abp_dates','abp_location' ]         ,[ dr.getFieldByName('burialDate'),dr.getFieldByName('burialLocation') ])
            , ((slice(1,None),slice(8,9)) ,[ 'abp_age']                           ,[ dr.getFieldByName('age')])
#            , ((slice(1,None),slice(9,10)) ,[ dr.getFieldByName('priester')])
#            , ((slice(1,None),slice(10,11)),[ dr.getFieldByName('notes')])
           ]
        
        
        lTemplate = lTemplateIE
#         lTemplate = lTemplateIE2
        
#         if self.htrModelID is not None: self.htrWithTemplate(table, lTemplate, self.htrModelID)
        
        self.extractData(table,dr,lTemplate)
        # select best solutions
        # store inthe proper final format
        return dr 
    
    def run(self,doc):
        """
        main issue: how to select the template: to be done by CVL
            assuming IE and htr info are stored in the template
        
        """
#         self.firstPage = 9
#         self.lastPage= 9
        self.doc= doc
        self.ODoc = XMLDSDocument()
        self.ODoc.loadFromDom(self.doc,listPages = range(self.firstPage,self.lastPage+1))        

        self.lPages = self.ODoc.getPages()   
        
        dr = deathRecord(self.sModelName,self.sModelDir)     
        
        ## selection of the templates first with X tables
        
        ### 
        
        for page in self.lPages:
            print("page: ", page.getNumber())
#             self.testGTText(page)
#             continue
            lTables = page.getAllNamedObjects(XMLDSTABLEClass)
            
            for table in lTables:
                if self.BuseStoredTemplate:
                    self.processWithTemplate(table, dr)
        
        self.evalData = dr.generateOutput(self.evalData)
#         print self.evalData.serialize('utf-8',True)

    def generateTestOutput(self):
        """
  <PAGE number="1" pagenum="1" nbrecords="10" years="0">
    <RECORD lastname="Riesinger" firstname="Korona" role="Verstorbener" location="Neuhofen" occupation="" year="0" month="0" day="0"/>
    <RECORD lastname="Ringseisen" firstname="Georg" role="Verstorbener" location="Etzing" occupation="" year="0" month="0" day="0"/>
    <RECORD lastname="Nebel" firstname="Theresia" role="Verstorbener" location="Sandbach" occupation="" year="0" month="0" day="0"/>
    <RECORD lastname="Schlögl" firstname="Cäcilia" role="Verstorbener" location="Stampfing" occupation="" year="0" month="0" day="0"/>
    <RECORD lastname="Riedinger" firstname="Theresia" role="Verstorbener" location="Lapperding" occupation="Austragsbäuerin" year="0" month="0" day="0"/>
    <RECORD lastname="Wührer" firstname="Joseph" role="Verstorbener" location="Haizing" occupation="" year="0" month="0" day="0"/>
    <RECORD lastname="Wilmerdinger" firstname="Theresia" role="Verstorbener" location="Hausmanning" occupation="" year="0" month="0" day="0"/>
    <RECORD lastname="Ratzinger" firstname="Mathias" role="Verstorbener" location="Kafferding" occupation="Bauer" year="0" month="0" day="0"/>
    <RECORD lastname="Deixelberger" firstname="Joseph" role="Verstorbener" location="Gaishofen" occupation="Inwohner" year="0" month="0" day="0"/>
    <RECORD lastname="Beham" firstname="Martin" role="Verstorbener" location="Socking" occupation="Austragsbauer" year="0" month="0" day="0"/>
  </PAGE>            
            
        """
        root = etree.Element('DOCUMENT')
        self.evalData = etree.ElementTree(root)
        for page in lPages:
            
            domp=etree.Elemen('PAGE')
            domp.set('number',page.getAttribute('number'))
            domp.set('pagenum',os.path.basename(page.getAttribute('imageFilename'))[:-4])
            root.append(domp)
            domp.set('template',page.getNode().prop('template'))
            domp.set('reftemplate',page.getNode().prop('reftemplate'))
        
        return self.evalData
        
        
    
    def testFirstNameLastNameRecord(self,srefData,srunData, bVisual):
        """
            test firstname in record
            
            group by page
                
        """

        cntOk = cntErr = cntMissed = 0
#         srefData = srefData.decode('utf-8')
        #.strip("\n")
        
        RefData = etree.XML(srefData.strip("\n").encode('utf-8'))
        RunData = etree.XML(srunData.strip("\n").encode('utf-8'))            
         
        lRef = []
        lPages = RefData.xpath('//%s' % ('PAGE[@number]'))

        lRefKeys={}
        for page in lPages:
            pnum=page.get('number')
            key= page.get('pagenum')
            lRefKeys[key]=1
            xpath = "./%s" % ("RECORD")
            lrecord = page.xpath(xpath)
            if len(lrecord)==0:
                lRef.append([])
            else:
                for record in lrecord:
                    xpath = "./%s" % ("./@firstname")
                    lf=  record.xpath(xpath)
                    xpath = "./%s" % ("./@lastname")
                    ln= record.xpath(xpath)
                    if len(lf) > 0:
                        lRef.append((pnum,key,lf[0],ln[0]))

        
        lPageMapping={}
        lRunKeys={}
        lRun = []
        if RunData:
            lpages = RunData.xpath('//%s' % ('PAGE[@number]'))
            for page in lpages:
                pnum=page.get('number')
                key= page.get('pagenum')
                if key in lRefKeys:
                    lPageMapping[key]=pnum
                    #record level!
                    xpath = "./%s" % ("RECORD[@firstname and @lastname]")
                    lrecord = page.xpath(xpath)            
                    if len(lrecord)==0:
                        pass
                    else:
                        for record in lrecord:
                            xpath = "./%s" % ("./@firstname")
                            lf=  record.xpath(xpath)
                            xpath = "./%s" % ("./@lastname")
                            ln= record.xpath(xpath)
                            if len(lf) > 0: # and lf[0].getContent() != ln[0].getContent():
    #                             lRun.append((pnum,lf[0].getContent().decode('utf-8').encode('utf-8'),ln[0].getContent().decode('utf-8').encode('utf-8')))
                                lRun.append((pnum,key,lf[0],ln[0]))
                                
        
        runLen = len(lRun)
        refLen = len(lRef)
#         bVisual = True
        ltisRefsRunbErrbMiss= list()
        lRefCovered = []
        for i in range(0,len(lRun)):
            iRef =  0
            bFound = False
            bErr , bMiss= False, False
            runElt = lRun[i]
#             print '\t\t===',runElt
            while not bFound and iRef <= refLen - 1:  
                curRef = lRef[iRef]
                if runElt and curRef not in lRefCovered and self.testCompareRecordFirstNameLastName(curRef,runElt):
                    bFound = True
                    lRefCovered.append(curRef)
                iRef+=1
            if bFound:
                if bVisual:print("FOUND:", runElt, ' -- ', lRefCovered[-1])
                cntOk += 1
            else:
                curRef=''
                cntErr += 1
                bErr = True
                if bVisual:print("ERROR:", runElt)
            if bFound or bErr:
                ltisRefsRunbErrbMiss.append( (runElt[1],int(runElt[0]), curRef, runElt,bErr, bMiss) )
        for i,curRef in enumerate(lRef):
            if curRef not in lRefCovered:
                if bVisual:print("MISSED:", curRef)
#                 ltisRefsRunbErrbMiss.append( (curRef[1],int(curRef[0]), curRef, '',False, True) )
                ltisRefsRunbErrbMiss.append( (curRef[1],int(lPageMapping[curRef[1]]), curRef, '',False, True) )
                
                cntMissed+=1
                
        #sorting
#         ltisRefsRunbErrbMiss.sort(key=lambda k,x,y,z,t,u:k)
        ltisRefsRunbErrbMiss.sort(key=lambda x:x[0])

        return (cntOk, cntErr, cntMissed,ltisRefsRunbErrbMiss)              
    
    def testRecordField(self,fieldName,srefData,srunData, bVisual):
        """
            test firstname in record
            
            group by page
                
        """

        cntOk = cntErr = cntMissed = 0
#         srefData = srefData.decode('utf-8')
        #.strip("\n")
        
        RefData = etree.XML(srefData.strip("\n").encode('utf-8'))
        RunData = etree.XML(srunData.strip("\n").encode('utf-8'))

        
        lPageMapping={}
        lRef = []
        lPages = RefData.xpath('//%s' % ('PAGE[@number]'))
        lRefKeys={}
        for page in lPages:
            pnum=page.get('number')
            key=page.get('pagenum')
            lRefKeys[key]=1
            xpath = "./%s" % ("RECORD")
            lrecord = page.xpath(xpath)
            if len(lrecord)==0:
                lRef.append([])
            else:
                for record in lrecord:
                    xpath = "./%s" % ("./@%s"%fieldName)
                    ln= record.xpath(xpath)
                    if len(ln[0])>0:
                        lRef.append((pnum,key,ln[0]))
                        
        
        lRun = []
        if RunData:
            lpages = RunData.xpath('//%s' % ('PAGE[@number]'))
            for page in lpages:
                pnum=page.get('number')
                key=page.get('pagenum')
                if key in lRefKeys:
                    #record level!
                    lPageMapping[key]=pnum
                    xpath = "./%s" % ("RECORD[@%s]"%fieldName)
                    lrecord = page.xpath(xpath)            
                    if len(lrecord)==0:
                        pass
    #                     lRun.append([])
                    else:
                        for record in lrecord:
                            xpath = "./%s" % ("./@%s"%fieldName)
                            lf= record.xpath(xpath)
                            if len(lf[0])>0:
                                lRun.append((pnum,key,lf[0]))

        ctxt.xpathFreeContext()

        runLen = len(lRun)
        refLen = len(lRef)
#         bVisual = True
        ltisRefsRunbErrbMiss= list()
        lRefCovered = []
        for i in range(0,len(lRun)):
            iRef =  0
            bFound = False
            bErr , bMiss= False, False
            runElt = lRun[i]
#             print '\t\t===',runElt
            while not bFound and iRef <= refLen - 1:  
                curRef = lRef[iRef]
                if runElt and curRef not in lRefCovered and self.testCompareRecordField(curRef,runElt):
                    bFound = True
                    lRefCovered.append(curRef)
                iRef+=1
            if bFound:
                if bVisual:print("FOUND:", runElt, ' -- ', lRefCovered[-1])
                cntOk += 1
            else:
                curRef=''
                cntErr += 1
                bErr = True
                if bVisual:print("ERROR:", runElt)
            if bFound or bErr:
                ltisRefsRunbErrbMiss.append( (runElt[1],int(runElt[0]), curRef, runElt,bErr, bMiss) )
        for i,curRef in enumerate(lRef):
            if curRef not in lRefCovered:
                if bVisual:print("MISSED:", curRef)
#                 ltisRefsRunbErrbMiss.append( (curRef[1],int(curRef[0]), curRef, '',False, True) )
                ltisRefsRunbErrbMiss.append( (curRef[1],int(lPageMapping[curRef[1]]), curRef, '',False, True) )

                cntMissed+=1
        ltisRefsRunbErrbMiss.sort(key=lambda x:x[0])

        return (cntOk, cntErr, cntMissed,ltisRefsRunbErrbMiss)  
    
    def testCompareRecordFirstName(self, refdata, rundata, bVisual=False):
        return refdata[0] == rundata[0] and refdata[1].lower() == rundata[1].lower()
    
    def testCompareRecordFirstNameLastName(self, refdata, rundata, bVisual=False):
        # same page !!
        # N same key !
#         print  refdata[1],rundata[1]
        if refdata[1] != rundata[1]: return False
        
        TH = 75 #(len(refdata[2])-2.0)/len(refdata[2])*100
        refall= refdata[2].lower()+refdata[3].lower()
        reflen= len(refdata[2])+len(refdata[3])
        runall= rundata[2].lower()+rundata[3].lower()
        runlen= len(rundata[2])+len(rundata[3])    
        runall.replace('n̄','nn') 
        runall.replace('m̄','mm')
         
#         res2, val = matchLCS(TH,(refdata[2].lower(),len(refdata[2])), (rundata[2].lower(),len(rundata[2])) )
        res1, val = matchLCS(TH,(refall,reflen), (runall,runlen) )

        return res1 # and res2

    def testCompareRecordField(self, refdata, rundata, bVisual=False):
        # same page !!
        if refdata[1] != rundata[1]: return False
        
        TH = 80
         
        res, val = matchLCS(TH,(refdata[2].lower(),len(refdata[2])), (rundata[2].lower(),len(rundata[2])) )
        return res 
    
    def testCompareFullRecord(self, refdata, rundata, bVisual=False):
        bOK=True
        for i,attr in enumerate(refdata):
            bOK = bOK and refdata[i] == rundata[i]
        return bOK

    ################ TEST ##################
    
    
    def createFakeData(self):
        """
            for testing purpose
  <PAGE number="1" pagenum="1" nbrecords="10" years="0">
    <RECORD lastname="Riesinger" firstname="Korona" role="Verstorbener" location="Neuhofen" occupation="" year="0" month="0" day="0"/>
    <RECORD lastname="Ringseisen" firstname="Georg" role="Verstorbener" location="Etzing" occupation="" year="0" month="0" day="0"/>
    <RECORD lastname="Nebel" firstname="Theresia" role="Verstorbener" location="Sandbach" occupation="" year="0" month="0" day="0"/>
    <RECORD lastname="Schlögl" firstname="Cäcilia" role="Verstorbener" location="Stampfing" occupation="" year="0" month="0" day="0"/>
    <RECORD lastname="Riedinger" firstname="Theresia" role="Verstorbener" location="Lapperding" occupation="Austragsbäuerin" year="0" month="0" day="0"/>
    <RECORD lastname="Wührer" firstname="Joseph" role="Verstorbener" location="Haizing" occupation="" year="0" month="0" day="0"/>
    <RECORD lastname="Wilmerdinger" firstname="Theresia" role="Verstorbener" location="Hausmanning" occupation="" year="0" month="0" day="0"/>
    <RECORD lastname="Ratzinger" firstname="Mathias" role="Verstorbener" location="Kafferding" occupation="Bauer" year="0" month="0" day="0"/>
    <RECORD lastname="Deixelberger" firstname="Joseph" role="Verstorbener" location="Gaishofen" occupation="Inwohner" year="0" month="0" day="0"/>
    <RECORD lastname="Beham" firstname="Martin" role="Verstorbener" location="Socking" occupation="Austragsbauer" year="0" month="0" day="0"/>
  </PAGE>            
                       
        """
        self.evalData = libxml2.newDoc('1.0')
        root = libxml2.newNode('DOCUMENT')
        self.evalData.setRootElement(root)
        domp=libxml2.newNode('PAGE')
        domp.setProp('number','182')
        domp.setProp('nbrecords','None')
        domp.setProp('years','1876;1877')

        root.addChild(domp)
        record = libxml2.newNode('RECORD')
        domp.addChild(record)
        record.setProp('lastname','Riedinger')
        record.setProp('firstname','Theresia')
        
        print(etree.tostring(self.evalData,encoding='unicode',pretty_print=True))
        return self.evalData        
        
    def testRun(self, filename, outFile=None):
        """
        testRun is responsible for running the component on this file and returning a string that reflects the result in a way
        that is understandable to a human and to a program. Nicely serialized Python data or XML is fine
        """
        
        self.evalData=None
        doc = self.loadDom(filename)
        self.run(doc)
#         self.generateTestOutput()
#         self.createFakeData()
        if outFile: self.writeDom(doc)
        # return unicode
        return  etree.tostring(self.evalData,encoding='unicode', pretty_print=True)
    
    def testCompare(self, srefData, srunData, bVisual=False):
        """
        Our comparison is very simple: same or different. N
        We anyway return this in term of precision/recall
        If we want to compute the error differently, we must define out own testInit testRecord, testReport
        """
        dicTestByTask = dict()
        dicTestByTask['Names']= self.testFirstNameLastNameRecord(srefData, srunData,bVisual)
#         dicTestByTask['occupation']= self.testRecordField('occupation',srefData, srunData,bVisual)
#         dicTestByTask['location']= self.testRecordField('location',srefData, srunData,bVisual)

#         dicTestByTask['Year']= self.testYear(srefData, srunData,bVisual)
    
        return dicTestByTask    
     
     
    def testRecordHtml(self, filename, data, nOk, nErr, nMiss):
        
        
        if nOk == None:
            assert nErr == None and nMiss == None, "INTERNAL ERROR"
            #we are reporting on multiple tasks!!
            lltisRefsRunbErrbMiss = data #this is a list of (taskName, nOk, nErr, nMiss, ltisRefsRunbErrbMiss)
        else:
            lltisRefsRunbErrbMiss = [ (None, nOk, nErr, nMiss, data) ]
            
        #let's produce an HTML report!! 
        sCollecDir = os.path.dirname(self.testDirXML)
        sCollec = os.path.basename(sCollecDir)
        sFile =   os.path.basename(self.getRefFileName(filename))[:-4]
        sViewBaseUrl = "http://" #+ sHttpHost 
        
        
        fHtml = open(self.getHtmRunFileName(filename), "w",encoding='utf-8')
        
        sCss = """
<style type="text/css">
.OK {
color: green;
}
.Error {
color: red;
}
.Error\+Miss {
color: darkred;
}
.Miss {
color: orange;
}
</style>       
"""
        sRpt = self.makeHTMLReportHeader(sViewBaseUrl, "dla_pdf", sCss
                                         , sCollec + " - " + sFile
                                         , sCollec + " - " + sFile)

        fHtml.write(sRpt)
        
        #sRpt += " Doc Prec. Recall F1\t   nOk\t  nErr\t nMiss\tFilename\n"
        for taskName, nOk, nErr, nMiss, ltisRefsRunbErrbMiss in lltisRefsRunbErrbMiss:        
            if taskName == None: taskName = ""
            sRpt = """
            <hr/>
            <h2>%s</h2>
<table>
    <tr align="left">
        <th></th>
        <th>Page</th>
        <th>Reference</th>
        <th>Run</th>
        <th></th>
    </tr>
"""        % taskName
            fHtml.write(sRpt)
            ipnum_prev = -1

            for (key,ipnum, sRef, sRun, bErr, bMiss) in ltisRefsRunbErrbMiss:
                if bErr and bMiss:
                    sRptType = "Error+Miss"
                else:
                    if bErr:
                        sRptType = "Error"
                    elif bMiss:
                        sRptType = "Miss"
                    else:
                        sRptType = "OK"
                    
                sPfFile = sCollec + "/" + sFile + "/" + "pf%06d"%ipnum
                    
                srefenc= " ".join(x for x in sRef[2:])
                srun = " ".join(x for x in sRun[2:])
                if ipnum > ipnum_prev: #a new page
                    fHtml.write('<tr ><td>%s (%s)</td></tr>' % (key,ipnum))
                    fHtml.write('<tr class="%s"><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>\n' % (sRptType, sRptType
                                , "" #ipnum
                                , srefenc  #sRef
                                , srun #sRun
#                                 , " - ".join(lsViews)
                                ))
                else: #some more results for the same pafe
                    fHtml.write('<tr class="%s"><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>\n' % (sRptType, sRptType
                                , ""
                                , srefenc
                                , srun
                                , ""
                                ))
                ipnum_prev = ipnum
                
            fHtml.write('</table>')
            fHtml.write('<p/>')
            fHtml.write( self.genHtmlTableReport(sCollec, None, [(filename, nOk, nErr, nMiss, None)] ) )

        fHtml.write('<hr>')
        
        fHtml.close()
        
        return

                
if __name__ == "__main__":

    iec = IETest()
    #prepare for the parsing of the command line
    iec.createCommandLineParser()
    iec.add_option("--coldir", dest="coldir", action="store", type="string", help="collection folder")
    iec.add_option("--docid", dest="docid", action="store", type="string", help="document id")
    iec.add_option("--htrid", dest="htrid", action="store", type="string", default=None,help="HTR model ID")
    iec.add_option("--template", dest="template", action="store", type='string', help="table template for tagging")
    iec.add_option("--usetemplate", dest="UseStoredTemplate", action="store_true", default=False,help="use stored template (ABP)")
    iec.add_option('-f',"--first", dest="first", action="store", type="int", help="first page to be processed")
    iec.add_option('-l',"--last", dest="last", action="store", type="int", help="last page to be processed")
    iec.add_option("--modelName", dest="modelName", action="store", type="string", help="model to be used")
    iec.add_option("--modelDir", dest="modelDir", action="store", type="string", help="model folder")

    #parse the command line
    dParams, args = iec.parseCommandLine()
    
    #Now we are back to the normal programmatic mode, we set the componenet parameters
    iec.setParams(dParams)
    doc = iec.loadDom()
    iec.run(doc)
    if iec.evalData is not None:
#         iec.writeEval( etree.tostring(iec.evalData,xml_declaration=True,encoding='utf-8',pretty_print=True), os.path.join(iec.colname,'run',iec.docid+'.run'), True)
        iec.evalData.write( os.path.join(iec.colname,'run',iec.docid+'.run'),
                            xml_declaration=True,encoding='utf-8',pretty_print=True)
        
#     if iec.getOutputFileName() != '-':
#         iec.writeDom(doc, bIndent=True) 
    
