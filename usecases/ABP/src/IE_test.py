# -*- coding: utf-8 -*-
"""


    IE module: for test

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
import libxml2
sys.path.append (os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))) + os.sep+'src')


import common.Component as Component
import config.ds_xml_def as ds_xml
from ObjectModel.xmlDSDocumentClass import XMLDSDocument
from ObjectModel.XMLDSTEXTClass  import XMLDSTEXTClass
from ObjectModel.treeTemplateClass import treeTemplateClass
from ObjectModel.XMLDSGRAHPLINEClass import XMLDSGRAPHLINEClass
from ObjectModel.XMLDSTABLEClass import XMLDSTABLEClass
from ObjectModel.XMLDSRowClass import XMLDSTABLEROWClass
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
    description = "description: test"

    #--- INIT -------------------------------------------------------------------------------------------------------------    
    def __init__(self):
        """
        Always call first the Component constructor.
        """
        Component.Component.__init__(self, "IETest", self.usage, self.version, self.description) 
        
        self.colname = None
        self.docid= None
        
        self.sTemplate = None
        self.BuseStoredTemplate = False
        
        # assume table (withrow/col) is provided
        self.bDoIE=False
        
        # for --test
        self.evalData = None
        
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
        if dParams.has_key("docie"):         
            self.bDoIE = dParams["docie"]

        if dParams.has_key("template"):         
            self.sTemplate = dParams["template"]

        if dParams.has_key("UseStoredTemplate"):         
            self.BuseStoredTemplate = dParams["UseStoredTemplate"]            
            
    def tagCells(self, table):
        """
            cells are 'fake' cells from template tool:
            type RI  RB RI RE RO
            group text according
            
        """
        for col in table.getColumns():
            lNewCells=[]
            # keep original positions
            col.resizeMe(XMLDSTABLECELLClass)
            for cell in col.getCells():
#                 print cell
                curChunk=[]
                lChunks = []
#                 print map(lambda x:x.getAttribute('type'),cell.getObjects())
#                 print map(lambda x:x.getID(),cell.getObjects())
                cell.getObjects().sort(key=lambda x:x.getY())
                for txt in cell.getObjects():
#                     print txt.getAttribute("type")
                    if txt.getAttribute("type") == 'RS':
                        if curChunk != []:
                            lChunks.append(curChunk)
                            curChunk=[]
                        lChunks.append([txt])
                    elif txt.getAttribute("type") in ['RI', 'RE']:
                        curChunk.append(txt)
                    elif txt.getAttribute("type") == 'RB':
                        if curChunk != []:
                            lChunks.append(curChunk)
                        curChunk=[txt]
                        
                if curChunk != []:
                    lChunks.append(curChunk)
                    
                if lChunks != []:
                    # create new cells
                    table.delCell(cell)
                    irow= cell.getIndex()[0]
                    for i,c in enumerate(lChunks):
#                         print map(lambda x:x.getAttribute('type'),c)
                        #create a new cell per chunk and replace 'cell'
                        newCell = XMLDSTABLECELLClass()
                        newCell.setPage(cell.getPage())
                        newCell.setParent(table)
                        newCell.setName(ds_xml.sCELL)
                        newCell.setIndex(irow+i,cell.getIndex()[1])
                        newCell.setObjectsList(c)
                        newCell.resizeMe(XMLDSTEXTClass)
                        newCell.tagMe2()
                        for o in newCell.getObjects():
                            o.setParent(newCell)
                            o.tagMe()
#                         table.addCell(newCell)
                        lNewCells.append(newCell)
                    cell.getNode().unlinkNode()
                    del(cell)
            col.setObjectsList(lNewCells[:])
            [table.addCell(c) for c in lNewCells]        
        
#             print col.tagMe()
        

    def processRows(self,table,predefinedCuts=[]):
        """
        apply mining to get Y cuts for rows
        """
        rowMiner= tableRowMiner()
        lYcuts = rowMiner.columnMining(table,predefinedCuts)
        # shift up offset
        [x.setValue(x.getValue()-10) for x in lYcuts]

        table.createRowsWithCuts(lYcuts)
        table.reintegrateCellsInColRow()

        table.buildNDARRAY()
        
        
    def labelTable(self,table):
        """
            toy example
            label columns with tags 
        """
        table.getColumns()[0].label()
        


    def generateRessources(self,table,record):
        """
            Once template has decorated table cells: 
                Extract when fieldvalue when reliable:
                    each field has to define reliable string/context (RE?)
                    
            Not for generic fields  (name, date,...) ?
            
            for generic fields: extract addition content (cum sacrement,....)        
        """
        
        
    def mineForTemplate(self):
        """
            how to automatically generate IETableTemplate
            tag all cells with taggers  (too long?)
            mine per column?  (spm to keep order?)
            
            
        """
        
        
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
        
        table.buildNDARRAY()
        if lTemplate is not None:
            # convert string to tableTemplateObject
            template = tableTemplateClass()
            template.buildFromPattern(lTemplate)
            template.labelTable(table)
        else: return None
        
        #tag fields with template
        for cell in table.getCells():
#             if cell.getFields() != []:
#                 print table.getPage(),cell.getIndex(), cell.getFields(), cell.getContent().encode('utf-8')
            lres= []
            for field in cell.getFields():
                if field is not None:
                    print cell.getIndex(), cell.getFields(), cell.getContent().encode('utf-8')
                    res = field.applyTaggers(cell)
#                     print field, res
                    if res != []:
                        field.setValue(res[0])
                    if res != []:
                        lres.append(res)

        
        ### now at record level ?
        ### scope = propagation using only docObject (hardcoded ?)
        ### where to put the propagation mechanism?
#         myRecord.propagate(table)
        
        #for each cell: take the record and
        ### FR NOW: TAKE THE FIRST COLUMN
        firstCol = table.getColumns()[0]
        for cell in firstCol.getCells():
            myRecord.addCandidate(cell)
        
        myRecord.display()
        
        myRecord.rankCandidates()
        
        lcand = myRecord.getCandidates()
#         print lcand
        
        self.evalData = myRecord.generateOutput(self.evalData)
        
        
    def mergeLineAndCells(self,lPages):
        """
            assing lines(TEXT) to cells
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
                
                

    def processWithTemplate(self,table,dr):
        """
            acocridng to the # of columns, apply the corresponding template 
        """

        lTemplate10 = [  
             ((slice(1,None),slice(0,1)) ,[ dr.getFieldByName('lastname'), dr.getFieldByName('firstname') ])
#            , ((slice(1,None),slice(1,2)) ,[ dr.getFieldByName('occupation'), dr.getFieldByName('religion') ])
#            , ((slice(1,None),slice(2,3)) ,[ dr.getFieldByName('location') ]) 
#            , ((slice(1,None),slice(3,4)) ,[ dr.getFieldByName('familySituation') ])
#            , ((slice(1,None),slice(4,5)) ,[ dr.getFieldByName('deathCause'), dr.getFieldByName('doctor') ])
#            , ((slice(1,None),slice(5,6)) ,[ dr.getFieldByName('deathDate') ])
#            , ((slice(1,None),slice(6,7)) ,[ dr.getFieldByName('burialDate'),dr.getFieldByName('burialLocation') ])
#            , ((slice(1,None),slice(7,8)) ,[ dr.getFieldByName('age')])
#            , ((slice(1,None),slice(8,9)) ,[ dr.getFieldByName('priester')])
#            , ((slice(1,None),slice(9,10)),[ dr.getFieldByName('notes')])
           ]
        
        
        lTemplate = lTemplate10
        
        
        self.extractData(table,dr,lTemplate)
        # select best solutions
        # store inthe proper final format
        
    def run(self,doc):
        """
            
        """
        self.doc= doc
        self.ODoc = XMLDSDocument()
        self.ODoc.loadFromDom(self.doc,listPages = range(self.firstPage,self.lastPage+1))        
#         self.ODoc.loadFromDom(self.doc,listPages = range(30,31))        

        self.lPages = self.ODoc.getPages()   
        
        # not always?
#         self.mergeLineAndCells(self.lPages)
     
        for page in self.lPages:
            print("page"), page.getNumber()
            lTables = page.getAllNamedObjects(XMLDSTABLEClass)
            for table in lTables:
                if self.bDoIE:
                    dr = deathRecord()
                    if self.BuseStoredTemplate:
#                         rowscuts = map(lambda r:r.getY(),table.getRows())
#                         self.tagCells(table)
#                         self.processRows(table,rowscuts)                        
                        self.processWithTemplate(table, dr)
#                     else:
#                         self.extractData(table,dr,lTemplate)
                else:
                    rowscuts = map(lambda r:r.getY(),table.getRows())
                    self.tagCells(table)
                    self.processRows(table,rowscuts)
                ## field tagging
                
        
        ## upload pagexml
        ## now run HTR
        

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
        self.evalData = libxml2.newDoc('1.0')
        root = libxml2.newNode('DOCUMENT')
        self.evalData.setRootElement(root)
        for page in lPages:
            domp=libxml2.newNode('PAGE')
            domp.setProp('number',page.getAttribute('number'))
            root.addChild(domp)
            domp.setProp('template',page.getNode().prop('template'))
            domp.setProp('reftemplate',page.getNode().prop('reftemplate'))
        
        return self.evalData
        
        
    def testFirstNameRecord(self,srefData,srunData, bVisual):
        """
            test firstname in record
            
            group by page
                
        """
        cntOk = cntErr = cntMissed = 0
        RefData = libxml2.parseMemory(srefData.strip("\n"), len(srefData.strip("\n")))
        try:
            RunData = libxml2.parseMemory(srunData.strip("\n"), len(srunData.strip("\n")))
        except:
            RunData = None
            return (cntOk, cntErr, cntMissed)        
         
         
        lRun = []
        if RunData:
            ctxt = RunData.xpathNewContext()
            lpages = ctxt.xpathEval('//%s' % ('PAGE'))
            for page in lpages:
                pnum=page.prop('number')
                xpath = "./%s" % ("RECORD/@firstname")
                ctxt.setContextNode(page)
                ltemp = ctxt.xpathEval(xpath)
                if len(ltemp)==0:
                    lRun.append([])
                for attr in ltemp:
                    if len(attr.getContent()) >0:
                        lRun.append((pnum,attr.getContent().decode('utf-8')))
            ctxt.xpathFreeContext()

        lRef = []
        ctxt = RefData.xpathNewContext()
        lPages = ctxt.xpathEval('//%s' % ('PAGE'))
        for page in lPages[:2]:
            pnum=page.prop('pagenum')
            xpath = "./%s" % ("RECORD/@firstname")
            ctxt.setContextNode(page)
            ltemp = ctxt.xpathEval(xpath)
            if len(ltemp)==0:
                lRef.append([])
            for attr in ltemp:
                if len(attr.getContent()) >0:
                    print pnum,attr.getContent().decode('utf-8')
                    lRef.append((pnum,attr.getContent().decode('utf-8')))
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
            while not bFound and iRef <= refLen - 1:  
                curRef = lRef[iRef]
                if runElt and curRef not in lRefCovered and self.testCompareRecordFirstName(runElt, curRef):
                    bFound = True
                    lRefCovered.append(curRef)
                iRef+=1
            if bFound:
                if bVisual:print "FOUND:", runElt, ' -- ', lRefCovered[-1]
                cntOk += 1
            else:
                cntErr += 1
                bErr = True
                if bVisual:print "ERROR:", runElt
            if bFound or bErr:
                ltisRefsRunbErrbMiss.append( (i, curRef, runElt,bErr, bMiss) )
             
        for i,curRef in enumerate(lRef):
            if curRef not in lRefCovered:
                if bVisual:print "MISSED:", curRef
                ltisRefsRunbErrbMiss.append( (i, curRef, '',False, True) )
                cntMissed+=1
        ltisRefsRunbErrbMiss.sort(key=lambda (x,y,z,t,u):x)

        return (cntOk, cntErr, cntMissed,ltisRefsRunbErrbMiss)              
    
    def testFirstNameLastNameRecord(self,srefData,srunData, bVisual):
        """
            test firstname in record
            
            group by page
                
        """
        cntOk = cntErr = cntMissed = 0
        RefData = libxml2.parseMemory(srefData.strip("\n"), len(srefData.strip("\n")))
        try:
            RunData = libxml2.parseMemory(srunData.strip("\n"), len(srunData.strip("\n")))
        except:
            RunData = None
            return (cntOk, cntErr, cntMissed)        
         
        lRun = []
        if RunData:
            ctxt = RunData.xpathNewContext()
            lpages = ctxt.xpathEval('//%s' % ('PAGE'))
            for page in lpages:
                pnum=page.prop('number')
                #record level!
                xpath = "./%s" % ("RECORD[@firstname and @lastname]")
                ctxt.setContextNode(page)
                lrecord = ctxt.xpathEval(xpath)            
                if len(lrecord)==0:
                    pass
#                     lRun.append([])
                else:
                    for record in lrecord:
                        xpath = "./%s" % ("./@firstname")
                        ctxt.setContextNode(record)
                        lf=  ctxt.xpathEval(xpath)
                        xpath = "./%s" % ("./@lastname")
                        ctxt.setContextNode(record)
                        ln= ctxt.xpathEval(xpath)
                        if len(lf) > 0: # and lf[0].getContent() != ln[0].getContent():
                            lRun.append((pnum,lf[0].getContent().decode('utf-8').encode('utf-8'),ln[0].getContent().decode('utf-8').encode('utf-8')))
            ctxt.xpathFreeContext()

        lRef = []
        ctxt = RefData.xpathNewContext()
        lPages = ctxt.xpathEval('//%s' % ('PAGE'))
        for page in lPages:
            pnum=page.prop('pagenum')
            xpath = "./%s" % ("RECORD")
            ctxt.setContextNode(page)
            lrecord = ctxt.xpathEval(xpath)
            if len(lrecord)==0:
                lRef.append([])
            else:
                for record in lrecord:
                    xpath = "./%s" % ("./@firstname")
                    ctxt.setContextNode(record)
                    lf=  ctxt.xpathEval(xpath)
                    xpath = "./%s" % ("./@lastname")
                    ctxt.setContextNode(record)
                    ln= ctxt.xpathEval(xpath)
                    if len(lf) > 0:
                        lRef.append((pnum,lf[0].getContent().decode('utf-8').encode('utf-8'),ln[0].getContent().decode('utf-8').encode('utf-8')))
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
            print '\t\t===',runElt
            while not bFound and iRef <= refLen - 1:  
                curRef = lRef[iRef]
                if runElt and curRef not in lRefCovered and self.testCompareRecordFirstNameLastName(runElt, curRef):
                    bFound = True
                    lRefCovered.append(curRef)
                iRef+=1
            if bFound:
                if bVisual:print "FOUND:", runElt, ' -- ', lRefCovered[-1]
                cntOk += 1
            else:
                curRef=''
                cntErr += 1
                bErr = True
                if bVisual:print "ERROR:", runElt
            if bFound or bErr:
                ltisRefsRunbErrbMiss.append( (int(runElt[0]), curRef, runElt,bErr, bMiss) )
             
        for i,curRef in enumerate(lRef):
            if curRef not in lRefCovered:
                if bVisual:print "MISSED:", curRef
                ltisRefsRunbErrbMiss.append( (int(curRef[0]), curRef, '',False, True) )
                cntMissed+=1
        ltisRefsRunbErrbMiss.sort(key=lambda (x,y,z,t,u):x)

        return (cntOk, cntErr, cntMissed,ltisRefsRunbErrbMiss)              
        
    def testCompareRecordFirstName(self, refdata, rundata, bVisual=False):
        return refdata[0] == rundata[0] and refdata[1].lower() == rundata[1].lower()
    def testCompareRecordFirstNameLastName(self, refdata, rundata, bVisual=False):
        if refdata[0] != rundata[0]: return False
        TH = 80 #(len(refdata[2])-2.0)/len(refdata[2])*100
        res2, val = matchLCS(TH,(refdata[2].lower(),len(refdata[2])), (rundata[2].lower(),len(rundata[2])) )
        res1, val = matchLCS(TH,(refdata[1].lower(),len(refdata[1])), (rundata[1].lower(),len(rundata[1])) )
#         print refdata,rundata, res2 ,res2

#         if res1 :
#             print refdata[2].lower().encode('utf-8') ,  rundata[2].lower().encode('utf-8') ,val

        return refdata[0] == rundata[0] and res1  and res2
#         return refdata[0] == rundata[0]  and refdata[2].lower() == rundata[2].lower() and refdata[1].lower() == rundata[1].lower()  

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
        
        print self.evalData.serialize('utf-8',1)
        return self.evalData        
        
    def testRun(self, filename, outFile=None):
        """
        testRun is responsible for running the component on this file and returning a string that reflects the result in a way
        that is understandable to a human and to a program. Nicely serialized Python data or XML is fine
        """
        
        
        doc = self.loadDom(filename)
        self.run(doc)
#         self.generateTestOutput()
#         self.createFakeData()
        if outFile: self.writeDom(doc)
        return self.evalData.serialize('utf-8',1)
    
    def testCompare(self, srefData, srunData, bVisual=False):
        """
        Our comparison is very simple: same or different. N
        We anyway return this in term of precision/recall
        If we want to compute the error differently, we must define out own testInit testRecord, testReport
        """
        dicTestByTask = dict()
        dicTestByTask['FullRecord']= self.testFirstNameLastNameRecord(srefData, srunData,bVisual)
    #         dicTestByTask['FirstName']= self.testFirstNameRecord(srefData, srunData,bVisual)
#         dicTestByTask['Year']= self.testYear(srefData, srunData,bVisual)
    
        return dicTestByTask    
                
if __name__ == "__main__":

    
    iec = IETest()
    #prepare for the parsing of the command line
    iec.createCommandLineParser()
    iec.add_option("--coldir", dest="coldir", action="store", type="string", help="collection folder")
    iec.add_option("--docid", dest="docid", action="store", type="string", help="document id")
    iec.add_option("--doie", dest="docie", action="store_true", default=False, help="onlyperform ie")
    iec.add_option("--template", dest="template", action="store", type='string', help="table template for tagging")
    iec.add_option("--usetemplate", dest="UseStoredTemplate", action="store_true", default=False,help="use stored template (ABP)")
    iec.add_option('-f',"--first", dest="first", action="store", type="int", help="first page to be processed")
    iec.add_option('-l',"--last", dest="last", action="store", type="int", help="last page to be processed")

    #parse the command line
    dParams, args = iec.parseCommandLine()
    
    #Now we are back to the normal programmatic mode, we set the componenet parameters
    iec.setParams(dParams)
    
    doc = iec.loadDom()
    iec.run(doc)
    if iec.getOutputFileName() != '-':
        iec.writeDom(doc, bIndent=True) 
    
