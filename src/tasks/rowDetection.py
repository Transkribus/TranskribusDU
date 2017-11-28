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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))


import common.Component as Component
from common.trace import traceln
import config.ds_xml_def as ds_xml
from ObjectModel.xmlDSDocumentClass import XMLDSDocument
from ObjectModel.XMLDSTEXTClass  import XMLDSTEXTClass
from ObjectModel.XMLDSTABLEClass import XMLDSTABLEClass
from ObjectModel.XMLDSCELLClass import XMLDSTABLECELLClass

from spm.spmTableRow import tableRowMiner
from xml_formats.Page2DS import primaAnalysis

class RowDetection(Component.Component):
    """
        
            row detection once
                column detection done
                BIES tagging done for text elements
                
    """
    usage = "" 
    version = "v.01"
    description = "description: test"

    #--- INIT -------------------------------------------------------------------------------------------------------------    
    def __init__(self):
        """
        Always call first the Component constructor.
        """
        Component.Component.__init__(self, "RowDetection", self.usage, self.version, self.description) 
        
        self.colname = None
        self.docid= None

        self.do2DS= False
        
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
        if dParams.has_key("dsconv"):         
            self.do2DS = dParams["dsconv"]
                        
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
                    elif txt.getAttribute("type") == 'RO':
                        ## add Other as well???
                        curChunk.append(txt)
                        
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

        # shift up offset / find a better way to do this: integration skewing 
        [x.setValue(x.getValue()-10) for x in lYcuts]

        table.createRowsWithCuts(lYcuts)
        table.reintegrateCellsInColRow()

        table.buildNDARRAY()
        
        
        
#     def mergeLineAndCells(self,lPages):
#         """
#             assing lines(TEXT) to cells
#         """
#         
#         for page in lPages:
#             lLines = page.getAllNamedObjects(XMLDSTEXTClass)
#             lCells = page.getAllNamedObjects(XMLDSTABLECELLClass)
#             dAssign={}
#             for line in lLines:
#                 bestscore= 0.0
#                 for cell in lCells:
#                     ratio = line.ratioOverlap(cell)
#                     if ratio > bestscore:
#                         bestscore=ratio
#                         dAssign[line]=cell
#                 
#             [dAssign[line].addObject(line) for line in dAssign.keys()]
#             [cell.getObjects().sort(key=lambda x:x.getY()) for cell in lCells]
#                 


    def findRowsInTable(self,table):
        """ 
            find row in this table
        """
        rowscuts = map(lambda r:r.getY(),table.getRows())
        self.tagCells(table)
        self.processRows(table,rowscuts)
        
                     
    def findRowsInDoc(self,ODoc):
        """
        find rows
        """
        self.lPages = ODoc.getPages()   
        
        # not always?
#         self.mergeLineAndCells(self.lPages)
     
        for page in self.lPages:
            traceln("page: %d" %  page.getNumber())
            lTables = page.getAllNamedObjects(XMLDSTABLEClass)
            for table in lTables:
                rowscuts = map(lambda r:r.getY(),table.getRows())
                self.tagCells(table)
                self.processRows(table,rowscuts)        

    def run(self,doc):
        """
           load dom and find rows 
        """
        
        # conver to DS if needed
        if self.do2DS:
            dsconv = primaAnalysis()
            self.doc = dsconv.convert2DS(doc,self.docid)
        else:
            self.doc= doc
        self.ODoc = XMLDSDocument()
        self.ODoc.loadFromDom(self.doc,listPages = range(self.firstPage,self.lastPage+1))        
#         self.ODoc.loadFromDom(self.doc,listPages = range(30,31))        

        self.findRowsInDoc(self.ODoc)
        
        return self.doc
        
#         self.lPages = self.ODoc.getPages()   
        
#         # not always?
# #         self.mergeLineAndCells(self.lPages)
#      
#         for page in self.lPages:
#             traceln("page: %d" %  page.getNumber())
#             lTables = page.getAllNamedObjects(XMLDSTABLEClass)
#             for table in lTables:
#                 rowscuts = map(lambda r:r.getY(),table.getRows())
#                 self.tagCells(table)
#                 self.processRows(table,rowscuts)
        

    ################ TEST ##################
    

    def testRun(self, filename, outFile=None):
        """
        testRun is responsible for running the component on this file and returning a string that reflects the result in a way
        that is understandable to a human and to a program. Nicely serialized Python data or XML is fine

    
        evaluate using ABP new table dataset with tablecell
        """
        
        self.evalData=None
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

                
if __name__ == "__main__":

    
    rdc = RowDetection()
    #prepare for the parsing of the command line
    rdc.createCommandLineParser()
    rdc.add_option("--coldir", dest="coldir", action="store", type="string", help="collection folder")
    rdc.add_option("--docid", dest="docid", action="store", type="string", help="document id")
    rdc.add_option("--dsconv", dest="dsconv", action="store_true", default=False, help="convert page fomar to DS")

    rdc.add_option('-f',"--first", dest="first", action="store", type="int", help="first page to be processed")
    rdc.add_option('-l',"--last", dest="last", action="store", type="int", help="last page to be processed")

    #parse the command line
    dParams, args = rdc.parseCommandLine()
    
    #Now we are back to the normal programmatic mode, we set the componenet parameters
    rdc.setParams(dParams)
    
    doc = rdc.loadDom()
    doc = rdc.run(doc)
    if rdc.getOutputFileName() != '-':
        rdc.writeDom(doc, bIndent=True) 
    
