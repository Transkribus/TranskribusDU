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
sys.path.append (os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))) + os.sep+'src')


import common.Component as Component

from ObjectModel.xmlDSDocumentClass import XMLDSDocument
from ObjectModel.XMLDSTEXTClass  import XMLDSTEXTClass
from ObjectModel.treeTemplateClass import treeTemplateClass
from ObjectModel.XMLDSGRAHPLINEClass import XMLDSGRAPHLINEClass
from ObjectModel.XMLDSTABLEClass import XMLDSTABLEClass
from ObjectModel.XMLDSRowClass import XMLDSTABLEROWClass
from ObjectModel.XMLDSCELLClass import XMLDSTABLECELLClass

from spm.spmTableRow import tableRowMiner

import ABPIEOntology

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


    def tagCells(self,table):
        """
            cells are 'fake' cells from template tool:
            type RI  RB RI RE RO
            group text according
        """
        for col in table.getColumns():
            for cell in col.getCells():
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
                        newCell.setIndex(irow+i,cell.getIndex()[1])
                        ## need the deminetion
                        newCell.setObjectsList(c)
                        newCell.resizeMe(XMLDSTEXTClass)
                        newCell.tagMe2()
                        for o in newCell.getObjects():
                            o.setParent(newCell)
                            o.tagMe()
                        table.addCell(newCell)
                    cell.getNode().unlinkNode()
                    del(cell)
        
        
        
        table.buildColumnFromCells()

    def processRows(self,table):
        """
        apply mining to get Y cuts for rows
        """
        rowMiner= tableRowMiner()
        lYcuts = rowMiner.columnMining(table)
        print lYcuts
        table.createRowsWithCuts(lYcuts)
#         table.buildNDARRAY()
        ## finalize the row:col:cell indexes
        
        
        
    
    def labelTable(self,table):
        """
            toy example
            label columns with tags 
        """
        table.getColumns()[0].label()
        
        
    def extractData(self):
        """
            layout 
            tag content
            use scoping for propagating 
                scoping: for tagging and for data
                scope fieldname    scope (fiedlname, filedvalue)   
            find layout level for record completion
            extract data/record
              -inference if IEOnto
        """
        
        
    def run(self,doc):
        """
            
        """
        self.doc= doc
        self.ODoc = XMLDSDocument()
        self.ODoc.loadFromDom(self.doc,listPages=range(self.firstPage,self.lastPage+1))        

        self.lPages= self.ODoc.getPages()   
        
        for page in self.lPages:
            print("page")
            lTables = page.getAllNamedObjects(XMLDSTABLEClass)
            for table in lTables: 
                self.tagCells(table)
                self.processRows(table)
                ## field tagging
                
        ## upload pagexml
        ## now run HTR
        
                
if __name__ == "__main__":

    
    iec = IETest()
    #prepare for the parsing of the command line
    iec.createCommandLineParser()
    iec.add_option("--coldir", dest="coldir", action="store", type="string", help="collection folder")
    iec.add_option("--docid", dest="docid", action="store", type="string", help="document id")
        
    #parse the command line
    dParams, args = iec.parseCommandLine()
    
    #Now we are back to the normal programmatic mode, we set the componenet parameters
    iec.setParams(dParams)
    
    doc = iec.loadDom()
    iec.run(doc)
    iec.writeDom(doc, bIndent=True)
    