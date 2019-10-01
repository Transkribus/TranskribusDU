# -*- coding: utf-8 -*-
"""


    build Table columns

     H. Déjean
    

    copyright Naver 2018
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

from lxml import etree

import common.Component as Component
from common.trace import traceln
import config.ds_xml_def as ds_xml
from ObjectModel.xmlDSDocumentClass import XMLDSDocument
from ObjectModel.XMLDSObjectClass import XMLDSObjectClass
from ObjectModel.XMLDSTEXTClass  import XMLDSTEXTClass
from ObjectModel.XMLDSTABLEClass import XMLDSTABLEClass
from ObjectModel.XMLDSCELLClass import XMLDSTABLECELLClass
from ObjectModel.XMLDSTableColumnClass import XMLDSTABLECOLUMNClass
from spm.spmTableColumn import tableColumnMiner
from xml_formats.Page2DS import primaAnalysis

class columnDetection(Component.Component):
    """
        build table column 
    """
    usage = "" 
    version = "v.01"
    description = "description: column Detection"

    #--- INIT -------------------------------------------------------------------------------------------------------------    
    def __init__(self):
        """
        Always call first the Component constructor.
        """
        Component.Component.__init__(self, "columnDetection", self.usage, self.version, self.description) 
        
        self.colname = None
        self.docid= None

        self.do2DS= False
        
        self.THHighSupport = 0.75
        # for --test
        self.bCreateRef = False
        self.evalData = None
        
    def setParams(self, dParams):
        """
        Always call first the Component setParams
        Here, we set our internal attribute according to a possibly specified value (otherwise it stays at its default value)
        """
        Component.Component.setParams(self, dParams)
#         if dParams.has_key("coldir"): 
#             self.colname = dParams["coldir"]
        if "docid" in dParams:         
            self.docid = dParams["docid"]
        if "dsconv" in dParams:         
            self.do2DS = dParams["dsconv"]
        
        self.bMining = dParams["mining"]
                         
        if "createref" in dParams:         
            self.bCreateRef = dParams["createref"]                        
    
    
    
    def createTable(self,page):
        """
            BB of all elements?
            todo: Ignore O!
        """
        x1,y1,x2,y2 = self.getBounbingBox(page)
        if x1 is None: return None
        myTable = XMLDSTABLEClass()
        myTable.setX(x1)
        myTable.setY(y1)
        myTable.setWidth(x2-x1)
        myTable.setHeight(y2-y1)
                
        page.addObject(myTable)
        return myTable
        
    def processPage(self,page,emptyTable):
        from util.XYcut import mergeSegments
        
        ### skrinking to be done: use center ?
#         lCuts, _, _ = mergeSegments([(x.getX(),x.getX2(),x) for x in page.getAllNamedObjects(XMLDSTEXTClass)],0)
        lCuts, _, _ = mergeSegments([(x.getX(),x.getX()+0.25*x.getWidth(),x) for x in page.getAllNamedObjects(XMLDSTEXTClass)],0)
#         lCuts, _, _ = mergeSegments([(x.getX()+0.5*x.getWidth()-0.25*x.getWidth(),x.getX()+0.5*x.getWidth()+0.25*x.getWidth(),x) for x in page.getAllNamedObjects(XMLDSTEXTClass)],0)

        for i, (x,_,cut) in enumerate(lCuts):
            ll =list(cut)
            ll.sort(key=lambda x:x.getY())
            #add column
            myCol= XMLDSTABLECOLUMNClass(i)
            myCol.setPage(page)
            myCol.setParent(emptyTable)
            emptyTable.addObject(myCol)
            myCol.setX(x)
            myCol.setY(emptyTable.getY())
            myCol.setHeight(emptyTable.getHeight())
            if i +1  < len(lCuts):
                myCol.setWidth(lCuts[i+1][0]-x)
            else: # use table 
                myCol.setWidth(emptyTable.getX2()-x)
            emptyTable.addColumn(myCol)
            if not self.bMining:
                myCol.tagMe(ds_xml.sCOL)
    
    def getBounbingBox(self,page):
        lElts= page.getAllNamedObjects(XMLDSTEXTClass)
        if lElts == []: return None,None,None,None
        
        lX1,lX2,lY1,lY2 = zip(*[(x.getX(),x.getX2(),x.getY(),x.getY2()) for x in lElts]) 
        return min(lX1), min(lY1), max(lX2), max(lY2)
    
    def findColumnsInDoc(self,lPages):
        """
        find columns for each table in ODoc
        """
        
        for page in lPages:
            traceln("page: %d" %  page.getNumber())
            lch,lcv = self.mergeHorVerClusters(page)
#             table = self.createTable(page)
#             if table is not None:
# #                 table.tagMe(ds_xml.sTABLE)
#                 self.processPage(page,table)        

        
    def createContourFromListOfElements(self, lElts):
        """
            create a polyline from a list of elements
            input : list of elements
            output: Polygon object
        """
        from shapely.geometry import Polygon as pp
        from shapely.ops import cascaded_union
        
        lP = []
        for elt in lElts:
            
            sPoints = elt.getAttribute('points')
            if sPoints is None:
                lP.append(pp([(elt.getX(),elt.getY()),(elt.getX(),elt.getY2()), (elt.getX2(),elt.getY2()),(elt.getX2(),elt.getY())] ))
            else:    
                lP.append(pp([(float(x),float(y)) for x,y in zip(*[iter(sPoints.split(','))]*2)]))
        try:ss = cascaded_union(lP)
        except ValueError: 
            print(lElts,lP)
            return None
        
        return ss #list(ss.convex_hull.exterior.coords)    
    
    def mergeHorVerClusters(self,page):
        """
            build Horizontal and vertical clusters
        """
        from util import TwoDNeighbourhood as  TwoDRel
        lTexts = page.getAllNamedObjects(XMLDSTEXTClass)

        for e in lTexts:
            e.lright=[]
            e.lleft=[]
            e.ltop=[]
            e.lbottom=[]
        lVEdge = TwoDRel.findVerticalNeighborEdges(lTexts)         
        for  a,b in lVEdge:
            a.lbottom.append( b )
            b.ltop.append(a)                 
        for elt in lTexts: 
            # dirty!
            elt.setHeight(max(5,elt.getHeight()-3))
            elt.setWidth(max(5,elt.getWidth()-3))
            TwoDRel.rotateMinus90degOLD(elt)              
        lHEdge = TwoDRel.findVerticalNeighborEdges(lTexts)
        for elt in lTexts:
#             elt.tagMe()
            TwoDRel.rotatePlus90degOLD(elt)
#         return     
        for  a,b in lHEdge:
            a.lright.append( b )
            b.lleft.append(a)             
#         ss
        for elt in lTexts:
            elt.lleft.sort(key = lambda x:x.getX(),reverse=True)
#             elt.lright.sort(key = lambda x:x.getX())
            if len(elt.lright) > 1:
                elt.lright = []
            elt.lright.sort(key = lambda x:elt.signedRatioOverlapY(x),reverse=True)
#             print (elt, elt.getY(), elt.lright)
            elt.ltop.sort(key = lambda x:x.getY())
            if len(elt.lbottom) >1:
                elt.lbottom = []
            elt.lbottom.sort(key = lambda x:elt.signedRatioOverlapX(x),reverse=True)
            


        lHClusters = []
        # Horizontal  
        lTexts.sort(key = lambda x:x.getX())
        lcovered=[]
        for text in lTexts:
            if text not in lcovered:
#                 print ('START :', text, text.getContent())
                lcovered.append(text)
                lcurRow = [text]
                curText= text
                while curText is not None:
                    try:
                        nextT = curText.lright[0]
#                         print ('\t',[(x,curText.signedRatioOverlapY(x)) for x in curText.lright])
                        if nextT not in lcovered:
                            lcurRow.append(nextT)
                            lcovered.append(nextT)
                        curText = nextT
                    except IndexError:curText = None
#                 lHClusters.append(lcurRow)         
#                 print ("FINAL", list(map(lambda x:(x,x.getContent()),lcurRow)) )
                if len(lcurRow) > 0:
#                     # create a contour for visualization
#                     # order by col: get top and  bottom polylines for them
                    contour = self.createContourFromListOfElements(lcurRow)
                    lHClusters.append((lcurRow,contour))



        # Vertical
        lVClusters = [] 
        lTexts.sort(key = lambda x:x.getY())
        lcovered=[]
        for text in lTexts:
            if text not in lcovered:
#                 print ('START :', text, text.getContent())
                lcovered.append(text)
                lcurCol = [text]
                curText= text
                while curText is not None:
                    try:
                        nextT = curText.lbottom[0]
#                         print ('\t',[(x,curText.signedRatioOverlapY(x)) for x in curText.lright])
                        if nextT not in lcovered and len(nextT.lbottom) == 1:
                            lcurCol.append(nextT)
                            lcovered.append(nextT)
                        curText = nextT
                    except IndexError:curText = None
                
# #                 print ("FINAL", list(map(lambda x:(x,x.getContent()),lcurCol)) )
                if len(lcurCol)> 0:
                    contour = self.createContourFromListOfElements(lcurCol)
                    lVClusters.append((lcurCol,contour))                
                    if contour:
#                         print (contour.bounds)   
                        r = XMLDSObjectClass()
                        r.setName('cc')
                        r.setParent(page)
#                         r.addAttribute('points',spoints)
                        x1,y1,x2,y2 = contour.bounds
                        r.setXYHW(x1, y1, y2-y1, x2-x1)
                        page.addObject(r)
#                         r.tagMe('BLOCK')

        print (page.getAllNamedObjects('cc'))    
        return lHClusters, lVClusters
    
    
    def documentMining(self,lPages):
        """
        need to clean up REGION nodes   
         
        """
        seqMiner = tableColumnMiner()
        seqMiner.columnMining(lPages,self.THHighSupport,sTag=ds_xml.sCOL)        
    
    def checkInputFormat(self,lPages):
        """
            delete regions : copy regions elements at page object
            unlink subnodes
        """
        for page in lPages:
            
            lRegions = page.getAllNamedObjects("REGION")
            lElts=[]
            [lElts.extend(x.getObjects()) for x in lRegions]
            [page.addObject(x,bDom=True) for x in lElts]
            [page.removeObject(x,bDom=True) for x in lRegions]

    def run(self,doc):
        """
           load dom and find rows 
        """
        # conver to DS if needed
        if self.bCreateRef:
            if self.do2DS:
                dsconv = primaAnalysis()
                doc = dsconv.convert2DS(doc,self.docid)
            
            refdoc = self.createRef(doc)
            return refdoc
            # single ref per page
            refdoc= self.createRefPerPage(doc)
            return None
        
        if self.do2DS:
            dsconv = primaAnalysis()
            self.doc = dsconv.convert2DS(doc,self.docid)
        else:
            self.doc= doc
        self.ODoc = XMLDSDocument()
        self.ODoc.loadFromDom(self.doc,listPages = range(self.firstPage,self.lastPage+1))        
        self.lPages = self.ODoc.getPages()
        
        self.checkInputFormat(self.lPages)     
        self.findColumnsInDoc(self.lPages)

        if self.bMining:
            self.documentMining(self.lPages)
        
        if self.bCreateRef:
            refdoc = self.createRef(self.doc)
            return refdoc
        
        
#         if self.do2DS:
#             # bakc to PageXml
#             conv= DS2PageXMLConvertor()
#             lPageXDoc = conv.run(self.doc)
#             conv.storeMultiPageXml(lPageXDoc,self.getOutputFileName())
#             print self.getOutputFileName()
#             return None
        return self.doc
        
        

    ################ TEST ##################
    

    def testRun(self, filename, outFile=None):
        """
        evaluate using ABP new table dataset with tablecell
        """
        
        self.evalData=None
        doc = self.loadDom(filename)
        doc =self.run(doc)
        self.evalData = self.createRef(doc)
        if outFile: self.writeDom(doc)
#         return self.evalData.serialize('utf-8',1)
        return etree.tostring(self.evalData,encoding='unicode',pretty_print=True)
    
    
    def overlapX(self,zone):
        
    
        [a1,a2] = self.getX(),self.getX()+ self.getWidth()
        [b1,b2] = zone.getX(),zone.getX()+ zone.getWidth()
        return min(a2, b2) >=   max(a1, b1) 
        
    def overlapY(self,zone):
        [a1,a2] = self.getY(),self.getY() + self.getHeight()
        [b1,b2] = zone.getY(),zone.getY() + zone.getHeight()
        return min(a2, b2) >=  max(a1, b1)      
    def signedRatioOverlap(self,z1,z2):
        """
         overlap self and zone
         return surface of self in zone 
        """
        [x1,y1,h1,w1] = z1.getX(),z1.getY(),z1.getHeight(),z1.getWidth()
        [x2,y2,h2,w2] = z2.getX(),z2.getY(),z2.getHeight(),z2.getWidth()
        
        fOverlap = 0.0
        
        if self.overlapX(z2) and self.overlapY(z2):
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
            
        return  fOverlap     
    
    def findSignificantOverlap(self,TOverlap,ref,run):
        """
            return 
        """ 
        pref,rowref= ref
        prun, rowrun= run
        if pref != prun: return  False
        
        return rowref.ratioOverlap(rowrun) >=TOverlap 
        
        
    def testCPOUM(self, TOverlap, srefData, srunData, bVisual=False):
        """
            TOverlap: Threshols used for comparing two surfaces
            
            
            Correct Detections:
            under and over segmentation?
        """

        cntOk = cntErr = cntMissed = 0
        
        RefData = etree.XML(srefData.strip("\n").encode('utf-8'))
        RunData = etree.XML(srunData.strip("\n").encode('utf-8'))
#         try:
#             RunData = libxml2.parseMemory(srunData.strip("\n"), len(srunData.strip("\n")))
#         except:
#             RunData = None
#             return (cntOk, cntErr, cntMissed)        
        lRun = []
        if RunData:
            lpages = RunData.xpath('//%s' % ('PAGE'))
            for page in lpages:
                pnum=page.get('number')
                #record level!
                lRows = page.xpath(".//%s" % ("ROW"))
                lORows = map(lambda x:XMLDSTABLECOLUMNClass(0,x),lRows)
                for row in lORows:
                    row.fromDom(row._domNode)
                    row.setIndex(row.getAttribute('id'))
                    lRun.append((pnum,row))            
        print (lRun)
        
        lRef = []
        lPages = RefData.xpath('//%s' % ('PAGE'))
        for page in lPages:
            pnum=page.get('number')
            lRows = page.xpath(".//%s" % ("ROW"))
            lORows = map(lambda x:XMLDSTABLECOLUMNClass(0,x),lRows)
            for row in lORows:    
                row.fromDom(row._domNode)
                row.setIndex(row.getAttribute('id'))
                lRef.append((pnum,row))  


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
                if runElt and curRef not in lRefCovered and self.findSignificantOverlap(TOverlap,runElt, curRef):
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
                ltisRefsRunbErrbMiss.append( (int(runElt[0]), curRef, runElt,bErr, bMiss) )
             
        for i,curRef in enumerate(lRef):
            if curRef not in lRefCovered:
                if bVisual:print("MISSED:", curRef)
                ltisRefsRunbErrbMiss.append( (int(curRef[0]), curRef, '',False, True) )
                cntMissed+=1
        ltisRefsRunbErrbMiss.sort(key=lambda xyztu:xyztu[0])

#         print cntOk, cntErr, cntMissed,ltisRefsRunbErrbMiss
        return (cntOk, cntErr, cntMissed,ltisRefsRunbErrbMiss)              
                
        
    def testCompare(self, srefData, srunData, bVisual=False):
        """
        as in Shahad et al, DAS 2010

        Correct Detections 
        Partial Detections 
        Over-Segmented 
        Under-Segmented 
        Missed        
        False Positive
                
        """
        dicTestByTask = dict()
        dicTestByTask['T50']= self.testCPOUM(0.50,srefData,srunData,bVisual)
#         dicTestByTask['T75']= self.testCPOUM(0.750,srefData,srunData,bVisual)
#         dicTestByTask['T100']= self.testCPOUM(0.50,srefData,srunData,bVisual)

    #         dicTestByTask['FirstName']= self.testFirstNameRecord(srefData, srunData,bVisual)
#         dicTestByTask['Year']= self.testYear(srefData, srunData,bVisual)
    
        return dicTestByTask    
    
    def createColumnsWithCuts(self,lXCuts,table,tableNode,bTagDoc=False):
        """
            create column dom node
        """
        
        prevCut = None
        lXCuts.sort()
        for index,cut in enumerate(lXCuts):
            # first correspond to the table: no rpw
            if prevCut is not None:
                colNode= etree.Element("COL")
                tableNode.append(colNode)
                colNode.set('x',str(prevCut))
                colNode.set('width',"{:.2f}".format(cut - prevCut))
                colNode.set('y',str(table.getY()))
                colNode.set('height',str(table.getHeight()))
                colNode.set('id',str(index-1))
            prevCut= cut
        
        #last
        cut=table.getX2()
        colNode= etree.Element("COL")
        tableNode.append(colNode)
        colNode.set('x',"{:.2f}".format(prevCut))
        colNode.set('width',"{:.2f}".format(cut - prevCut))
        colNode.set('y',str(table.getY()))
        colNode.set('height',str(table.getHeight()))        
        colNode.set('id',str(index))

            
    def createRef(self,doc):
        """
            create a ref file from the xml one
        """
        self.ODoc = XMLDSDocument()
        self.ODoc.loadFromDom(doc,listPages = range(self.firstPage,self.lastPage+1))        
  
  
        root=etree.Element("DOCUMENT")
        refdoc=etree.ElementTree(root)
        

        for page in self.ODoc.getPages():
            #imageFilename="..\col\30275\S_Freyung_021_0001.jpg" width="977.52" height="780.0">
            pageNode = etree.Element('PAGE')
            pageNode.set("number",page.getAttribute('number'))
            pageNode.set("pagekey",os.path.basename(page.getAttribute('imageFilename')))
            pageNode.set("width",page.getAttribute('width'))
            pageNode.set("height",page.getAttribute('height'))

            root.append(pageNode)   
            lTables = page.getAllNamedObjects(XMLDSTABLEClass)
            for table in lTables:
                dCol={}
                tableNode = etree.Element('TABLE')
                tableNode.set("x",table.getAttribute('x'))
                tableNode.set("y",table.getAttribute('y'))
                tableNode.set("width",table.getAttribute('width'))
                tableNode.set("height",table.getAttribute('height'))
                pageNode.append(tableNode)
                for cell in table.getAllNamedObjects(XMLDSTABLECELLClass):
                    try:dCol[int(cell.getAttribute("col"))].append(cell)
                    except KeyError:dCol[int(cell.getAttribute("col"))] = [cell]
        
                lXcuts = []
                for colid in sorted(dCol.keys()):
                    lXcuts.append(min(list(map(lambda x:x.getX(),dCol[colid]))))
                self.createColumnsWithCuts(lXcuts,table,tableNode)

        return refdoc
    
if __name__ == "__main__":

    
    rdc = columnDetection()
    #prepare for the parsing of the command line
    rdc.createCommandLineParser()
#     rdc.add_option("--coldir", dest="coldir", action="store", type="string", help="collection folder")
    rdc.add_option("--docid", dest="docid", action="store", type="string", help="document id")
    rdc.add_option("--dsconv", dest="dsconv", action="store_true", default=False, help="convert page format to DS")
    rdc.add_option("--createref", dest="createref", action="store_true", default=False, help="create REF file for component")
    rdc.add_option("--mining", dest="mining", action="store_true", default=False, help="apply pattern mining")

    rdc.add_option('-f',"--first", dest="first", action="store", type="int", help="first page to be processed")
    rdc.add_option('-l',"--last", dest="last", action="store", type="int", help="last page to be processed")

    #parse the command line
    dParams, args = rdc.parseCommandLine()
    
    #Now we are back to the normal programmatic mode, we set the component parameters
    rdc.setParams(dParams)
    
    doc = rdc.loadDom()
    doc = rdc.run(doc)
    if doc is not None and rdc.getOutputFileName() != '-':
        rdc.writeDom(doc, bIndent=True) 
    
