# -*- coding: utf-8 -*-
"""


    layoutGenerator.py
    
    generate Layout annotated data 
    
    copyright Naver Labs 2017
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
    
    @author: H. Déjean
"""

from __future__ import unicode_literals

import libxml2
import sys, os.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))))

from dataGenerator.numericalGenerator import  integerGenerator
from dataGenerator.generator import Generator
from dataGenerator.layoutGenerator import layoutZoneGenerator
from dataGenerator.listGenerator import listGenerator


class doublePageGenerator(layoutZoneGenerator):
    """
        a double page generator
        allows for layout on two pages (table)!
        
        useless??? could be useles for many stuff. so identify when it is useful! 
            pros: better for simulating breakpages (white pages for chapter?)
                  see which constraints when in alistGenerator: need 2pages generated. the very first :white. when chapter break 
                  structure=[left, right]
                  
                  a way to handle mirrored structure !  (at layout or content: see bar for marginalia)
                  
            cons: simply useless!
            
        for page break: need to work with contentgenerator?
    """
    def __init__(self,h,w,m,r):
        layoutZoneGenerator.__init__(self)
        
        ml,mr= m
        self.leftPage  = pageGenerator(h,w,ml,r)
        self.leftPage.leftOrRight = 1 #left
        self.rightPage = pageGenerator(h,w,mr,r)
        self.rightPage.leftOrRight = 2 #right
        
        self._structure = [
                            ((self.leftPage,1,100),(self.rightPage,1,100),100)
                            ]
    
class pageGenerator(layoutZoneGenerator):
    ID=1
    def __init__(self,h,w,m,r):
        layoutZoneGenerator.__init__(self)
        self._label='PAGE'
        hm,hsd=  h
        self.pageHeight = integerGenerator(hm,hsd)
        self.pageHeight.setLabel('height')
        wm,wsd=  w
        self.pageWidth = integerGenerator(wm,wsd)
        self.pageWidth.setLabel('width')
        
        self.leftOrRight = None
        # WHITE SPACES
        self._pagenumber = None  # should come from documentGen.listGen(page)?
        self._margin = marginGenerator(*m)
        self._ruling = gridGenerator(*r)

        # need to build margin zones! (for text in margin)
        # should be replaced by a layoutZoneGenerator
        self._typeArea_x1 = None
        self._typeArea_y1 = None
        self._typeArea_x2 = None
        self._typeArea_y2 = None
        self._typeArea_h = None
        self._typeArea_w = None        

        #once generation is done
        self._lColumns = []

        # define 
        self._marginRegions=[]
        self._typeArea  = [ self._typeArea_x1 , self._typeArea_y1 , self._typeArea_x2 , self._typeArea_x2 , self._typeArea_h , self._typeArea_w ]
        
        
        self._structure = [
#                         (  (self.pageHeight,1,100),(self.pageWidth,1,100),100) 
                        (  (self.pageHeight,1,100),(self.pageWidth,1,100),(self._margin,1,100),(self._ruling,1,100),100) 

                          ]

    
    def getLeftMargin(self):pass
    def getRightMaring(self):pass
        
    def getColumns(self):
        """
            assume generation done
        """
        self._lColumns= self._ruling._generation[1:]
        return self._lColumns
    
    def computeAllValues(self,H,W,t,b,l,r):
        """
            from page dim and margins: compute typearea
        """
        self._typeArea_x1 = l
        self._typeArea_y1 = t
        self._typeArea_x2 = W - r
        self._typeArea_y2 = H - b
        self._typeArea_h  = H - t - b
        self._typeArea_w  = W - l - r
        self._typeArea = [ self._typeArea_x1 , self._typeArea_y1 , self._typeArea_x2 , self._typeArea_x2 , self._typeArea_h , self._typeArea_w]

        #define the 4 margins as layoutZone

    def generate(self):
        """
            bypass layoutZoneGen: specific to page
        """
        
        self._generation = []
        for obj in self._instance[:2]:
            obj.generate()
            self._generation.append(obj)        
        # see datGenerator: setValue
        # here cascade values(generation) for _structure elt
        if self._margin:
            self._margin.setPage(self)
            self._margin.generate()
            self._generation.append(self._margin)
        t,b,l,r = map(lambda x:x._generation,self._margin._generation)
#         
#         self.pageHeight.generate()
        pH = self.pageHeight._generation
#         self.pageWidth.generate()
        pW = self.pageWidth._generation
#         
        self.computeAllValues(pH,pW,t, b, l, r)
        
        if self._ruling:
            self._ruling.setPage(self)
            self._ruling.setTypeArea(self._typeArea)
            self._ruling.generate()
            self._generation.append(self._ruling)
        
class gridGenerator(Generator):
    """
        create a grid generator
        need to define here the (mu,sd) for the positions
    """
    def __init__(self,nbcolumns,gutter=(0,0)):
        Generator.__init__(self)
        cm,cs = nbcolumns
        self.nbcolumns = integerGenerator(cm, cs)
        self.nbcolumns.setLabel('nbCol')
        gm,gs = gutter
        self.gutter = integerGenerator(gm,gs)
        self.ColumnsListGen  = listGenerator(columnGenerator, self.nbcolumns ,None)
        self.ColumnsListGen.setLabel("GRIDCOL")
        nbCols=(9,0)
        nbRows=(15,5) 
        self.fullPageTable = tableGenerator(nbCols,nbRows)
        # required at line level!
        self.leading=None
        
        # kind of parent
        self._typeArea = None
        self._columnWidth = None
        self._structure = [
                ((self.ColumnsListGen,1,100),100)
#                 ((self.fullPageTable,1,100),100)

                 ]
    def setPage(self,p): self._page=p
    def getPage(self): return self._page
    
    def setTypeArea(self,TA):
        """
            not better to generate a layoutZoneObject for the TA?
        """
        self._typeArea= TA
        

    def generate(self):
        """
            simply generate nbcolumns and gutter
        """
        self._generation = []
        # need to use self._instance
        for obj in self._instance:
            if isinstance(obj,tableGenerator):
                self._generation = []
                x1,y1,x2,y2,h,w = self._typeArea
                obj.setPositionalGenerators((x1,0),(y1,0),(h,0),(w,0))
                obj.generate()
                self._generation.append(obj)
                 
            elif  isinstance(obj,listGenerator):
                nbCols =  self.ColumnsListGen.getValuedNb()
                self._columnWidth  = self._typeArea[5] / nbCols   #replace by a generator integerGenerator(self.TAW / nbCol,sd)??
                self._columnHeight = self._typeArea[4]
                
                x1,y1,x2,y2,h,w = self._typeArea
        
                self._generation = [self.nbcolumns]
                for i,colGen in enumerate(self.ColumnsListGen._instance):
        #             print i, colGen
                    colx = x1 + ( ( i * self._columnWidth) + 0)
                    coly = y1
                    colH = h
                    colW = self._columnWidth    
                    colGen.setPositionalGenerators((colx,5),(coly,5),(colH,5),(colW,5))
                    colGen.setGrid(self)       
                    colGen.setPage(self.getPage())
                    colGen.generate()
                    self._generation.append(colGen)
        
        
    def XMLDSFormatAnnotatedData(self,linfo,obj):
        self.domNode = libxml2.newNode(obj.getLabel())
        
        for info,tag in linfo:
            if isinstance(tag,Generator):
                node=tag.XMLDSFormatAnnotatedData(info,tag)
                self.domNode.addChild(node)
            else:
                self.domNode.setProp(tag,str(info))
        
        return self.domNode
            
    def noiseMerge(self):
        """
            merge horizontally aligned lines from different columns
            TH: ratio of lines merged.
        """
        
    def exportAnnotatedData(self,foo):
        self._GT=[]
#         print self._generation, self._instance
        for obj in self._generation:
            if type(obj._generation) == unicode:
                self._GT.append((obj._generation,obj.getLabel()))
            elif type(obj._generation) == int:
                self._GT.append((obj._generation,obj.getLabel()))
            else:        
                if obj is not None:
#                     print obj,obj.exportAnnotatedData([])
                    self._GT.append( (obj.exportAnnotatedData([]),obj))
        
        return self._GT 



        
class columnGenerator(layoutZoneGenerator):
    """
        a column generator
        requires a parent : x,y,h,w computed in the parent:
        
        see  CSS Box Model: margin,border, padding
        
    """
    def __init__(self,x=None,y=None,h=None,w=None):
        layoutZoneGenerator.__init__(self,x=x,y=y,h=h,w=w)
        self.setLabel("COLUMN")
        
        # lines or linegrid ??
        self.nbLines = integerGenerator(40, 5)
        self.nbLines.setLabel('nbLines')
        self.LineListGen  = listGenerator(LineGenerator, self.nbLines ,None)
        self.LineListGen.setLabel("colLine") 
        self._mygrid = None
        self.leading= 70
        
        # table  (+ caption)
        #self.fullPageTable = tableGenerator(nbCols,nbRows)
        
        # other elements? image+ caption
        self._structure = [
                            ((self.getX(),1,100),(self.getY(),1,100),(self.getHeight(),1,100),(self.getWidth(),1,100),(self.LineListGen,1,100),100)
                          ]
    
    def setGrid(self,g):self._mygrid = g
    def getGrid(self): return self._mygrid
    def setPage(self,p):self._page = p
    def getPage(self): return self._page    
    
    def generate(self):
        """
            prepare data for line
            nbLines: at instanciate() level?
            
        """
        # x,h, w,h
        self._generation = []
        for obj in self._instance[:-1]:
            obj.generate()
            self._generation.append(obj)
            
        for i,lineGen in enumerate(self.LineListGen._instance):
            # too many lines
            if (i * self.leading) + self.getY()._generation > (self.getY()._generation + self.getHeight()._generation):
                continue
            linex =self.getX()._generation
            liney = (i * self.leading) + self.getY()._generation
            lineH = 50
            lineW = self.getWidth()._generation   
            lineGen.setPage(self.getPage()) 
            lineGen.setPositionalGenerators((linex,5),(liney,5),(lineH,5),(lineW,5))
#             lineGen.setParent(self)        
            lineGen.generate()
            self._generation.append(lineGen)
    
class marginGenerator(Generator):
    """
        define margins:  top, bottom, left, right
            and also the print space coordinates
            
        restricted to 1?2-column grid max for the moment? 
    """
    def __init__(self,top,bottom,left, right):
        Generator.__init__(self)
        m,sd = top
        self._top= integerGenerator(m,sd)
        self._top.setLabel('top')
        m,sd = bottom
        self._bottom = integerGenerator(m,sd)
        self._bottom.setLabel('bottom')
        m,sd = left
        self._left = integerGenerator(m,sd)
        self._left.setLabel('left')
        m,sd = right
        self._right= integerGenerator(m,sd)
        self._right .setLabel('right')
        
        self._label='margin'
        
        
        self.leftMarginGen=layoutZoneGenerator()
        self.leftMarginGen.setLabel('leftMargin')
        self.rightMarginGen=layoutZoneGenerator()
        self.rightMarginGen.setLabel('rightMargin')
        self.topMarginGen=layoutZoneGenerator()
        self.topMarginGen.setLabel('topMargin')
        self.bottomMarginGen=layoutZoneGenerator()
        self.bottomMarginGen.setLabel('bottomMargin')
        
        
        self._structure = [ ((self._top,1,100),(self._bottom,1,100),(self._left,1,100),(self._right,1,100),100) ]
    
    def setPage(self,p):self._page=p 
    def getPage(self):return self._page
    
    def getDimensions(self): return self._top,self._bottom,self._left, self._right
        
    def getMarginZones(self):
        """
            return the 4 margins as layoutZone
        """
    def exportAnnotatedData(self,foo=None):
         
        self._GT=[]
        for obj in self._generation:
            if type(obj._generation) == unicode:
                self._GT.append((obj._generation,obj.getLabel()))
            elif type(obj._generation) == int:
                self._GT.append((obj._generation,obj.getLabel()))
            else:        
                if obj is not None:
#                     print obj,obj.exportAnnotatedData([])
                    self._GT.append( (obj.exportAnnotatedData([]),obj.getLabel()))
        
        return self._GT  

    def XMLDSFormatAnnotatedData(self,linfo,obj):
        self.domNode = libxml2.newNode(obj.getLabel())
         
        for info,tag in linfo:
            self.domNode.setProp(tag,str(info))
         
        return self.domNode



class marginaliaGenerator(layoutZoneGenerator):
    """
        marginalia Gen: assume relation with 'body' part
    """
    def __init__(self,x=None,y=None,h=None,w=None):
        layoutZoneGenerator.__init__(self,x=x,y=y,h=h,w=w)
        self.setLabel("MARGINALIA")
        
        #pointer to the parent structures!! line? page,?
        #lineGen!!
        
        self._structure = [
                            ((self.getX(),1,100),(self.getY(),1,100),(self.getHeight(),1,100),(self.getWidth(),1,100),100)
                            ]        
            
class LineGenerator(layoutZoneGenerator):
    """
        what is specific to a line? : content
            content points to a textGenerator
            
        
        for table noteGen must be positioned better!
            if parent= column
            if parent= cell
            if parent =...
            
    """ 
    def __init__(self,x=None,y=None,h=None,w=None):
        layoutZoneGenerator.__init__(self,x=x,y=y,h=h,w=w)
        self.setLabel("LINE")
        
        self._noteGen = marginaliaGenerator()
        self._justifixation = None # center, left, right, just, random
        
        self.bSkew=None  # (angle,std)
        
        self._structure = [
                            ((self.getX(),1,100),(self.getY(),1,100),(self.getHeight(),1,100),(self.getWidth(),1,100),(self._noteGen,1,010),100)
#                             ((self.getX(),1,100),(self.getY(),1,100),(self.getHeight(),1,100),(self.getWidth(),1,100),100)

                            ]    
    def setPage(self,p):self._page=p
    def getPage(self): return self._page
    def generate(self):
        """
        need a pointer to the column to select the possible margins
        need a pointer to the margin to add the element
        """
        self._generation = []
        for obj in self._instance[:4]:
            obj.generate()
            self._generation.append(obj)

        #if marginalia
        if len(self._instance) == 5:
            # left or right margin
            # need to go up to the grid to know where the column is
            print self.getPage().leftOrRight
            if self.getPage().leftOrRight == 1: 
                # get left margin
                myregion= self.getPage().getLeftMargin()
                #left page: put on the left margin, right otherwise? 
                marginaliax = 10
            else:
                marginaliax = 600 
            marginaliay = self.getY()._generation
            marginaliaH = 50
            marginaliaW = 50   
            self._noteGen.setPositionalGenerators((marginaliax,5),(marginaliay,5),(marginaliaH,5),(marginaliaW,5))
            self._noteGen.generate()
            self._generation.append(self._noteGen)        
        return self
    
class cellGenerator(layoutZoneGenerator):
    """
        cellGen
    """ 
    def __init__(self,x=None,y=None,h=None,w=None):
        layoutZoneGenerator.__init__(self,x=x,y=y,h=h,w=w)
        self.setLabel("CELL")
        self.leading = 25
        self.nbLines = integerGenerator(2, 1)
        self._LineListGen = listGenerator(LineGenerator, self.nbLines ,None)
        self._LineListGen.setLabel("cellline")
        self._structure =[((self.getX(),1,100),(self.getY(),1,100),(self.getHeight(),1,100),(self.getWidth(),1,100),(self._LineListGen,1,100),100)]
        
    def generate(self):
        self._generation=[]
        for obj in self._instance[:-1]:
            obj.generate()
            self._generation.append(obj)
#         print self.getLabel(),self._generation
        self._LineListGen.instantiate()
        for i,lineGen in enumerate(self._LineListGen._instance):
            # too many lines
            if (i * self.leading) + self.getY()._generation > (self.getY()._generation + self.getHeight()._generation):
                continue
            linex = self.getX()._generation
            liney = (i * self.leading) + self.getY()._generation
            lineH = 20
            lineW = self.getWidth()._generation    
            lineGen.setPositionalGenerators((linex,5),(liney,5),(lineH,5),(lineW,5))
#             lineGen.setParent(self)        
            lineGen.generate()
            lineGen.setLabel('LINE')
            self._generation.append(lineGen)
        return self    

class tableGenerator(layoutZoneGenerator):
    """
        a table generator
    """   
    def __init__(self,nbCols,nbRows):
        layoutZoneGenerator.__init__(self)
        self.setLabel('TABLE')
        self.nbCols = integerGenerator(nbCols[0],nbCols[1])
        self.nbCols.setLabel('nbCols')
        self.nbRows = integerGenerator(nbRows[0],nbRows[1])
        self.nbRows.setLabel('nbRows')
        
        self._lRowsGen = listGenerator(layoutZoneGenerator, self.nbRows ,None)
        self._lRowsGen.setLabel("row")
        self._lColumnsGen = listGenerator(layoutZoneGenerator, self.nbCols ,None)
        self._lColumnsGen.setLabel("col")
        self._structure = [
            ((self.getX(),1,100),(self.getY(),1,100),(self.getHeight(),1,100),(self.getWidth(),1,100),
             (self.nbCols,1,100),(self.nbRows,1,100),
             (self._lColumnsGen,1,100),(self._lRowsGen,1,100),100)
            ]
        
    def generate(self):
        """
            generate the rows, the columns, and then the cells
        """
        self._generation = []
        for obj in self._instance[:-2]:
            obj.generate()
            self._generation.append(obj)
                    
        nbCols =  len(self._lColumnsGen._instance)
        nbRows= len(self._lRowsGen._instance)
        self._columnWidth  = int(round(self.getWidth()._generation / nbCols))
        self._rowHeight = int(round(self.getHeight()._generation / nbRows))
        
        self.lCols=[]
        self.lRows=[]
        for i,colGen in enumerate(self._lColumnsGen._instance):
#             print i, colGen
            colx = self.getX()._generation + ( i * self._columnWidth)
            coly = self.getY()._generation
            colH = self.getHeight()._generation 
            colW = self._columnWidth    
            colGen.setNumber(i)
            colGen.setPositionalGenerators((colx,5),(coly,5),(colH,5),(colW,5))
#             colGen.setGrid(self)       
            colGen.setLabel("COL")
            colGen.generate()
            self._generation.append(colGen)
            self.lCols.append(colGen)
            
#         print self._lRowsGen._instance,len(self._lRowsGen._instance),self._lRowsGen.getValuedNb() ,self._rowHeight , self.getHeight()._generation , nbRows
        for i,rowGen in enumerate(self._lRowsGen._instance):
            rowx = self.getX()._generation 
            # here   genertor for height variation!
            rowy = self.getY()._generation + (i * self._rowHeight)
            rowH = self._rowHeight
            rowW = self.getWidth()._generation
            rowGen.setLabel("ROW")
            rowGen.setNumber(i)
            rowGen.setPositionalGenerators((rowx,5),(rowy,5),(rowH,5),(rowW,5))
            rowGen.generate()
#             print i, rowy, self.getHeight()._generation
            self.lRows.append(rowGen)
            self._generation.append(rowGen)            
            
            
        ## table specific stuff
        ## table headers, stub,....
        ## creation of the cells; then content in the cells
        self.lCellGen=[]
        for col in  self.lCols:
            for row in self.lRows:
                cell=cellGenerator()
                cell.setLabel("CELL")
                cell.setPositionalGenerators((col.getX()._generation,0),(row.getY()._generation,0),(row.getHeight()._generation,0),(col.getWidth()._generation,0))
                self.lCellGen.append(cell)
                cell.instantiate()
                cell.generate()
                self._generation.append(cell)
        
            
class documentGenerator(Generator):
    """
        a document generator

        need to store the full list of parameters for sub elements
        
        in json!!
        
        pageNumber
        pageHeight
        pageWidth
        marginTop
        marginBottom
        marginLeft
        marginRight
        
        gridnbCol
            
            listObjects (structure!): weight for the potential objects
            colnbLines    ## replaced by a structure corresponding to a content stream (header/paragraph. table;...)
                lineleading 
                lineskew
                lineUnderline
                lineHjustification
                lineVjustification
                ...
            tableWidth
            tableHeight
            tablenbCol
            tablenbRow
            tableCellNbLines (similar to columnlines) 
                celllineleading  ...
                lineHjustification
                lineVjustification



        levels between document and page/double-page: usefull?
    """
    def __init__(self,tpageH,tpageW,tnbpages,tMargin=None,tRuling=None):
        
        Generator.__init__(self)
        self._name = 'DOC'

        # missing elements:
        self._isCropped = False  # cropped pages
        self._hasBackground = False # background is visible
        
        #how to serialize the four cover pages?
        #in instantiate (or generation?): put the 3-4cover at the end?
        self.lCoverPages = None
        self._hasBinding = False  # binding image
        
        self.scanner = None ## the way the pages are "scanned"
        # 1: single page ; 2 double page 3 cropped 4 _twoPageImage  # background
        
        # portion of the other page visible
        self._twoPageImage= False # two pages in one image 
        
        self._nbpages = integerGenerator(tnbpages[0],tnbpages[1])
        self._nbpages.setLabel('nbpages')

        self._margin = tMargin
        self._ruling = tRuling      
        
        
        self.pageListGen = listGenerator(pageGenerator,self._nbpages,tpageH,tpageW,self._margin,self._ruling)
        self.pageListGen.setLabel('pages')
        self._structure = [
                            #firstSofcover (first and second)
                            ((self.pageListGen,1,100),100) 
                            #lastofcover (if fistofcover??)
                            ]
    
    def XMLDSFormatAnnotatedData(self,gtdata):
        """
            convert into XMLDSDformat
            (write also PageXMLFormatAnnotatedData! )
        """
    
        self.docDom = libxml2.newDoc("1.0")
        root  = libxml2.newNode("DOCUMENT")
        self.docDom.setRootElement(root)
        for info,page in gtdata:
            pageNode = page.XMLDSFormatAnnotatedData(info,page)
            root.addChild(pageNode)
        return self.docDom
        
        
    def generate(self):
        self._generation = []
        
        ## 1-2 cover pages
        
        for i,pageGen in enumerate(self.pageListGen._instance):
            #if double page: start with 1 = right?
            pageGen.generate()
            self._generation.append(pageGen)
    
        ## 3-4 covcer pages
        return self
    
    def exportAnnotatedData(self,foo):
        """
            build a full version of generation: integration of the subparts (subtree)
            
            what are the GT annotation for document?  
             
        """
        ## export (generated value, label) for terminal 

        self._GT=[]
        for obj in self._generation:
            self._GT.append((obj.exportAnnotatedData([]),obj))
        
        return self._GT
    
class DocMirroredPages(documentGenerator):
    def __init__(self,tpageH,tpageW,tnbpages,tMargin=None,tRuling=None):
        documentGenerator.__init__(self,tpageH,tpageW,tnbpages,tMargin,tRuling)
        self._lmargin, self._rmargin = tMargin
        self._ruling= tRuling
        self.pageListGen = listGenerator(doublePageGenerator,self._nbpages,tpageH,tpageW,(self._lmargin,self._rmargin),self._ruling)
        self.pageListGen.setLabel('pages')
        self._structure = [
                            #firstSofcover (first and second)
                            ((self.pageListGen,1,100),100) 
                            #lastofcover (if fistofcover??)
                            ]
        
    def generate(self):
        self._generation = []
        
        for i,pageGen in enumerate(self.pageListGen._instance):
            pageGen.generate()
            self._generation.append(pageGen)
        return self  
    
class paragraphGenerator(Generator):
    """
        needed: number of lines: no not at this level: simply 'length' and typographical specification
        no 'positional' info at this level: one stream_wise objects
        
        need to be recursive ==> treeGenerator (ala listGenerator!=
    """
    LEFT      = 1
    RIGHT     = 2
    CENTER    = 3
    JUSTIFIED = 4
    
    def __init__(self,nblinesGen, length):
        Generator.__init__(self)
        
        self.alignment = None
        self.firstLineIndent = None
        self.spaceBefore = None
        self.spaceAfter = None
        
        self.nblines = nblinesGen
        self.firstLine = LineGenerator()
        self.lines = LineGenerator()
        self._structure = [
                (  (self.firstLine,1,100),(self.lines,self.nblines,100),100) 
                    ]

class content(Generator):
    """
        stream_wise content : sequence of par/heders, , tables,...
        
        json profile: kind of css : css for headers, content, listitems,...
    """
    def __init__(self,nbElt,lenElt):
        Generator.__init__(self)
        
        mElt,sElt = nbElt
        self._nbElts = integerGenerator(mElt, sElt)
        mlenElt,slenElt = lenElt
        self._lenElt = integerGenerator(mlenElt, slenElt)        
        self.contentObjectListGen = listGenerator(paragraphGenerator,self._nbElts,self._lenElt)

def docm():
    tlMarginGen = ((100, 10),(100, 10),(150, 10),(50, 10))
    trMarginGen = ((100, 10),(100, 10),(50, 10),(150, 10))

    tGrid = ( (1,0),(0,0) )
    mydoc = DocMirroredPages((1200, 10),(700, 10),(10, 0),tMargin=(tlMarginGen,trMarginGen),tRuling=tGrid)
    mydoc.instantiate()
    mydoc.generate()
    gt =  mydoc.exportAnnotatedData(())
    print gt
    docDom = mydoc.XMLDSFormatAnnotatedData(gt)
    print docDom.serialize("utf-8", True)
    docDom.saveFileEnc("tmp.ds_xml",'utf-8')        

if __name__ == "__main__":

    docm()
#     tMarginGen = ((100, 10),(100, 10),(50, 10),(50, 10))
#     tGrid = ( (3,0),(0,0) )
#     mydoc = documentGenerator((1200, 10),(1500, 10),(100, 0),tMarginGen,tGrid)
#     mydoc.instantiate()
#     mydoc.generate()
#     gt =  mydoc.exportAnnotatedData(())
#     docDom = mydoc.XMLDSFormatAnnotatedData(gt)
# #     print docDom.serialize("utf-8", True)
#     docDom.saveFileEnc("tmp.ds_xml",'utf-8')
    
#     contGen=content((5,2),(10,3))
#     contGen.instantiate()
#     contGen.generate()
#     
#     print 'done!'


    
    