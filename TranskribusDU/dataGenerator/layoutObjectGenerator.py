# -*- coding: utf-8 -*-
"""


    layoutGenerator.py
    
    generate Layout annotated data 
    
    copyright Naver Labs 2017
    READ project 


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
    @author: H. Déjean
"""





try:basestring
except NameError:basestring = str

from lxml import etree
import sys, os.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))

from dataGenerator.numericalGenerator import numericalGenerator
from dataGenerator.numericalGenerator import  integerGenerator
from dataGenerator.generator import Generator
from dataGenerator.layoutGenerator import layoutZoneGenerator
from dataGenerator.listGenerator import listGenerator
# from booleanGenerator import booleanGenerator

class doublePageGenerator(layoutZoneGenerator):
    """
        a double page generator
        allows for layout on two pages (table)!
        
        useless??? could be useless for many stuff. so identify when it is useful! 
            pros: better for simulating breakpages (white pages for chapter?)
                  see which constraints when in alistGenerator: need 2pages generated. the very first :white. when chapter break 
                  structure=[left, right]
                  
                  a way to handle mirrored structure !  (at layout or content: see bar for marginalia)
                  
            cons: simply useless!
            
        for page break: need to work with content generator?
    """
    def __init__(self,h,w,m,r,config=None):
        layoutZoneGenerator.__init__(self)
        
        ml,mr= m
        self.leftPage  = pageGenerator(h,w,ml,r,config)
        self.leftPage.leftOrRight = 1 #left
        self.rightPage = pageGenerator(h,w,mr,r,config)
        self.rightPage.leftOrRight = 2 #right
        
        self._structure = [
                            ((self.leftPage,1,100),(self.rightPage,1,100),100)
                            ]
    
    
class pageGenerator(layoutZoneGenerator):
    """
     need to add background zone 
    """
    ID=1
    def __init__(self,h,w,m,r,dConfig):
        layoutZoneGenerator.__init__(self)
        self._label='PAGE'
        hm,hsd=  h
        self.pageHeight = integerGenerator(hm,hsd)
        self.pageHeight.setLabel('height')
        wm,wsd=  w
        self.pageWidth = integerGenerator(wm,wsd)
        self.pageWidth.setLabel('width')

        self.myConfig=dConfig
        ##background 

        ##also need X0 and y0 
        self._x0 = 0
        self._y0 = 0
        
        (gridType,(cm,cs),(gm,gs)) = r
        assert gridType == 'regular'
        
        self.nbcolumns = integerGenerator(cm, cs)
        self.nbcolumns.setLabel('nbCol')
        self.gutter = integerGenerator(gm,gs)
        self.ColumnsListGen  = listGenerator(columnGenerator, self.nbcolumns ,None)
        self.ColumnsListGen.setLabel("GRIDCOL")
        
        # required at line level!
        self.leading=None        
        
        
        self.leftOrRight = None
        # WHITE SPACES
        self.pageNumber = None  # should come from documentGen.listGen(page)?
        if self.myConfig['pnum']:
            self.pageNumber = pageNumberGenerator()
        self._margin = marginGenerator(*m)
#         self._ruling = gridGenerator(*r)

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
        self._marginRegions = []
        self._typeArea  = [ self._typeArea_x1 , self._typeArea_y1 , self._typeArea_x2 , self._typeArea_x2 , self._typeArea_h , self._typeArea_w ]
        

        mystruct =  [ (self.pageHeight,1,100),(self.pageWidth,1,100)]

        if self.pageNumber is not None:
            mystruct.append((self.pageNumber,1,100))
            
        mystruct.append((self._margin,1,100))
        mystruct.append((self.ColumnsListGen,1,100))
                             
        mystruct.append(100)
        self._structure = [
                        mystruct
                          ]

    def getLeftMargin(self): return self._marginRegions[2]
    
    def getRightMargin(self):return self._marginRegions[3]
        
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
        
        self._marginRegions = [(self._x0,self._y0,self._typeArea_y1,self.pageWidth._generation), #top
                               (self._x0,H - b,b,self.pageWidth._generation), #bottom
                               (self._x0,self._y0,self.pageHeight._generation,l), #left 
                               (W - r,self._y0,self.pageHeight._generation,r)  #right
                               
                               ]
        self._typeArea = [ self._typeArea_x1 , self._typeArea_y1 , self._typeArea_x2 , self._typeArea_x2 , self._typeArea_h , self._typeArea_w]

        #define the 4 margins as layoutZone

    def addPageNumber(self,p):
        """
        """
        zoneIndx = self.myConfig['pnumZone']
        region = self._marginRegions[zoneIndx]
        
        # in the middle of the zone
        p.setPositionalGenerators((region[0]+region[3]*0.5,5),(region[1]+region[2]*0,5),(10,0),(10,1))
        
    def generate(self):
        """
            bypass layoutZoneGen: specific to page
        """
#         self.setNumber(self.getParent().getNumber())
        self.setNumber(1)
        self._generation = []
        for obj in self._instance[:2]:
            obj.generate()
            self._generation.append(obj)        
        
            
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

        ## margin elements: page numbers
        if self.pageNumber is  not None:
            self.addPageNumber(self.pageNumber)
            self.pageNumber.generate()                
            self._generation.append(self.pageNumber)
            
            
        obj = self._instance[-1]    
        nbCols =  self.ColumnsListGen.getValuedNb()
        self._columnWidth  = self._typeArea[5] / nbCols   #replace by a generator integerGenerator(self.TAW / nbCol,sd)??
        self._columnHeight = self._typeArea[4]
        
        x1,y1,x2,y2,h,w = self._typeArea

        self._generation.append(self.nbcolumns)
        for i,colGen in enumerate(self.ColumnsListGen._instance):
#             print i, colGen
            colx = x1 + ( ( i * self._columnWidth) + 0)
            coly = y1
            colH = h
            colW = self._columnWidth
            colGen.setPositionalGenerators((colx,5),(coly,5),(colH,5),(colW,5))
            colGen.setGrid(self)       
            colGen.setPage(self)
            try:content=self.myConfig['colStruct'][0](*self.myConfig['colStruct'][-1])
            except KeyError as e: content=None
            if content is not None:
                colGen.updateStructure((content,1,100))
                colGen.instantiate()
                colGen.generate()
                self._generation.append(colGen)            
            
            
        #how generate page content
    

    def PageXmlFormatAnnotatedData(self, linfo, obj):
        """
            PageXml format 
        """
        self.domNode = etree.Element(obj.getLabel())
        if obj.getNumber() is not None:
            self.domNode.set('number',str(obj.getNumber()))   
        for info,tag in linfo:
            if isinstance(tag,Generator):
                node=tag.PageXmlFormatAnnotatedData(info,tag)
                self.domNode.append(node)
            else:
                self.domNode.set(tag,str(info))
        
        return self.domNode
        
    def XMLDSFormatAnnotatedData(self, linfo, obj):
        """
            how to store GT info: need to respect DS format!
            PAGE + margin info
        """
        self.domNode = etree.Element(obj.getLabel())
        if obj.getNumber() is not None:
            self.domNode.set('number',str(obj.getNumber()))    
        for info,tag in linfo:
            if isinstance(tag,Generator):
                node=tag.XMLDSFormatAnnotatedData(info,tag)
                self.domNode.append(node)
            else:
                self.domNode.set(tag,str(info))
        
        return self.domNode


class columnGenerator(layoutZoneGenerator):
    """
        a column generator
        requires a parent : x,y,h,w computed in the parent:
        
        see  CSS Box Model: margin,border, padding
        
    """
    def __init__(self,x=None,y=None,h=None,w=None):
        layoutZoneGenerator.__init__(self,x=x,y=y,h=h,w=w)
        self.setLabel("COLUMN")
        
#         # lines or linegrid ??
#         self.nbLines = integerGenerator(40, 5)
#         self.nbLines.setLabel('nbLines')
#         self.LineListGen  = listGenerator(LineGenerator, self.nbLines ,None)
#         self.LineListGen.setLabel("colLine") 
#         self._mygrid = None
#         self.leading= 12
        
        
        # table  (+ caption)
        #self.fullPageTable = tableGenerator(nbCols,nbRows)
        
        # other elements? image+ caption
        self._structure = [
#                             [(self.getX(),1,100),(self.getY(),1,100),(self.getHeight(),1,100),(self.getWidth(),1,100),(self.LineListGen,1,100),100]
                            [(self.getX(),1,100),(self.getY(),1,100),(self.getHeight(),1,100),(self.getWidth(),1,100),100]

                          ]
    
    def setGrid(self,g):self._mygrid = g
    def getGrid(self): return self._mygrid
#     def setPage(self,p):self._page = p
#     def getPage(self): return self._page    
    
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
        
        colContent =  self._instance[-1]
        if isinstance(colContent,tableGenerator):
            x1,y1,h,w = self.getX()._generation,self.getY()._generation,self.getHeight()._generation,self.getWidth()._generation
            colContent.setPositionalGenerators((x1,0),(y1,0),(h,0),(w,0))
            colContent.setPage(self.getPage())
            colContent.generate()
            self._generation.append(colContent)            
            
        elif isinstance(colContent,listGenerator):
            
            for i,lineGen in enumerate(colContent._instance):
                # too many lines
                if (i * self.leading) + self.getY()._generation > (self.getY()._generation + self.getHeight()._generation):
                    continue
                linex =self.getX()._generation
                liney = (i * self.leading) + self.getY()._generation
                lineH = 10
                lineW = self.getWidth()._generation   
                lineGen.setParent(self)
                lineGen.setPage(self.getPage()) 
                lineGen.setPositionalGenerators((linex,2),(liney,2),(lineH,2),(lineW,2))
    #             lineGen.setParent(self)        
                lineGen.generate()
                self._generation.append(lineGen)
    
        
    
class pageNumberGenerator(layoutZoneGenerator):
    """
        a pagenumgen
    """
    def __init__(self,x=None,y=None,h=None,w=None):
        layoutZoneGenerator.__init__(self,x=x,y=y,h=h,w=w)
        self._label='LINE'
        
    
    def XMLDSFormatAnnotatedData(self, linfo, obj):
        self.domNode = etree.Element(obj.getLabel())
        self.domNode.set('pagenumber','yes')
        self.domNode.set('type','RO')       
        for info,tag in linfo:
            if isinstance(tag,Generator):
                node=tag.XMLDSFormatAnnotatedData(info,tag)
                self.domNode.append(node)
            else:
                self.domNode.set(tag,str(info))
        
        return self.domNode        
        
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
            if isinstance(obj._generation,basestring):
                self._GT.append((obj._generation,obj.getLabel()))
            elif type(obj._generation) == int:
                self._GT.append((obj._generation,obj.getLabel()))
            else:        
                if obj is not None:
#                     print obj,obj.exportAnnotatedData([])
                    self._GT.append( (obj.exportAnnotatedData([]),obj.getLabel()))
        
        return self._GT  

    def PageXmlFormatAnnotatedData(self,linfo,obj):
        
        self.domNode = etree.Element(obj.getLabel())
         
        for info,tag in linfo:
            self.domNode.set(tag,str(info))
         
        return self.domNode
        
    def XMLDSFormatAnnotatedData(self,linfo,obj):
        
        self.domNode = etree.Element(obj.getLabel())
         
        for info,tag in linfo:
            self.domNode.set(tag,str(info))
         
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
        
        self.BIES = 'RO'
        self._noteGen = marginaliaGenerator()
        self._justifixationGen = None #justificationGenerator() # center, left, right, just, random
        
        self.bSkew=None  # (angle,std)
        
        self._structure = [
#                             ((self.getX(),1,100),(self.getY(),1,100),(self.getHeight(),1,100),(self.getWidth(),1,100),(self._noteGen,1,010),100)
                            ((self.getX(),1,100),(self.getY(),1,100),(self.getHeight(),1,100),(self.getWidth(),1,100),100)

                            ]    
    def setPage(self,p):self._page=p
    def getPage(self): return self._page
    
    def computeBIES(self,pos,nbLines):
        
        if   nbLines == 1        : self.BIES='RS'
        elif pos     == 0        : self.BIES='RB'
        elif pos     == nbLines-1: self.BIES='RE'
        else                     : self.BIES='RI'
        
    def generate(self):
        """
        need a pointer to the column to select the possible margins
        need a pointer to the margin to add the element  if any
        
        need to be able to position vertically and horizontally
        
        """
        self._generation = []
        for obj in self._instance[:4]:
            obj.generate()
            self._generation.append(obj)

        #if marginalia
        if len(self._instance) == 5:
            # left or right margin
            # need to go up to the grid to know where the column is
            if self.getPage().leftOrRight == 1: 
                # get left margin
                myregion= self.getPage().getLeftMargin()
#                 print myregion
                #left page: put on the left margin, right otherwise? 
                marginaliax = myregion[0]+10
            else:
                #marginaliax = 600 
                myregion= self.getPage().getRightMargin()
                marginaliax = myregion[0]+10
                
            marginaliay = self.getY()._generation
            marginaliaH = 50
            marginaliaW = 50   
            # compute position according to the justifiaction : need parent, 
            self._noteGen.setPositionalGenerators((marginaliax,5),(marginaliay,5),(marginaliaH,5),(marginaliaW,5))
            self._noteGen.generate()
            self._generation.append(self._noteGen)        
        return self
    
    def XMLDSFormatAnnotatedData(self,linfo,obj):
        self.domNode = etree.Element(obj.getLabel())
        # for listed elements
        self.domNode.set('type',str(self.BIES))        

        for info,tag in linfo:
            if isinstance(tag,Generator):
                self.domNode.append(tag.XMLDSFormatAnnotatedData(info,tag))
            else:
                self.domNode.set(tag,str(info))
        
        return self.domNode
    
class cellGenerator(layoutZoneGenerator):
    """
        cellGen
        
        for the set of lines: define at this level the horizontal and vertical justification
        
        
        similar to column? a cell containts a grid???
            for instance: padding as well
        
        
    """ 
    def __init__(self,x=None,y=None,h=None,w=None):
        layoutZoneGenerator.__init__(self,x=x,y=y,h=h,w=w)
        self.setLabel("CELL")
        
        self._index = None
#         self.VJustification = booleanGenerator(0.1)
#         self.VJustification.setLabel('VJustification')
#         self.HJustification = integerGenerator(3, 1)
        self.leading = integerGenerator(20, 1)
        self.leading.setLabel('leading')
        self.nbLines = integerGenerator(5, 3)
        self._LineListGen = listGenerator(LineGenerator, self.nbLines ,None)
        self._LineListGen.setLabel("cellline")
        self._structure =[((self.getX(),1,100),(self.getY(),1,100),(self.getHeight(),1,100),(self.getWidth(),1,100),
                           (self.leading,1,100),
#                            (self.VJustification,1,100),
                           (self._LineListGen,1,100),100)]
    
    
    def getIndex(self): return self._index
    def setIndex(self,i,j): self._index=(i,j)
    
    def computeYStart(self,bVJustification,blockH):
        """
            compute where to start 'writing' according to justification and number of lines (height of the block)
        """
        if bVJustification:
            return  (0.5 * self.getHeight()._generation)  -  (0.5 * blockH)
        else:
            return 0
        
    def generate(self):
        self._generation=[]
        for obj in self._instance[:-1]:
            obj.generate()
            self._generation.append(obj)
#         print self.getLabel(),self._generation
        self._LineListGen.instantiate()
        
        # vertical justification : find the y start
#         ystart=self.computeYStart(self.VJustification._generation, self._LineListGen.getValuedNb()*self.leading._generation)
        ystart=self.computeYStart(False, self._LineListGen.getValuedNb()*self.leading._generation)
        xstart = self.getWidth()._generation * 0.25
        rowPaddingGen = numericalGenerator(10,2)
        rowPaddingGen.generate()
        
        lineH = 15
        nexty= ystart +  self.getY()._generation + rowPaddingGen._generation
        lLines=[]
        for i,lineGen in enumerate(self._LineListGen._instance):
            # too many lines
#             if (i * self.leading._generation) + (self.getY()._generation + lineH) > (self.getY()._generation + self.getHeight()._generation):
            if nexty +lineH >  (self.getY()._generation + self.getHeight()._generation):
                continue
            ## centered by default?
            linex = self.getX()._generation + (xstart)
            liney = nexty
            lineW = self.getWidth()._generation    
            lineGen.setPositionalGenerators((linex,5),(liney,5),(lineH,5),(lineW * 0.5,lineW * 0.1))
#             lineGen.setPositionalGenerators((linex,0),(liney,0),(lineH,0),(lineW * 0.5,lineW * 0.1))
            lineGen.setPage(self.getPage())  
            lineGen.setParent(self)
            lLines.append(lineGen)
            lineGen.generate()
            rowPaddingGen.generate()
            nexty= lineGen.getY()._generation + lineGen.getHeight()._generation+ rowPaddingGen._generation
            lineGen.setLabel('LINE')
            self._generation.append(lineGen)
        
        nbLines=len(lLines)
        for i,lineGen in enumerate(lLines):
            lineGen.computeBIES(i,nbLines)
        return self    

    def XMLDSFormatAnnotatedData(self,linfo,obj):
        self.domNode = etree.Element(obj.getLabel())
        # for listed elements
        self.domNode.set('row',str(self.getIndex()[0]))        
        self.domNode.set('col',str(self.getIndex()[1]))        

        for info,tag in linfo:
            if isinstance(tag,Generator):
                self.domNode.append(tag.XMLDSFormatAnnotatedData(info,tag))
            else:
                self.domNode.set(tag,str(info))
        
        return self.domNode


class tableGenerator(layoutZoneGenerator):
    """
        a table generator
        
        "padding" between two rows  (either a line and smal padding, or a larger space)
        idem for columns 
        
        
        either: use number of  rows/columns
                or rows/column height/width  (or constraint = allthesamevalue)
        
    """   
    def __init__(self,nbCols,nbRows):
        layoutZoneGenerator.__init__(self)
        self.setLabel('TABLE')
        self.nbCols = integerGenerator(nbCols[0],nbCols[1])
        self.nbCols.setLabel('nbCols')
        self.nbRows = integerGenerator(nbRows[0],nbRows[1])
        self.nbRows.setLabel('nbRows')
        
        self._bSameRowHeight=True
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
        if nbRows == 0: 
            return 
        if nbCols == 0:
            return 
        self._columnWidthM  = int(round(self.getWidth()._generation / nbCols))
        self._columnWidthG = numericalGenerator(self._columnWidthM, self._columnWidthM*0.2)

        self._rowHeightM = int(round(self.getHeight()._generation / nbRows))
        self._rowHeightG = numericalGenerator(self._rowHeightM,self._rowHeightM*0.25)
        
        self.lCols=[]
        self.lRows=[]
        nextx= self.getX()._generation
        
        for i,colGen in enumerate(self._lColumnsGen._instance):
            if nextx > self.getX()._generation + self.getWidth()._generation:
                continue
            self._columnWidthG.generate()
            
            colx = nextx #self.getX()._generation + ( i * self._columnWidth)
            coly = self.getY()._generation
            colH = self.getHeight()._generation 
            colW = self._columnWidthG._generation
            colGen.setNumber(i)
            colGen.setPositionalGenerators((colx,5),(coly,5),(colH,5),(colW,5))
#             colGen.setGrid(self)       
            colGen.setLabel("COL")
            colGen.setPage(self.getPage())
            colGen.generate()
            nextx= colGen.getX()._generation + colGen.getWidth()._generation
            self._generation.append(colGen)
            self.lCols.append(colGen)
            
        ## here 
        
        rowH = None
        nexty = self.getY()._generation
        for i,rowGen in enumerate(self._lRowsGen._instance):
            if nexty > self.getHeight()._generation + self.getY()._generation:
                continue
            rowx = self.getX()._generation 
            # here   generator for height variation!
            if self._bSameRowHeight:
                if rowH is None:
                    self._rowHeightG.generate() 
                    rowH = self._rowHeightG._generation
            else:
                self._rowHeightG.generate() 
                rowH = self._rowHeightG._generation
            rowy = nexty 
            # here test that that there is anough space for the row!!
#             print self._rowHeightM, self._rowHeightG._generation
            rowW = self.getWidth()._generation
            rowGen.setLabel("ROW")
            rowGen.setNumber(i)
            rowGen.setPage(self.getPage())
            rowGen.setPositionalGenerators((rowx,1),(rowy,1),(rowH,1),(rowW,1))
            rowGen.generate()
            nexty = rowGen.getY()._generation + rowGen.getHeight()._generation 
#             print i, rowy, self.getHeight()._generation
            self.lRows.append(rowGen)
            self._generation.append(rowGen)            
            
        ## table specific stuff
        ## table headers, stub,....


        #### assume the grid col generated?
        ### introduce a hierarchical column
        #### split the col into N subcols  :: what is tricky: headers  split the first         
        ### hierarchical row: for this needs big rows and split
        
        ## creation of the cells; then content in the cells
        self.lCellGen=[]
        for icol,col in enumerate(self.lCols):
            for irow, row in enumerate(self.lRows):
                cell=cellGenerator()
                cell.setLabel("CELL")
                cell.setPositionalGenerators((col.getX()._generation,0),(row.getY()._generation,0),(row.getHeight()._generation,0),(col.getWidth()._generation,0))
                self.lCellGen.append(cell)
                cell.instantiate()
                cell.setPage(self.getPage())
                cell.generate()
                cell.setIndex(irow,icol)
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
    def __init__(self,dConfig,tpageH,tpageW,tnbpages,tMargin=None,tRuling=None):
        
        Generator.__init__(self)
        self._name = 'DOC'

        self.myConfig = dConfig
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
    
    
    def PageXmlFormatAnnotatedData(self,gtdata):
        """
            convert into PageXMl
        """
        root  = etree.Element("PcGts")
        self.docDom = etree.ElementTree(root)
#         self.docDom.setRootElement(root)
        for info,page in gtdata:
            pageNode = page.XMLDSFormatAnnotatedData(info,page)
            root.append(pageNode)
        return self.docDom
        
        
        
    def XMLDSFormatAnnotatedData(self,gtdata):
        """
            convert into XMLDSDformat
            (write also PageXMLFormatAnnotatedData! )
        """
        root  = etree.Element("DOCUMENT")
        self.docDom = etree.ElementTree(root)
#         self.docDom.setRootElement(root)
        metadata= etree.Element("METADATA")
        root.append(metadata)
        metadata.text = str(self.myConfig)
        for info,page in gtdata:
            pageNode = page.XMLDSFormatAnnotatedData(info,page)
            root.append(pageNode)
        return self.docDom
        
        
    def generate(self):
        self._generation = []
        
        ## 1-2 cover pages
        
        for i,pageGen in enumerate(self.pageListGen._instance):
            #if double page: start with 1 = right?
            pageGen.myConfig= self.myConfig
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
#     def __init__(self,tpageH,tpageW,tnbpages,tMargin=None,tRuling=None):
    def __init__(self,dConfig):
        
#         scanning = dConfig['scanning']
        tpageH = dConfig['pageH']
        tpageW = dConfig['pageW']
        tnbpages = dConfig['nbPages']
        tMargin = (dConfig['lmargin'],dConfig['rmargin'])
        tRuling = dConfig['grid']
        
        
        documentGenerator.__init__(self,dConfig,tpageH,tpageW,tnbpages,tMargin,tRuling)
        self.myConfig = dConfig
        self._lmargin, self._rmargin = tMargin
        self._ruling= tRuling
        self.pageListGen = listGenerator(doublePageGenerator,self._nbpages,tpageH,tpageW,(self._lmargin,self._rmargin),self._ruling,dConfig)

        self.pageListGen.setLabel('pages')
        self._structure = [
                            #firstSofcover (first and second)
                            ((self.pageListGen,1,100),100) 
                            #lastofcover (if fistofcover??)
                            ]
        
    def generate(self):
        self._generation = []
        
        for pageGen in self.pageListGen._instance:
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
    
    # scanningZone: relative to the page 
    # % of zoom in 
    pageScanning = ((5, 2),(10, 2),(5, 3),(5, 2))
    
    tlMarginGen = ((100, 10),(100, 10),(150, 10),(50, 10))
    trMarginGen = ((100, 10),(100, 10),(50, 10),(150, 10))

    tGrid = ( 'regular',(2,0),(0,0) )
    
    #self.nbLines = integerGenerator(40, 5)
#         self.nbLines.setLabel('nbLines')
#         self.LineListGen  = listGenerator(LineGenerator, self.nbLines ,None)
#         self.LineListGen.setLabel("colLine") 
#         self._mygrid = None
#         self.leading= 12    
    
    
    Config = {
        "scanning": pageScanning,
        "pageH":    (700, 10),
        "pageW":    (500, 10),
        "nbPages":  (2,0),
        "lmargin":  tlMarginGen,
        "rmargin":  trMarginGen,
        "grid"  :   tGrid,
        "leading":  (12,1), 
        "lineHeight":(10,1)
        }
#     mydoc = DocMirroredPages((1200, 10),(700, 10),(1, 0),tMargin=(tlMarginGen,trMarginGen),tRuling=tGrid)
    mydoc = DocMirroredPages(Config)

    mydoc.instantiate()
    mydoc.generate()
    gt =  mydoc.exportAnnotatedData(())
#     print gt
    docDom = mydoc.XMLDSFormatAnnotatedData(gt)
#     print etree.tostring(docDom,encoding="utf-8", pretty_print=True)
    docDom.write("tmp.ds_xml",encoding='utf-8',pretty_print=True)    

def tableDataset(nbpages):
    """
        keep all parameters in the synthetic object!! 
    """
    tlMarginGen = ((50, 5),(50, 5),(50, 10),(50, 10))
    trMarginGen = ((50, 5),(50, 5),(50, 10),(50, 10))

    tGrid = ( 'regular',(1,0),(0,0) )
    
    Config = {
        "scanning": None,
        "pageH":    (780, 50),
        "pageW":    (1000, 50),
        "nbPages":  (nbpages,0),
        "lmargin":  tlMarginGen,
        "rmargin":  trMarginGen,
        'pnum'  :True,
        "pnumZone": 0,
        "grid"  :   tGrid,
        "leading":  (12,1), 
        "lineHeight":(10,1),
        "colStruct": (tableGenerator,1,nbpages,((9,3),(12,5)))
        
        }    
    
    mydoc = DocMirroredPages(Config)
    mydoc.myConfig = Config
    mydoc.instantiate()
    mydoc.generate()
    gt =  mydoc.exportAnnotatedData(())
#     print gt
    docDom = mydoc.XMLDSFormatAnnotatedData(gt)
    return docDom

if __name__ == "__main__":

    try: outfile =sys.argv[2]
    except IndexError as e:
        print("usage: layoutObjectGenerator.py #sample <FILENAME>")
        sys.exit() 
    
    try: nbpages = int(sys.argv[1])
    except IndexError as e: nbpages = 1
    
    dom1 = tableDataset(nbpages)
    dom1.write(outfile,xml_declaration=True,encoding='utf-8',pretty_print=True)

    print("saved in %s"%outfile)    


    
    
