# -*- coding: utf-8 -*-
"""


    Samples of layout generators
    
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
from __future__ import absolute_import
from __future__ import  print_function
from __future__ import unicode_literals


try:basestring
except NameError:basestring = str

from lxml import etree
import sys, os.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))

from dataGenerator.numericalGenerator import numericalGenerator
from dataGenerator.numericalGenerator import  integerGenerator
from dataGenerator.numericalGenerator import positiveIntegerGenerator
from dataGenerator.generator import Generator
from dataGenerator.layoutGenerator import layoutZoneGenerator
from dataGenerator.listGenerator import listGenerator
from dataGenerator.typoGenerator import horizontalTypoGenerator,verticalTypoGenerator

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
    def __init__(self,config,configKey):
        """
          "page":{
            "scanning": None,
            "pageH":    (780, 50),
            "pageW":    (1000, 50),
            "nbPages":  (nbpages,0),
            "lmargin":  tlMarginGen,
            "rmargin":  trMarginGen,
            'pnum'  :True,
            "pnumZone": 0,
            "grid"  :   tGrid
        """
        layoutZoneGenerator.__init__(self,config,"page")
        self.leftPage  = pageGenerator(config,"page")
        self.leftPage.setLeftOrRight(1)
        self.leftPage.setParent(self)
        self.rightPage = pageGenerator(config,'page')
        self.rightPage.setLeftOrRight(2)
        self.rightPage.setParent(self)
        
        self._structure = [
                            ((self.leftPage,1,100),(self.rightPage,1,100),100)
                            ]
    
    
class pageGenerator(layoutZoneGenerator):
    """
     
     need to add background zone
     
     
     a page will be /is composed of
                     top margin
         left margin content  right-margin
                     bottom pargin
         
         
    See css box  included boxes:  margin/Border/Padding/Content
         
         
    """
    ID=1
    def __init__(self,config,key):
        layoutZoneGenerator.__init__(self,config,key)
        self._label='PAGE'
        # x, y, x2, y2
        h = self.getMyConfig()['pageH']
        w = self.getMyConfig()['pageW']
        r = self.getMyConfig()["grid"]
        
        hm,hsd=  h
        self.pageHeight = integerGenerator(hm,hsd)
        self.pageHeight.setLabel('height')
        wm,wsd=  w
        self.pageWidth = integerGenerator(wm,wsd)
        self.pageWidth.setLabel('width')

        ##background 

        ##also need X0 and y0 
        self._x0 = 0
        self._y0 = 0
        
        (gridType,(cm,cs),(gm,gs)) = r
        assert gridType == 'regular'
        
        self.nbcolumns = integerGenerator(cm, cs)
        self.nbcolumns.setLabel('nbCol')
        self.gutter = integerGenerator(gm,gs)
        self.ColumnsListGen  = listGenerator(config,'struct',columnGenerator, self.nbcolumns)
        self.ColumnsListGen.setLabel("GRIDCOL")
        
        # required at line level!
        self.leading=None        
        
        
        self.leftOrRight = None
        # WHITE SPACES
        self.pageNumber = None  # should come from documentGen.listGen(page)?
        if self.getMyConfig()['pnum']:
            self.pageNumber = pageNumberGenerator(config)
        
        self._margin = marginGenerator(config)
        
        
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
#         self._marginRegions = []  #->replace by layoutZone?
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


    def setLeftOrRight(self,n): self.leftOrRight = n
    def getLeftMargin(self): return self._margin.getMarginZones()[2]
    
    def getRightMargin(self):return self._margin.getMarginZones()[3]
        
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
        
        # textlayout
        
        #top
        self._margin.getMarginZones()[0].setPositionalGenerators((self._x0,0),(self._y0,0),
                                                                 (self._x0 + self.pageWidth._generation, 0),(self._y0 + t, 0),
                                                                 (self._typeArea_y1,0),(self.pageWidth._generation,0))
        #bottom
        self._margin.getMarginZones()[1].setPositionalGenerators((self._x0,0),(H - b,0),(b,0),
                                                                 (self._x0+self.pageWidth._generation,0),(self._y0 + self.pageHeight._generation,0),
                                                                 (self.pageWidth._generation,0))
        #left
        self._margin.getMarginZones()[2].setPositionalGenerators((self._x0,0),(self._y0,0),
                                                                 (self._x0+l,0),(self._y0 + self.pageHeight._generation,0),
                                                                 (self.pageHeight._generation,0),(l,0))
        #right
        self._margin.getMarginZones()[3].setPositionalGenerators((W - r,0),(self._y0,0),
                                                                 (self._x0+self.pageWidth._generation,0),(self._y0 + self.pageHeight._generation,0),
                                                                 (self.pageHeight._generation,0),(r,0))
        
        for m in  self._margin.getMarginZones(): m._lastYposition = m.getY()._generation
        [m.instantiate() for m in self._margin.getMarginZones()]
        [m.generate() for m in self._margin.getMarginZones()]
#         self._marginRegions = [(self._x0,self._y0,self._typeArea_y1,self.pageWidth._generation), #top
#                                (self._x0,H - b,b,self.pageWidth._generation), #bottom
#                                (self._x0,self._y0,self.pageHeight._generation,l), #left 
#                                (W - r,self._y0,self.pageHeight._generation,r)  #right
#                                 
#                                ]
        #layout
        self._typeArea = [ self._typeArea_x1 , self._typeArea_y1 , self._typeArea_x2 , self._typeArea_x2 , self._typeArea_h , self._typeArea_w]

        #define the 4 margins as layoutZone

    def addPageNumber(self,p):
        """
        """
        zoneIndx = self.getMyConfig()['pnumZone']
        region = self._margin.getMarginZones()[zoneIndx]
        
        # in the middle of the zone
        p.setPositionalGenerators((region.getX()._generation+region.getWidth()._generation*0.5,5),(region.getY()._generation+region.getHeight()._generation*0,5),
                                  (100,0),(100,1),  #fake
                                  (10,0),(10,1))
        
    def generate(self):
        """
            bypass layoutZoneGen: specific to page
        """
#         self.setConfig(self.getParent().getConfig())

        self.setNumber(1)
        self._generation = []
        for obj in self._instance[:2]:
            obj.generate()
            self._generation.append(obj)        
        
            
        if self._margin:
            self._margin.setPage(self)
            self._margin.generate()
            self._generation.append(self._margin)
        t,b,l,r = map(lambda x:x._generation,self._margin._generation[:4])
#         
#         self.pageHeight.generate()
        pH = self.pageHeight._generation
#         self.pageWidth.generate()
        pW = self.pageWidth._generation
        
        self.gutter.generate()
        self.computeAllValues(pH,pW,t, b, l, r)
#         print ([x._generation for x in self._margin.leftMarginGen.getPositionalGenerators()])
        ## margin elements: page numbers
        if self.pageNumber is  not None:
            self.addPageNumber(self.pageNumber)
            self.pageNumber.generate()                
            self._generation.append(self.pageNumber)
            
            
        obj = self._instance[-1]    
        nbCols =  self.ColumnsListGen.getValuedNb()
        self._columnWidth  = (self._typeArea[5] - (nbCols - 1) * self.gutter._generation) / nbCols
#         print (self._typeArea[5], nbCols, self._columnWidth)   
        self._columnHeight = self._typeArea[4]
        
        x1,y1,x2,y2,h,w = self._typeArea

        self._generation.append(self.nbcolumns)
        lastX=x1
        for i,colGen in enumerate(self.ColumnsListGen._instance):
            print (i, colGen)
            colx = lastX +self.gutter._generation #x1 + ( ( i * self._columnWidth) + self.gutter._generation)
            coly = y1
            colH = h
            colW = self._columnWidth
            colGen.setPositionalGenerators( (colx,0),(coly,5),(colx+colW,0),(coly+colH,0),(colH,5),(colW,0))
            colGen.setGrid(self)       
            colGen.setPage(self)
            ## for for the elements
            for layobj, configkey,prob in  self.getMyConfig()['struct']:
                print (layobj, configkey,prob)
                if type(layobj) == tuple:
                    #  layobj:  gen, 
                    content=listGenerator(self.getConfig(), configkey,layobj[1],integerGenerator(*layobj[2]))
                else:    
                    content=layobj(self.getConfig(),configkey)
    #             try:content=self.getConfig()['colStruct'][0](self.getConfig())
    #             except KeyError as e: content=None
                if content is not None:
                    colGen.updateStructure((content,1,prob))
            colGen.instantiate()
#             print (colGen._instance)
            colGen.generate()
#             print (colGen._generation)
            self._generation.append(colGen)          
            lastX = colGen.getX()._generation + colGen.getWidth()._generation  
            
            
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
        

class columnGenerator(layoutZoneGenerator):
    """
        a column generator
        requires a parent : x,y,h,w computed in the parent:
        
        see  CSS Box Model: margin,border, padding
        
    """
    def __init__(self,config,configKey,x=None,y=None,x2=None,y2=None,h=None,w=None):
        layoutZoneGenerator.__init__(self,config,configKey,x=x,y=y,x2=x2,y2=y2,h=h,w=w)
        print('??',configKey)
        self.setLabel("COLUMN")
        self._structure = [
                            [(self.getX(),1,100),(self.getY(),1,100),(self.getX2(),1,100),(self.getY2(),1,100),(self.getHeight(),1,100),(self.getWidth(),1,100),100]
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
        for obj in self._instance[:6]:
            obj.generate()
            self._generation.append(obj)
        
#         colobj =  self._instance[-1]
        self._lastYposition = self.getY()._generation
        for colobj in self._instance[6:]:
            if isinstance(colobj,tableGenerator):
                #table dimensions
                if 'height' in self.getConfig()['table'].keys():
                    m,s =  self.getHeight()._generation * 0.01 * self.getConfig()['table']['height'][0],self.getConfig()['table']['height'][1]
                    colobj.setHeight((m,s))
                    m,s =  self.getWidth()._generation * 0.01 * self.getConfig()['table']['width'][0],self.getConfig()['table']['width'][1]
                    colobj.setWidth((m,s))
                    colobj.getHeight().generate()
                    colobj.getWidth().generate()
                else:
                    # from column
                    colobj.setHeight((self.getHeight().getUple()[0],self.getHeight().getUple()[1]))          
                    colobj.setWidth((self.getWidth().getUple()[0],self.getWidth().getUple()[1]))
                    colobj.getHeight().generate()
                    colobj.getWidth().generate()
                x1,y1,h,w = self.getX()._generation,self._lastYposition,colobj.getHeight()._generation,colobj.getWidth()._generation
                colobj.setPositionalGenerators((x1,0),(y1,0),(x1+w,0),(y1+h,0),(h,0),(w,0))
                colobj.setPage(self.getPage())
                colobj.generate()
                self._generation.append(colobj)            
                
                self._lastYposition = self._lastYposition + colobj.getHeight()._generation
                
            elif isinstance(colobj,listGenerator):
#                 self.leading = integerGenerator(*self.getConfig()['line']['leading'])
                print(self.getConfigKey(),self.getMyConfig())
                self.leading = integerGenerator(*self.getMyConfig()['leading'])

                self.leading.generate()
                self.leading.setLabel('leading')
                
#                 print ('col',self.getY()._generation,self.getHeight()._generation,len(colobj._instance))
#                 print('start line',self.getY()._generation ,self.leading._generation, self.getHeight()._generation)
                for i,lineGen in enumerate(colobj._instance):
                    # too many lines
                    if self._lastYposition > (self.getY()._generation +  self.getHeight()._generation):
#                         print('stop!',  self._lastYposition ,self.getHeight()._generation )
                        continue
                    linex =self.getX()._generation
#                     liney = (i * self.leading._generation) self.getY()._generation
                    liney = self._lastYposition #self.getY()._generation
                    #lineH = 10
                    lineH=integerGenerator(*self.getConfig()['line']['lineHeight'])
                    lineH.generate()                    
                    lineW = self.getWidth()._generation
                    lineGen.setParent(self)
                    lineGen.setPage(self.getPage()) 
                    lineGen.setPositionalGenerators((linex,2),(liney,2),(linex+lineW,2),(liney+lineH._generation,2),(lineH._generation,2),(lineW,2))
#                     print (linex,liney,lineH._generation,lineW)
                    lineGen.generate()
#                     print (lineGen.getX()._generation,lineGen.getY()._generation,lineGen.getHeight()._generation,lineGen.getWidth()._generation,'\n')
                    self._generation.append(lineGen)
                    self._lastYposition = lineGen.getY()._generation + lineGen.getHeight()._generation+ self.leading._generation
#                     print (i, self._lastYposition)
            else:
                ## 
                colx  = self.getX()._generation
                coly = self._lastYposition
                # height: must be computed/updated afterwards
                colH = 10 #self.getConfig()['line']['marginalia']['height']
                colW =  self.getWidth()._generation
                colobj.setParent(self)
                colobj.setPage(self.getPage())
                colobj.setPositionalGenerators((colx,1),(coly,1),(colx+colW,1),(coly+colH,1),(colH,1),(colW,1) ) 
                colobj.generate()
                # update colH 
                self._generation.append(colobj)
                
            
class pageNumberGenerator(layoutZoneGenerator):
    """
        a pagenumgen
    """
    def __init__(self,config,x=None,y=None,h=None,w=None):
        layoutZoneGenerator.__init__(self,config,x=x,y=y,h=h,w=w)
        self._label='LINE'
        
    
    def XMLDSFormatAnnotatedData(self, linfo, obj):
        self.domNode = etree.Element(obj.getLabel())
        self.domNode.set('pagenumber','yes')
        self.domNode.set('DU_row','O')       
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
    def __init__(self,config):
        Generator.__init__(self,config)
        
        top = config['page']["margin"][0][0]
        bottom =  config['page']["margin"][0][1]
        left =  config['page']["margin"][0][2]
        right =  config['page']["margin"][0][3]        
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
        
        
        self.leftMarginGen=layoutZoneGenerator(config)
        self.leftMarginGen.setLabel('leftMargin')
        self.rightMarginGen=layoutZoneGenerator(config)
        self.rightMarginGen.setLabel('rightMargin')
        self.topMarginGen=layoutZoneGenerator(config)
        self.topMarginGen.setLabel('topMargin')
        self.bottomMarginGen=layoutZoneGenerator(config)
        self.bottomMarginGen.setLabel('bottomMargin')
        
        
        self._structure = [ ((self._top,1,100),(self._bottom,1,100),(self._left,1,100),(self._right,1,100)
                            ,(self.topMarginGen,1,100)
                            ,(self.bottomMarginGen,1,100)
                            ,(self.leftMarginGen,1,100)
                            ,(self.rightMarginGen,1,100)
                             ,100)
                              ]
    
    def setPage(self,p):self._page=p 
    def getPage(self):return self._page
    
    def getDimensions(self): return self._top,self._bottom,self._left, self._right
        
    def getMarginZones(self):
        """
            return the 4 margins as layoutZone
        """
        return [ self.topMarginGen, self.bottomMarginGen,self.leftMarginGen, self.rightMarginGen,]
        
        
    def generate(self):
        self._generation = []
        for obj in self._instance[:4]:
            obj.generate()
            self._generation.append(obj)
        
        t,b,l,r = map(lambda x:x._generation,self._generation)
        #top
        self.getMarginZones()[0].setPositionalGenerators((self.getPage()._x0,0),(self.getPage()._y0,0),
                                                         (self.getPage()._x0+self.getPage().pageWidth._generation,0),(self.getPage()._y0+self._top._generation,0),
                                                         (self._top._generation,0),(self.getPage().pageWidth._generation,0))
        # bottom
        self.getMarginZones()[1].setPositionalGenerators((self.getPage()._x0,0),(self.getPage().pageHeight._generation - b,0),
                                                         (self.getPage()._x0+self.getPage().pageWidth._generation,0),(self.getPage()._y0+self.getPage().pageHeight._generation,0),
                                                         (b,0),(self.getPage().pageWidth._generation,0))
        # left
        self.getMarginZones()[2].setPositionalGenerators((self.getPage()._x0,0),(self.getPage()._y0,0),
                                                         (self.getPage()._x0 + l,0),(self.getPage()._y0 + self.getPage().pageHeight._generation,0),
                                                         (self.getPage().pageHeight._generation,0),(l,0))
        # right
        self.getMarginZones()[3].setPositionalGenerators((self.getPage().pageWidth._generation - r,0),(self.getPage()._y0,0),
                                                         (self.getPage()._x0+self.getPage().pageWidth._generation,0),(self.getPage()._y0+self.getPage().pageHeight._generation,0),
                                                         (self.getPage().pageHeight._generation,0),(r,0))
                    
        for m in self.getMarginZones():
#             print ([(x._mean,x._std) for x in m._instance],m._generation)
            m.generate()
            self._generation.append(m) 
            m._lastYposition = m.getY()._generation 
        
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



class catchword(layoutZoneGenerator):
    """
        catchword: always bottom right?
    """
    def __init__(self,config,x=None,y=None,h=None,w=None):
        layoutZoneGenerator.__init__(self,config,x=x,y=y,h=h,w=w)
        self.setLabel("CATCHWORD")
        
        self._structure = [
                            [(self.getX(),1,100),(self.getY(),1,100),(self.getX2(),0),(self.getY2(),0),(self.getHeight(),1,100),(self.getWidth(),1,100),100]
                            ]        
                

class marginaliaGenerator(layoutZoneGenerator):
    """
        marginalia Gen: assume relation with 'body' part
    """
    def __init__(self,config,configKey,x=None,y=None,h=None,w=None):
        layoutZoneGenerator.__init__(self,config,configKey,x=x,y=y,h=h,w=w)
        self.setLabel("MARGINALIA")
        #pointer to the parent structures!! line? page,?
        #lineGen!!
          
        self._structure = [
                            [ (self.getX(),1,100),(self.getY(),1,100),(self.getX2(),1,100),(self.getY2(),1,100),(self.getHeight(),1,100),(self.getWidth(),1,100),100 ]
                            ]        
        for layobj, configkey,prob in  self.getMyConfig()['struct']:

            if type(layobj) == tuple:
                content=listGenerator(self.getConfig(), configkey,layobj[1],integerGenerator(*layobj[2]))
            else:    
                content=layobj(self.getConfig(),configkey)
            if content is not None:
                self.updateStructure((content,1,prob))
  
  
    def generate(self):
        """
            prepare data for line
            nbLines: at instanciate() level?
              
        """
        # x,h, w,h
        self._generation = []
        for obj in self._instance[:6]:
            obj.generate()
            self._generation.append(obj)
          
          
          
#         colobj =  self._instance[-1]
#         print ('M', self.getX()._generation,self.getY()._generation,self.getHeight()._generation)
        self._lastYposition = self.getY()._generation
        for colobj in self._instance[6:]:
            if isinstance(colobj,listGenerator):
                self.leading = integerGenerator(*self.getMyConfig()['leading'])
                self.leading.generate()
                self.leading.setLabel('leading')
#                 print (self._lastYposition , self.getHeight()._generation)
                for i,lineGen in enumerate(colobj._instance):
                    # too many lines
                    if self._lastYposition > ( self.getY()._generation + self.getHeight()._generation):
                        continue
                    linex =self.getX()._generation
#                     liney = (i * self.leading._generation) self.getY()._generation
                    liney = self.leading._generation + self._lastYposition #self.getY()._generation
                    self._lastYposition = liney
                    #lineH = 10
                    lineH=integerGenerator(*self.getMyConfig()['lineHeight'])
                    lineH.generate()                    
                    lineW = self.getWidth()._generation   
                    lineGen.setParent(self)
                    lineGen.setPage(self.getPage()) 
                    lineGen.setPositionalGenerators((linex,2),(liney,2),(lineH._generation,2),(lineW,2))
        #             lineGen.setParent(self)        
                    lineGen.generate()
                    self._generation.append(lineGen)
                    self._lastYposition += self.leading._generation              
            else:
                colobj.generate()
                self._generation.append(colobj)
            
class LineGenerator(layoutZoneGenerator):
    """
        what is specific to a line? : content
            content points to a textGenerator
            
        
        for table noteGen must be positioned better!
            if parent= column
            if parent= cell
            if parent =...
            
    """ 
    def __init__(self,config,configKey,x=None,y=None,h=None,w=None):
        layoutZoneGenerator.__init__(self,config,configKey,x=x,y=y,h=h,w=w)
        self.setLabel("LINE")
        self._noteGen = None
        self._noteGenProb = None
        print (self.getMyConfig(),configKey)
        if "marginalia" in self.getMyConfig():
            print( self.getMyConfig()["marginalia"]['generator'],self.getMyConfig()["marginalia"]['config'])
            self._noteGen = self.getMyConfig()["marginalia"]['generator'](self.getConfig(), self.getMyConfig()["marginalia"]['config'])
            self._noteGenProba= self.getMyConfig()["marginalia"]['proba']
#         self._justifixationGen = None #justificationGenerator() # center, left, right, just, random
        
        self._structure = [
                            ((self.getX(),1,100),(self.getY(),1,100),(self.getX2(),1,100),(self.getY2(),1,100),(self.getHeight(),1,100),(self.getWidth(),1,100),100)
                            ]
        if self._noteGen is not None:
            self._structure = [
                           ((self.getX(),1,100),(self.getY(),1,100),(self.getX2(),1,100),(self.getY2(),1,100),(self.getHeight(),1,100),(self.getWidth(),1,100),(self._noteGen,1,self._noteGenProba),100)
                            ]
        
#     def setPage(self,p):self._page=p
#     def getPage(self): return self._page
    
        
    def generate(self):
        """
        need a pointer to the column to select the possible margins
        need a pointer to the margin to add the element  if any
        
        need to be able to position vertically and horizontally
        
        """
        self._generation = []
        for obj in self._instance[:6]:
            obj.generate()
            self._generation.append(obj)
        
        self._lastYposition = self.getY()._generation
        for obj in  self._instance[6:]:
            if isinstance(obj,marginaliaGenerator):
                
                ### parent =   margin!!
                # left or right margin
                # need to go up to the grid to know where the column is
                if self.getPage().leftOrRight == 1: 
                    # get left margin
                    myregion= self.getPage().getLeftMargin()
                    #left page: put on the left margin, right otherwise? 
                    marginaliax = myregion.getX()._generation + 10
                else:
                    #marginaliax = 600 
                    myregion= self.getPage().getRightMargin()
                    marginaliax = myregion.getX()._generation + 20 #myregion[0]+10
                    
                marginaliay = self.getY()._generation
#                 self._noteGen.getHeight().generate()
                print(obj.getMyConfig())
                marginaliaH = obj.getMyConfig()['height']
#                 self._noteGen.getWidth()().generate()
                marginaliaW = myregion.getWidth()._generation
                
#                 marginaliaH, probH = self.getConfig()['line']['marginalia']['height']
#                 marginaliaW = myregion.getWidth()._generation * 0.66
                # compute position according to the justifiaction : need parent, 
#                 self._noteGen.setPositionalGenerators((marginaliax,5),(marginaliay,5),(marginaliax+marginaliaW,5),(marginaliay+marginaliaH,5),(marginaliaH,0),(marginaliaW,5))
                self._noteGen.setPositionalGenerators((marginaliax,5),(marginaliay,5),None,None,(marginaliaH,0),(marginaliaW,5))

                self._noteGen.setPage(self.getPage())
                self._noteGen.setParent(myregion)
                self._noteGen.generate()
#                 print ("qq",myregion._lastYposition,  self._noteGen.getY()._generation +  self._noteGen.getHeight()._generation)
                self._generation.append(self._noteGen)
                    
        return self
    
class cellGenerator(layoutZoneGenerator):
    """
        cellGen
        
        for the set of lines: define at this level the horizontal and vertical justification
        
        
        similar to column? a cell containts a grid???
            for instance: padding as well
        
        
    """ 
    def __init__(self,config,configKey,x=None,y=None,h=None,w=None):
        layoutZoneGenerator.__init__(self,config,configKey,x=x,y=y,h=h,w=w)
        self.setLabel("CELL")
        
        self._index = None
        self._rowSpan = positiveIntegerGenerator(1,0)
        self._rowSpan.setLabel('rowSpan')
        self._colSpan = positiveIntegerGenerator(1,0)
        self._colSpan.setLabel('colSpan')
        self.lineConfig = self.getConfig()[self.getMyConfig()['lineGen']]
        self.leading = positiveIntegerGenerator(*self.lineConfig['leading'])
        self.leading.setLabel('leading')
        # default value needed for  LineListGen
        self.nbLinesG = positiveIntegerGenerator(1, 0)
        self._LineListGen = listGenerator(config,self.getMyConfig()['lineGen'],LineGenerator, self.nbLinesG)
        self._LineListGen.setLabel("cellline")
        self._structure =[((self.getX(),1,100),(self.getY(),1,100),(self.getX2(),1,100),(self.getY2(),1,100),(self.getHeight(),1,100),(self.getWidth(),1,100),
                           (self.leading,1,100),(self._rowSpan,1,100),(self._colSpan,1,100),
#                            (self.VJustification,1,100),
                           (self._LineListGen,1,100),100)]
        
    
    def getIndex(self): return self._index
    def setIndex(self,i,j): self._index=(i,j)
    def setNbLinesGenerator(self,g):
        self.nbLinesG = g
        self._LineListGen = listGenerator(self.getConfig(),self.getMyConfig()['lineGen'],LineGenerator, self.nbLinesG)
    def getNbLinesGenerator(self): return self.nbLinesG
    
    def computeYStart(self,HJustification,nbLines, lineH,leading):
        """
            compute where to start 'writing' according to justification and number of lines (height of the block)
        """
        blockH=  (nbLines * (lineH+leading)) - leading
        if HJustification == horizontalTypoGenerator.TYPO_TOP:
            return 0
        if HJustification == horizontalTypoGenerator.TYPO_HCENTER: 
            return  (0.5 * self.getHeight()._generation)  -  (0.5 * blockH)
        if HJustification == horizontalTypoGenerator.TYPO_BOTTOM:
            return  (self.getHeight()._generation)  -  (blockH)
        
    def generate(self):
        self._generation=[]
        for obj in self._instance[:-1]:
            obj.generate()
            self._generation.append(obj)
#         print self.getLabel(),self._generation
        self._LineListGen.instantiate()

        self.vjustification = self.getMyConfig()['vjustification'].generate()._generation
        self.hjustification = self.getMyConfig()['hjustification'].generate()._generation
        # vertical justification : find the y start
#         ystart=self.computeYStart(self.VJustification._generation, self._LineListGen.getValuedNb()*self.leading._generation)
        lineH=integerGenerator(*self.lineConfig['lineHeight'])
        lineH.generate()
        ystart=self.computeYStart( self.hjustification, self._LineListGen.getValuedNb(),lineH._generation,self.leading._generation)
        ystart = max(ystart,-2)
#         print(self.getY()._generation, ystart ,self._LineListGen.getValuedNb(), self.hjustification,self.getIndex())
        
        xstart = self.getWidth()._generation * 0.05   # Generator !!
        rowPaddingGen = numericalGenerator(1,0)
        rowPaddingGen.generate()
        nexty= ystart +  self.getY()._generation + rowPaddingGen._generation
        lLines=[]
        for i,lineGen in enumerate(self._LineListGen._instance):
            # too many lines
#             if (i * self.leading._generation) + (self.getY()._generation + lineH) > (self.getY()._generation + self.getHeight()._generation):
            if nexty + lineH._generation >  5+ (self.getY()._generation + self.getHeight()._generation):
#                 print ('\t',nexty +lineH._generation ,  (self.getY()._generation + self.getHeight()._generation))
                continue

            liney = nexty
            #                                                    0.75                           0.1 
            lineW=integerGenerator(self.getWidth()._generation*0.5,self.getWidth()._generation*0.3)
            lineW.generate()
            
            if self.vjustification == verticalTypoGenerator.TYPO_LEFT:
                linex = self.getX()._generation + (xstart)        
            if self.vjustification == verticalTypoGenerator.TYPO_RIGHT:
                linex = self.getX()._generation + self.getWidth()._generation - lineW._generation     
            elif self.vjustification == verticalTypoGenerator.TYPO_VCENTER:
                linex =  self.getX()._generation + self.getWidth()._generation * 0.5 - lineW._generation *0.5  
            lineGen.setPositionalGenerators((linex,1),(liney,1),(linex+lineW._generation,0),(liney+lineH._generation,0),(lineH._generation,0.5),(lineW._generation,0))
#             lineGen.setPositionalGenerators((linex,0),(liney,0),(lineH,0),(lineW * 0.5,lineW * 0.1))
            lineGen.setPage(self.getPage())  
            lineGen.setParent(self)
            lLines.append(lineGen)
            lineGen.generate()
            rowPaddingGen.generate()
            nexty= lineGen.getY()._generation +self.leading._generation +  lineGen.getHeight()._generation #+  rowPaddingGen._generation
            lineGen.setLabel('LINE')
            self._generation.append(lineGen)
        
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
    def __init__(self,config,configKey):
        layoutZoneGenerator.__init__(self,config,configKey)

        self.setLabel('TABLE')
                 
        nbRows=self.getMyConfig()['nbRows']
        self.rowHeightVariation = self.getMyConfig()['rowHeightVariation']
        self.rowHStd=self.rowHeightVariation[1]
        self.columnWidthVariation = self.getMyConfig()['columnWidthVariation']
        
        self._rowPadding = positiveIntegerGenerator(*self.getMyConfig()['rowPadding'])
        self._rowPadding.setLabel('rowpadding')
        if 'widths' in self.getMyConfig()['column']:
            self.nbCols = positiveIntegerGenerator(len(self.getMyConfig()['column']['widths']),0)
        else:
            nbCols=self.getMyConfig()['nbCols']
            self.nbCols = positiveIntegerGenerator(nbCols[0],nbCols[1])
        self.nbCols.setLabel('nbCols')
        self.nbRows = positiveIntegerGenerator(nbRows[0],nbRows[1])
        self.nbRows.setLabel('nbRows')
        
        self._bSameRowHeight=self.getMyConfig()['row']['sameRowHeight']
        self._lRowsGen = listGenerator(config,'',layoutZoneGenerator, self.nbRows)
        self._lRowsGen.setLabel("row")
        self._lColumnsGen = listGenerator(self.getMyConfig()['column'],'',layoutZoneGenerator, self.nbCols )
        self._lColumnsGen.setLabel("col")
        
        self._structure = [
            ((self.getX(),1,100),(self.getY(),1,100),(self.getX2(),1,100),(self.getY2(),1,100),(self.getHeight(),1,100),(self.getWidth(),1,100),
             (self.nbCols,1,100),(self.nbRows,1,100), (self._rowPadding,1,100),
             (self._lColumnsGen,1,100),(self._lRowsGen,1,100),100)
            ]
        
    def generateRowHeight(self):
        """
            either 'nbRows"  or "nbLines" 
        """
        if 'nbLines' in self.getMyConfig()['column']:
                nbMaxLines = max(x[0] for x in self.getMyConfig()['column']['nbLines'])
#                 lineH=self.getConfig()['line']['lineHeight']
                cellconfig = self.getConfig()[self.getMyConfig()['cell']]
                lineConfig =  cellconfig['lineGen']
                lineH =  self.getConfig()[lineConfig]['lineHeight']
                lineHG=positiveIntegerGenerator(*lineH)
                lineHG.generate()
                
                nblineG=positiveIntegerGenerator(nbMaxLines,0)
                nblineG.generate()
#                 self._rowHeightG = positiveIntegerGenerator(nblineG._generation*lineHG._generation,self.rowHStd)
                self._rowHeightG = positiveIntegerGenerator(nblineG._generation*(self._rowPadding._generation+lineHG._generation),0)        
    
    
    def generateCellsForRow(self,row):
        """
            
        """
        
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
        self._rowHeightG = positiveIntegerGenerator(self._rowHeightM,self.rowHStd)
        
#         self._rowHeightM = int(round(self.getHeight()._generation / nbRows))
#         self._rowHeightG = numericalGenerator(self._rowHeightM,self._rowHeightM*0.5)
        self.lCols=[]
        self.lRows=[]
        nextx= self.getX()._generation
        
        
        for i,colGen in enumerate(self._lColumnsGen._instance):
            if nextx > self.getX()._generation + self.getWidth()._generation:
                continue
            colx = nextx #self.getX()._generation + ( i * self._columnWidth)
            coly = self.getY()._generation
            colH = self.getHeight()._generation
            if 'widths' in self.getMyConfig()['column']:
                colW = self.getMyConfig()['column']['widths'][i] * self.getWidth()._generation
#                 print (colW,self.getConfig()['table']['column']['widths'][i],self.getWidth()._generation)
            else:
                self._columnWidthG.generate()
                colW = self._columnWidthG._generation
            colGen.setNumber(i)
            colGen.setPositionalGenerators((colx,0),(coly,0),(colx+colW,0),(coly+colH,0),(colH,0),(colW,0))
#             colGen.setGrid(self)       
            colGen.setLabel("COL")
            colGen.setPage(self.getPage())
            colGen.generate()
            nextx= colGen.getX()._generation + colGen.getWidth()._generation
            self._generation.append(colGen)
            self.lCols.append(colGen)
            
        ## ROW
        ## height = either from nbRows / rowHeightVariation
        ##          or max(nblines)  for this row
        
        
        # max nblines 
        
        rowH = None
        nexty = self.getY()._generation
        for i,rowGen in enumerate(self._lRowsGen._instance):
            self.generateRowHeight()
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
#             print (i,rowH,self._rowPadding._generation)
            rowy = nexty 
            # here test that that there is enough space for the row!!
#             print self._rowHeightM, self._rowHeightG._generation
            rowW = self.getWidth()._generation
            rowGen.setLabel("ROW")
            rowGen.setNumber(i)
            rowGen.setPage(self.getPage())
            rowGen.setPositionalGenerators((rowx,0),(rowy,0),(rowx+rowW,0),(rowy+rowH,0),(rowH,0),(rowW,0))
            rowGen.generate()
            nexty = rowGen.getY()._generation + rowGen.getHeight()._generation 
#             print i, rowy, self.getHeight()._generation
            self.lRows.append(rowGen)
            self._generation.append(rowGen)  
#             print("%d %s %f"%(i,self._bSameRowHeight,rowGen.getHeight()._generation))     
            
        ## table specific stuff
        ## table headers, stub,....


        #### assume the grid col generated?
        ### introduce a hierarchical column
        #### split the col into N subcols  :: what is tricky: headers  split the first         
        ### hierarchical row: for this needs big rows and split
        
        ## creation of the cells; then content in the cells
        self.lCellGen=[]
        for icol,col in enumerate(self.lCols):
            if 'nbLines' in self.getMyConfig()['column']:
#                 print (icol, self.getConfig()['table']['column']['nbLines'])
                nblines=self.getMyConfig()['column']['nbLines'][icol]
                nbLineG = positiveIntegerGenerator(*nblines)
            else: 
                nblines=self.getMyConfig()["nbLines"]
                nbLineG = positiveIntegerGenerator(*nblines)
            for irow, row in enumerate(self.lRows):
                cell=cellGenerator(self.getConfig(),'cellTable')
                cell.setLabel("CELL")
                cell.setPositionalGenerators((col.getX()._generation,0),(row.getY()._generation,0),
                                             (col.getX()._generation+col.getWidth()._generation,0),(row.getY()._generation+row.getHeight()._generation,0),
                                             (row.getHeight()._generation,0),(col.getWidth()._generation,0))
                # colunm header? {'column':{'header':{'colnumber':1,'justification':'centered'}}
                
                if irow < self.getMyConfig()['column']['header']['colnumber'] :
#                     print (self.getMyConfig()['column']['header'])
                    cell.setConfigKey(self.getMyConfig()['column']['header']['cell'])
                
                if 'hjustification' in self.getMyConfig()['column']:
                    cell.getMyConfig()['hjustification']  = self.getMyConfig()['column']['hjustification'][icol]
#                 else:
#                     cell.getMyConfig()['vjustification'] = storedInitialVJustification
#                     print ('qq',cell.getMyConfig()['vjustification'].x,self.getConfig()['cellTable']['vjustification'].x)
#                 cell.getMyConfig()['hjustification'] = self.getConfig()['cellTable']['hjustification']
#                 print ('c',irow,self.getMyConfig()['column']['header']['colnumber'] ,cell.getMyConfig()['vjustification'].x)
                # row header?
                self.lCellGen.append(cell)
                cell.setNbLinesGenerator(nbLineG)
                cell.setIndex(irow,icol)
                cell.instantiate()
                cell.setPage(self.getPage())
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
    def __init__(self,dConfig):
        
#         tpageH = dConfig["page"]['pageH']
#         tpageW = dConfig["page"]['pageW']
        tnbpages = dConfig["page"]['nbPages']
#         tMargin = (dConfig["page"]['lmargin'],dConfig["page"]['rmargin'])
#         tRuling = dConfig["page"]['grid']
        
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

#         self._margin = tMargin
#         self._ruling = tRuling      
        
        
        self.pageListGen = listGenerator(dConfig,'page',pageGenerator,self._nbpages)
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
        metadata.text = str(self.getConfig())
        for info,page in gtdata:
            pageNode = page.XMLDSFormatAnnotatedData(info,page)
            root.append(pageNode)
        return self.docDom
        
        
    def generate(self):
        self._generation = []
        
        ## 1-2 cover pages
        
        for i,pageGen in enumerate(self.pageListGen._instance):
            #if double page: start with 1 = right?
            pageGen.setConfig(self.getConfig())
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
#         tpageH = dConfig["page"]['pageH']
#         tpageW = dConfig["page"]['pageW']
        tnbpages = dConfig["page"]['nbPages']
        self._nbpages = integerGenerator(tnbpages[0],tnbpages[1])
        self._nbpages.setLabel('nbpages')        
#         tMargin = (dConfig["page"]['margin'],dConf)
#         tRuling = dConfig["page"]['grid']
        
#         documentGenerator.__init__(self,tpageH,tpageW,tnbpages,tMargin,tRuling)
        documentGenerator.__init__(self,dConfig)
        
        self.setConfig(dConfig)

#         self._lmargin, self._rmargin = tMargin
#         self._ruling= tRuling
        self.pageListGen = listGenerator(dConfig,'page',doublePageGenerator,self._nbpages)
        self.pageListGen.setLabel('pages')
        self._structure = [
                            #firstSofcover (first and second)
                            ((self.pageListGen,1,100),100) 
                            #lastofcover (if fistofcover??)
                            ]
    
    
    def getPages_instances(self):
        return self.pageListGen._instance
    
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

# def docm():
#     
#     # scanningZone: relative to the page 
#     # % of zoom in 
#     pageScanning = ((5, 2),(10, 2),(5, 3),(5, 2))
#     
#     tlMarginGen = ((100, 10),(100, 10),(150, 10),(50, 10))
#     trMarginGen = ((100, 10),(100, 10),(50, 10),(150, 10))
# 
#     tGrid = ( 'regular',(2,0),(0,0) )
#     
#     Config = {
#         "scanning": pageScanning,
#         "pageH":    (700, 10),
#         "pageW":    (500, 10),
#         "nbPages":  (2,0),
#         "lmargin":  tlMarginGen,
#         "rmargin":  trMarginGen,
#         "grid"  :   tGrid,
#         "leading":  (12,1), 
#         "lineHeight":(10,1)
#         }
# #     mydoc = DocMirroredPages((1200, 10),(700, 10),(1, 0),tMargin=(tlMarginGen,trMarginGen),tRuling=tGrid)
#     mydoc = DocMirroredPages(Config)
# 
#     mydoc.instantiate()
#     mydoc.generate()
#     gt =  mydoc.exportAnnotatedData(())
# #     print gt
#     docDom = mydoc.XMLDSFormatAnnotatedData(gt)
# #     print etree.tostring(docDom,encoding="utf-8", pretty_print=True)
#     docDom.write("tmp.ds_xml",encoding='utf-8',pretty_print=True)    
# 
# def StAZHDataset(nbpages):
#     """
#         page header (centered)
#         page number (mirrored: yes and no)
#         catch word (bottom right)
#         marginalia (left margin; mirrored also?)
#          
#     """
#     tlMarginGen = ((50, 5),(50, 5),(50, 5),(50, 5))
#     trMarginGen = ((50, 5),(50, 5),(50, 5),(50, 5))
# 
#     tGrid = ( 'regular',(1,0),(0,0) )
#         
#     Config = {
#         "page":{
#             "scanning": None
#             ,"pageH":    (780, 50)
#             ,"pageW":    (500, 50)
#             ,"nbPages":  (nbpages,0)
#             ,"margin": [tlMarginGen, trMarginGen]
#             ,'pnum'  :{'position':"left"}
#             ,"pnumZone": 0
#             ,"grid"  :   tGrid
#         }
#         #column?
#         ,"line":{
#              "leading":     (15+5,1) 
#             ,"lineHeight":  (15,1)
#             ,"justification":'left'
#             ,'marginalia':[marginaliaGenerator,10]
#             ,'marginalialineHeight':10
#             }
#         
#         ,"colStruct": (listGenerator,LineGenerator,(20,0))
# #         ,'table':{
# #             "nbRows":  (40,0)
# #             ,"nbCols":  (5,0)
# #             ,"rowHeightVariation":(0,0)
# #             ,"columnWidthVariation":(0,0)
# #             ,'column':{'header':{'colnumber':1,'justification':'centered'}}
# #             ,'row':{"sameRowHeight": True }
# #             ,'cell':{'justification':'right','line':{"leading":(14,0)}}
# #             }
#         }    
#     mydoc = DocMirroredPages(Config)
#     mydoc.instantiate()
#     mydoc.generate()
#     gt =  mydoc.exportAnnotatedData(())
# #     print gt
#     docDom = mydoc.XMLDSFormatAnnotatedData(gt)
#     return docDom 
#         
#         
# def ABPRegisterDataset(nbpages):
#     """
#         ABP register
#         
#     """
#     tlMarginGen = ((50, 5),(50, 5),(50, 5),(50, 5))
#     trMarginGen = ((50, 5),(50, 5),(50, 5),(50, 5))
# 
#     tGrid = ( 'regular',(1,0),(0,0) )
#     
#     # should be replaced by an object?
#     ABPREGConfig = {
#         "page":{
#             "scanning": None
#             ,"pageH":    (780, 50)
#             ,"pageW":    (500, 50)
#             ,"nbPages":  (nbpages,0)
#             ,"margin": [tlMarginGen, trMarginGen]
#             ,'pnum'  :{'position':"left"}  # also ramdom?
#             ,"pnumZone": 0
#             ,"grid"  :   tGrid
#         }
#         #column?
#         ,"line":{
#              "leading":     (5,4) 
#             ,"lineHeight":  (18,2)
#             ,"justification":'left'
#             }
#         
#         ,"colStruct": (tableGenerator,1,nbpages)
#         ,'table':{
#             "nbRows":  (30,2)
#             ,"nbCols":  (5,1)
#             ,"rowHeightVariation":(0,0)
#             ,"columnWidthVariation":(0,0)
#             ,'column':{'header':{'colnumber':1,'justification':'centered'}}
#             ,'row':{"sameRowHeight": True }
#             ,'cell':{'justification':'right','line':{"leading":(14,0)}}
#             }
#         }    
#     
#     Config=ABPREGConfig
#     mydoc = DocMirroredPages(Config)
#     mydoc.instantiate()
#     mydoc.generate()
#     gt =  mydoc.exportAnnotatedData(())
# #     print gt
#     docDom = mydoc.XMLDSFormatAnnotatedData(gt)
#     return docDom    
# 
# def NAFDataset(nbpages):
#     tlMarginGen = ((50, 5),(50, 5),(50, 5),(50, 5))
#     trMarginGen = ((50, 5),(50, 5),(50, 5),(50, 5))
# 
#     tGrid = ( 'regular',(1,0),(0,0) )
#     #for NAF!: how to get the column width??? 
#     NAFConfig = {
#         "page":{
#             "scanning": None,
#             "pageH":    (780, 50),
#             "pageW":    (500, 50),
#             "nbPages":  (nbpages,0),
#             "margin": [tlMarginGen, trMarginGen],
#             'pnum'  :True,
#             "pnumZone": 0,
#             "grid"  :   tGrid
#         }
#         #column?
#         ,"line":{
#              "leading":     (5,4) 
#             ,"lineHeight":  (10,1)
#             ,"justification":'left'
#             }
#         
#         ,"colStruct": (tableGenerator,1,nbpages)
#         ,'table':{
#              "nbRows":  (35,10)
#             ,"nbCols":  (5,0)
#             ,"rowHeightVariation":(20,5)
#             ,"columnWidthVariation":(0,0)
#             #                                                                      proportion of col width known          
#             ,'column':{'header':{'colnumber':1,'justification':'centered'}
#                        ,'widths':(0.01,0.05,0.05,0.5,0.2,0.05,0.05,0.05,0.05,0.05,0.05)
#                        #nb textlines 
#                         ,'nbLines':((1,0.1),(1,0.1),(1,0.1),(4,1),(3,1),(1,1),(1,0.5),(1,1),(1,0.5),(1,0.5),(1,0.5))
# 
#                        }
#             ,'row':{"sameRowHeight": False }
#             ,'cell':{'justification':'right','line':{"leading":(14,0)}}
#             }
#         }  
#     Config=NAFConfig
#     mydoc = DocMirroredPages(Config)
#     mydoc.instantiate()
#     mydoc.generate()
#     gt =  mydoc.exportAnnotatedData(())
# #     print gt
#     docDom = mydoc.XMLDSFormatAnnotatedData(gt)
#     return docDom    
# 
# 
# def NAHDataset(nbpages):
#     """
#     @todo: need to put H centered lines
#     """
#     
#     tlMarginGen = ((50, 5),(50, 5),(50, 5),(50, 5))
#     trMarginGen = ((50, 5),(50, 5),(50, 5),(50, 5))
# 
#     tGrid = ( 'regular',(1,0),(0,0) )
#     #for NAF!: how to get the column width??? 
#     NAFConfig = {
#         "page":{
#             "scanning": None,
#             "pageH":    (780, 50),
#             "pageW":    (500, 50),
#             "nbPages":  (nbpages,0),
#             "margin": [tlMarginGen, trMarginGen],
#             'pnum'  :True,
#             "pnumZone": 0,
#             "grid"  :   tGrid
#         }
#         #column?
#         ,"line":{
#              "leading":     (5,4) 
#             ,"lineHeight":  (10,1)
#             ,"vjustification":verticalTypoGenerator([0.5,0.25,0.25])
#             #                  0: top
#             ,'hjustification':horizontalTypoGenerator([0.33,0.33,0.33])
#             }
#         
#         ,"colStruct": (tableGenerator,1,nbpages)
#         ,'table':{
#              "nbRows":  (35,10)
#             ,"nbCols":  (5,0)
#             ,"rowHeightVariation":(20,5)
#             ,"columnWidthVariation":(0,0)
#             #                                                                      proportion of col width known          
#             ,'column':{'header':{'colnumber':1,'vjustification':verticalTypoGenerator([0,1,0])}
#                        ,'widths':(0.01,0.05,0.05,0.5,0.2,0.05,0.05,0.05,0.05,0.05,0.05)
#                        #nb textlines 
#                         ,'nbLines':((1,0.1),(1,0.1),(1,0.1),(4,1),(3,1),(1,1),(1,0.5),(1,1),(1,0.5),(1,0.5),(1,0.5))
# 
#                        }
#             ,'row':{"sameRowHeight": False }
#             ,'cell':{'hjustification':horizontalTypoGenerator([0.75,0.25,0.0]),'vjustification':verticalTypoGenerator([0,0,1]),'line':{"leading":(14,0)}}
#             }
#         }  
#     Config=NAFConfig
#     mydoc = DocMirroredPages(Config)
#     mydoc.instantiate()
#     mydoc.generate()
#     gt =  mydoc.exportAnnotatedData(())
# #     print gt
#     docDom = mydoc.XMLDSFormatAnnotatedData(gt)
#     return docDom    
# 
# def NAH2Dataset(nbpages):
#     """
#     @todo: need to put H centered lines
#     """
#     
#     tlMarginGen = ((50, 5),(50, 5),(50, 5),(50, 5))
#     trMarginGen = ((50, 5),(50, 5),(50, 5),(50, 5))
# 
#     tGrid = ( 'regular',(1,0),(0,0) )
#     #for NAF!: how to get the column width??? 
#     NAFConfig = {
#         "page":{
#             "scanning": None,
#             "pageH":    (1200, 50),
#             "pageW":    (1600, 50),
#             "nbPages":  (nbpages,0),
#             "margin": [tlMarginGen, trMarginGen],
#             'pnum'  :True,
#             "pnumZone": 0,
#             "grid"  :   tGrid
#         }
#         #column?
#         ,"line":{
#              "leading":     (6,0) 
#             ,"lineHeight":  (14,1)
#             ,"vjustification":verticalTypoGenerator([0.5,0.25,0.25])
#             #                  0: top
#             ,'hjustification':horizontalTypoGenerator([0.33,0.33,0.33])
#             }
#         
#         ,"colStruct": ( (tableGenerator,100),  )
# #         ,"colStruct": (tableGenerator,None,None)
#         ,'table':{
#              "nbRows":  (50,5)
#             ,"nbCols":  (5,0)
#             ,"rowHeightVariation":(20,10)
#             ,"columnWidthVariation":(0,0)
#             # if none: use container dim
#             ,'height': None 
#             ,'width':None
#             #                                                                      proportion of col width known          
#             ,'column':{'header':{'colnumber':1,'vjustification':verticalTypoGenerator([0,1,0])}
#                        ,'widths':(0.01,0.05,0.05,0.05,0.05,0.3,0.01,0.05,0.2,0.05,0.05,0.05,0.05,0.05,0.05)
#                        #nb textlines 
#                        # nha with for 93  : more lines on the right  : nahv2_3
#                         ,'nbLines':((1,0.1),(1,0.1),(1,0.1),(1,0.1),(2,0.1),(6,2),(1,0),(0,0.5),(2,1),(1,1),(1,0.5),(3,1),(3,0.5),(1,0.5),(1,0.5))
#                         #basic[sic] Nha
# #                         ,'nbLines':((1,0.1),(1,0.1),(1,0.1),(1,0.1),(2,0.1),(6,2),(1,0),(0,0.5),(2,1),(1,1),(1,0.5),(1,1),(1,0.5),(1,0.5),(1,0.5))
#                         # nha with one line per cell: still need REG
# #                         ,  'nbLines':((1,0.1),(1,0.1),(1,0),(1,0.1),(1,0.1),(1,0),(0,0),(0,0.5),(1,1),(1,1),(1,0.5),(1,1),(1,0.5),(1,0.5),(1,0.5))
# 
#                        }
#             ,'row':{"sameRowHeight": False }
#             ,'cell':{'hjustification':horizontalTypoGenerator([0.66,0.20,0.13]),'vjustification':verticalTypoGenerator([0.75,0.25,0]),'line':{"leading":(14,1)}}
#             }
#         }  
#     Config=NAFConfig
#     mydoc = DocMirroredPages(Config)
#     mydoc.instantiate()
#     mydoc.generate()
#     gt =  mydoc.exportAnnotatedData(())
# #     print gt
#     docDom = mydoc.XMLDSFormatAnnotatedData(gt)
#     return docDom    
# 
# def TUEDataset(nbpages):
#     """
#     description: page with text and table
#     """
#     
#     tlMarginGen = ((50, 5),(50, 5),(50, 5),(50, 5))
#     trMarginGen = ((50, 5),(50, 5),(50, 5),(50, 5))
# 
#     tGrid = ( 'regular',(1,0),(0,0) )
#     #for NAF!: how to get the column width??? 
#     NAFConfig = {
#         "page":{
#             "scanning": None,
#             "pageH":    (600, 50),
#             "pageW":    (400, 50),
#             "nbPages":  (nbpages,0),
#             "margin": [tlMarginGen, trMarginGen],
#             'pnum'  :True,
#             "pnumZone": 0,
#             "grid"  :   tGrid
#         }
#         #column?
#         ,"line":{
#              "leading":     (15,4) 
#             ,"lineHeight":  (10,1)
#             ,"vjustification":verticalTypoGenerator([0.5,0.25,0.25])
#             #                  0: top
#             ,'hjustification':horizontalTypoGenerator([0.33,0.33,0.33])
#             }
#         
#         #                ( struct,                  (m,s) ) 
#         ,"colStruct": ( ((listGenerator,LineGenerator,(10,0)),100),(tableGenerator,100), ((listGenerator,LineGenerator,(10,0)),100))
# #         ,"colStruct":  (((listGenerator,LineGenerator,(20,0)),100),)
# 
#         ,'table':{
#              "nbRows":  (10,0)
#             ,"nbCols":  (5,2)
#             ,"rowHeightVariation":(20,2)
#             ,"columnWidthVariation":(0,0)
#             ## % od the container?
#             ,'height': (20,5)
#             ,'width':(100,20)
#             ,'nbLines':(1,0) ## common values for all columns  ('column/nblines does not exist)
#             #                                                                      proportion of col width known          
#             ,'column':{'header':{'colnumber':1,'vjustification':verticalTypoGenerator([0,1,0])}
# #                        ,'widths':(0.01,0.05,0.05,0.3,0.05,0.05,0.2,0.05,0.05,0.05,0.05,0.05,0.05)
#                        #nb textlines 
# #                         ,'nbLines':((1,0),)
# #                         ,'nbLines':((1,0.1),(1,0.1),(1,0.1),(6,2),(0,0),(0,0.5),(3,1),(1,1),(1,0.5),(1,1),(1,0.5),(1,0.5),(1,0.5))
# #                         ,  'nbLines':((1,0.1),(1,0.1),(1,0.1),(2,0),(0,0),(0,0.5),(1,1),(1,1),(1,0.5),(1,1),(1,0.5),(1,0.5),(1,0.5))
# 
#                        }
#             ,'row':{"sameRowHeight": False }
#             ,'cell':{'hjustification':horizontalTypoGenerator([100,0.0,0.0]),'vjustification':verticalTypoGenerator([0.75,0.25,0]),'line':{"leading":(14,1)}}
#             }
#         }  
#     Config=NAFConfig
#     mydoc = DocMirroredPages(Config)
#     mydoc.instantiate()
#     mydoc.generate()
#     gt =  mydoc.exportAnnotatedData(())
# #     print gt
#     docDom = mydoc.XMLDSFormatAnnotatedData(gt)
#     return docDom    
# 
# 
def testDataset(nbpages):
    """
        BAR 
         
        marginalia:  summ (0.5)  + num (100)  
        body : lines
         
    """
    tlMarginGen = ((50, 5),(50, 5),(50, 5),(50, 5))
    trMarginGen = ((50, 5),(50, 5),(50, 5),(50, 5))
 
    tGrid = ( 'regular',(1,0),(0,0) )
     
    # should be replaced by an object?
    BARConfig = {
        "page":{
            "scanning": None,
            "pageH":    (768, 10),
            "pageW":    (499, 10),
            "nbPages":  (nbpages,0),
            "margin": [tlMarginGen, trMarginGen],
            'pnum'  :True,
            "pnumZone": 0,
            "grid"  :   tGrid
            ,"struct": ( ((listGenerator,LineGenerator,(50,0)),'line',100),) 
        }
        ,"line":{
             "leading":     (6,0) 
            ,"lineHeight":  (20,1)
            ,"vjustification":verticalTypoGenerator([0.5,0.25,0.25])
            #                  0: top
            ,'hjustification':horizontalTypoGenerator([0.33,0.33,0.33])
            ,'marginalia':{"generator":marginaliaGenerator
                           ,"proba":30
                           ,'config':'note'
                           }
            }
        ,'note':{
                'height':(30,5)
                ,'lineHeight':(10,1)
                , "leading":  (5,0) 
               ,"vjustification":verticalTypoGenerator([0.5,0.25,0.25])
               ,'hjustification':horizontalTypoGenerator([0.33,0.33,0.33])     
               ,"struct": ( ((listGenerator,LineGenerator,(2,1)),'line',100),)                       
            
            }
        
        }     
     
     
    Config=BARConfig
    mydoc = DocMirroredPages(Config)
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
    
#     dom1 = ABPRegisterDataset(nbpages)
#     dom1 = NAFDataset(nbpages)
#     dom1 = NAH2Dataset(nbpages)

#     dom1 = StAZHDataset(nbpages)
    dom1 = testDataset(nbpages)
#     dom1 = TUEDataset(nbpages)

    dom1.write(outfile,xml_declaration=True,encoding='utf-8',pretty_print=True)

    print("saved in %s"%outfile)    


    
    
