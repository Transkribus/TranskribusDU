"""
A class that reflect a decoration to be made on certain XML node using WX

"""

import types, os
from collections import defaultdict
from lxml import etree
#import cStringIO
import wx

sEncoding = "utf-8"

def setEncoding(s):
    global sEncoding
    sEncoding = s

class DecoSeparator:
    """
    this is not properly a decoration but rather a separator of decoration in the toolbar
    """
    def __init__(self, cfg, sSurname, xpCtxt):
        """
        cfg is a configuration object
        sSurname is the surname of the decoration and the section name in the config file!
        xpCtxt is an XPath context
        """
        self.sSurname = sSurname
    
    def __str__(self):
        return "--------"
    
    def isSeparator(self):
        return True
    
    def setXPathContext(self, xpCtxt):
        pass


class Deco:
    """A general decoration class"""

    
    def __init__(self, cfg, sSurname, xpCtxt):
        """
        cfg is a configuration object
        sSurname is the surname of the decoration and the section name in the config file!
        xpCtxt is an XPath context
        """
        self.sSurname = sSurname
        self.xpMain = cfg.get(sSurname, "xpath") # a main XPath that select nodes to be decorated in this way
        self.xpCtxt = xpCtxt  #this context may include the declaration of some namespace
        sEnabled = cfg.get(sSurname, "enabled").lower()
        self.bEnabled = sEnabled in ['1', 'yes', 'true']

    
    def isSeparator(cls):
        return False
    isSeparator = classmethod(isSeparator)
    
    def __str__(self):
        return "(Surname=%s xpath==%s)" % (self.sSurname, self.xpMain)
    
    def getDecoClass(cls, sClass):
        """given a decoration type, return the associated class"""
        c = globals()[sClass]
        if type(c) != types.ClassType: raise Exception("No such decoration type: '%s'"%sClass)
        return c
    getDecoClass = classmethod(getDecoClass)
    
    def getSurname(self):
        return self.sSurname
    
    def getMainXPath(self):
        return self.xpMain
    
    def isEnabled(self):
        return self.bEnabled
    
    def setEnabled(self, b=True):
        self.bEnabled = b
        return b
    
    def isActionable(self):
        return False
    
    def setXPathContext(self, xpCtxt):
        self.xpCtxt = xpCtxt
    
    def xpathError(self, node, xpExpr, eExcpt, sMsg=""):
        """report an xpath error"""
        iMaxLen = 200 # to truncate the node serialization
        print "-"*60
        print "--- XPath ERROR on class %s"%self.__class__
        print "---   xpath=%s" % xpExpr
        print "---   Python Exception=%s" % str(eExcpt)
        if sMsg: print "---   Info: %s" % sMsg
        try:
            sNode = etree.tostring(node)
        except:
            sNode = str(node)
        if len(sNode) > iMaxLen: sNode = sNode[:iMaxLen] + "..."
        print "--- XML node = %s" % sNode
        print "-"*60
    
    def toInt(cls, s):
        try:
            return int(s)
        except ValueError:
            return int(round(float(s)))
    toInt = classmethod(toInt)
                    
    def xpathToInt(self, node, xpExpr, iDefault=0, bShowError=True):
        """The given XPath expression should return an int on the given node.
        The XPath expression should return a scalar or a one-node nodeset
        On error, return the default int value
        """
        try:
#            s = node.xpathEval(xpExpr)
            self.xpCtxt.setContextNode(node)
            s = self.xpCtxt.xpathEval(xpExpr)
            if type(s) == types.ListType: 
                try:
                    s = s[0].text
                except AttributeError:
                    s = s[0]    #should be an attribute value
            return Deco.toInt(s)
        except Exception, e:
            if bShowError: self.xpathError(node, xpExpr, e, "xpathToInt return %d as default value"%iDefault)
        return iDefault
    
    def xpathToStr(self, node, xpExpr, sDefault, bShowError=True):
        """The given XPath expression should return a string on the given node
        The XPath expression should return a scalar or a one-node nodeset
        On error, return the default int value
        """
        try:
#            s = node.xpathEval(xpExpr)
            self.xpCtxt.setContextNode(node)
            s = self.xpCtxt.xpathEval(xpExpr)
            if type(s) == types.ListType: 
                try:
                    s = s[0].text
                except AttributeError:
                    s = s[0]
            return s
        except Exception, e:
            if bShowError: self.xpathError(node, xpExpr, e, "xpathToStr return %s as default value"%sDefault)
        return sDefault
    
    def xpathEval(self, node, xpExpr):
        """ evaluate the xpath expression
        return None on error
        """
        try:
#            s = node.xpathEval(xpExpr)
            self.xpCtxt.setContextNode(node)
            return self.xpCtxt.xpathEval(xpExpr)
        except Exception, e:
            self.xpathError(node, xpExpr, e, "xpathEval return None")
            return None
        
    def beginPage(self, node):
        """called before any sequnce of draw for a given page"""
        pass
    
    def endPage(self, node):
        """called before any sequnce of draw for a given page"""
        pass
    
    def draw(self, wxh, node):
        """draw the associated decorations, return the list of wx created objects"""
        return []


class DecoBBXYWH(Deco):
    """A decoration with a bounding box defined by X,Y for its top-left corner and width/height.
    xpX, xpY, xpW, xpH are scalar XPath expressions to get the associated x,y,w,h values from the selected nodes
    """
    
    def __init__(self, cfg, sSurname, xpCtxt):
        """
        cfg is a config file
        sSurname is the decoration surname and the section name in the config file
        This section should contain the following items: x, y, w, h
        """
        Deco.__init__(self, cfg, sSurname, xpCtxt)
        #now get the xpath expressions that let uis find x,y,w,h from a selected node
        self.xpX, self.xpY = cfg.get(sSurname, "xpath_x"), cfg.get(sSurname, "xpath_y")
        self.xpW, self.xpH = cfg.get(sSurname, "xpath_w"), cfg.get(sSurname, "xpath_h")
        
        self.xpInc = cfg.get(sSurname, "xpath_incr")  #to increase the BB width and height
        
        self._node = None
        
    def __str__(self):
        s = Deco.__str__(self)
        s += "+(x=%s y=%s w=%s h=%s)" % (self.xpX, self.xpY, self.xpW, self.xpH)
        return s
    
    def runXYWHI(self, node):
        """get the X,Y values for a node and put them in cache"""
        if self._node != node: 
            self._x = self.xpathToInt(node, self.xpX, 1)
            self._y = self.xpathToInt(node, self.xpY, 1)
            self._w = self.xpathToInt(node, self.xpW, 1)
            self._h = self.xpathToInt(node, self.xpH, 1)
            self._inc = self.xpathToInt(node, self.xpInc, 0)
            self._x,self._y = self._x-self._inc, self._y-self._inc
            self._w,self._h = self._w+2*self._inc, self._h+2*self._inc
            self._node = node
        return (self._x, self._y, self._w, self._h, self._inc)
        
class DecoRectangle(DecoBBXYWH):
    """A rectangle
    """
    
    def __init__(self, cfg, sSurname, xpCtxt):
        DecoBBXYWH.__init__(self, cfg, sSurname, xpCtxt)
        #now get the xpath expressions that let us find the rectangle line and fill colors
        self.xpLineColor = cfg.get(sSurname, "xpath_LineColor")
        self.xpLineWidth = cfg.get(sSurname, "xpath_LineWidth")
        self.xpFillColor = cfg.get(sSurname, "xpath_FillColor")
        self.xpFillStyle = cfg.get(sSurname, "xpath_FillStyle")        
        
    def __str__(self):
        s = "%s="%self.__class__
        s += DecoBBXYWH.__str__(self)
        return s

    def draw(self, wxh, node):
        """draw itself using the wx handle
        return a list of created WX objects"""
        lo = DecoBBXYWH.draw(self, wxh, node)
        x,y,w,h,inc = self.runXYWHI(node)
        sLineColor = self.xpathToStr(node, self.xpLineColor, "#000000")
        iLineWidth = self.xpathToInt(node, self.xpLineWidth, 1)
        sFillColor = self.xpathToStr(node, self.xpFillColor, "#000000")
        sFillStyle = self.xpathToStr(node, self.xpFillStyle, "Solid")
        obj = wxh.AddRectangle((x, -y), (w, -h), 
                                             LineWidth=iLineWidth,
                                             LineColor=sLineColor,
                                             FillColor=sFillColor,
                                             FillStyle=sFillStyle)
        lo.append(obj)
        return lo
    
class DecoTextBox(DecoRectangle):
    """A text within a bounding box (a rectangle)
    """
    
    def __init__(self, cfg, sSurname, xpCtxt):
        DecoRectangle.__init__(self, cfg, sSurname, xpCtxt)
        self.xpContent  = cfg.get(sSurname, "xpath_content")
        self.xpFontSize = cfg.get(sSurname, "xpath_font_size")
        self.xpFontColor = cfg.get(sSurname, "xpath_font_color")

    def __str__(self):
        s = "%s="%self.__class__
        s += DecoRectangle.__str__(self)
        return s
    
    def draw(self, wxh, node):
        """draw itself using the wx handle
        return a list of created WX objects"""

        lo = DecoRectangle.draw(self, wxh, node)
        
        #add the text itself
        txt = self.xpathToStr(node, self.xpContent, "")
        iFontSize = self.xpathToInt(node, self.xpFontSize, 8)
        sFontColor = self.xpathToStr(node, self.xpFontColor, 'BLACK')
        x,y,w,h,inc = self.runXYWHI(node)
        obj = wxh.AddScaledTextBox(txt, (x, -y+inc),
                                   Size=iFontSize,
                                   Family=wx.ROMAN, Position='tl',
                                   Color=sFontColor, PadSize=0, LineColor=None)
        lo.append(obj)
        return lo

class DecoText(DecoBBXYWH):
    """A text 
    """
    
    def __init__(self, cfg, sSurname, xpCtxt):
        DecoBBXYWH.__init__(self, cfg, sSurname, xpCtxt)
        self.xpContent  = cfg.get(sSurname, "xpath_content")
        self.xpFontSize = cfg.get(sSurname, "xpath_font_size")
        self.xpFontColor = cfg.get(sSurname, "xpath_font_color")

    def __str__(self):
        s = "%s="%self.__class__
        s += DecoBBXYWH.__str__(self)
        return s
    
    def draw(self, wxh, node):
        """draw itself using the wx handle
        return a list of created WX objects"""

        lo = DecoBBXYWH.draw(self, wxh, node)
        
        #add the text itself
        txt = self.getText(wxh, node)
        iFontSize = self.xpathToInt(node, self.xpFontSize, 8)
        sFontColor = self.xpathToStr(node, self.xpFontColor, 'BLACK')
        x,y,w,h,inc = self.runXYWHI(node)
        obj = wxh.AddScaledTextBox(txt, (x, -y-h/2.0),
                                    Size=iFontSize,
                                    Family=wx.ROMAN, Position='cl',
                                    Color=sFontColor, PadSize=0, LineColor=None)
            lo.append(obj)
        return lo
    
    def getText(self, wxh, node):
        return self.xpathToStr(node, self.xpContent, "")


class DecoUnicodeChar(DecoText):
    """A character encoded in Unicode
    We assume the unicode index is given in a certain base, e.g. 10 or 16
    """
    def __init__(self, cfg, sSurname, xpCtxt):
        DecoText.__init__(self, cfg, sSurname, xpCtxt)
        self.base = int(cfg.get(sSurname, "code_base"))

    def getText(self, wxh, node):
        sEncodedText = self.xpathToStr(node, self.xpContent, "")
        try:
            return eval('u"\\u%04x"' %  int(sEncodedText, self.base))
        except ValueError:
            print "DecoUnicodeChar: ERROR: base=%d code=%s"%(self.base, sEncodedText)
            return ""


class DecoImageBox(DecoRectangle):
    """An image with a box around it
    """
    
    def __init__(self, cfg, sSurname, xpCtxt):
        DecoRectangle.__init__(self, cfg, sSurname, xpCtxt)
        self.xpHRef  = cfg.get(sSurname, "xpath_href")
        
    def __str__(self):
        s = "%s="%self.__class__
        s += DecoRectangle.__str__(self)
        return s
    
    def draw(self, wxh, node):
        """draw itself using the wx handle
        return a list of created WX objects"""

        lo = []
        
        #add the image itself
        x,y,w,h,inc = self.runXYWHI(node)
        sFilePath = self.xpathToStr(node, self.xpHRef, "")
        if sFilePath:
            try:
                img = wx.Image(sFilePath, wx.BITMAP_TYPE_ANY)
                obj = wxh.AddScaledBitmap(img, (x,-y), h)
                lo.append(obj)
            except Exception, e:
                print "DecoImageBox ERROR: File %s: %s"%(sFilePath, str(e))
        
        lo.append( DecoRectangle.draw(self, wxh, node) )
        return lo


class DecoImage(DecoBBXYWH):
    """An image
    """
    
    def __init__(self, cfg, sSurname, xpCtxt):
        DecoBBXYWH.__init__(self, cfg, sSurname, xpCtxt)
        self.xpHRef  = cfg.get(sSurname, "xpath_href")
        
    def __str__(self):
        s = "%s="%self.__class__
        s += DecoBBXYWH.__str__(self)
        return s
    
    def draw(self, wxh, node):
        """draw itself using the wx handle
        return a list of created WX objects"""

        lo = DecoBBXYWH.draw(self, wxh, node)
        
        #add the image itself
        x,y,w,h,inc = self.runXYWHI(node)
        sFilePath = self.xpathToStr(node, self.xpHRef, "")
        if sFilePath:
            if not os.path.exists(sFilePath): 
                print "WARNING: deco Image: file does not exists: '%s'"%sFilePath
            else:
                img = wx.Image(sFilePath, wx.BITMAP_TYPE_ANY)
                try:
                    obj = wxh.AddScaledBitmap(img, (x,-y), h)
                    lo.append(obj)
                except Exception, e:
                    print "DecoImage ERROR: File %s: %s"%(sFilePath, str(e))
        
        return lo
    
class DecoOrder(DecoBBXYWH):
    """Show the order with lines
    """
    
    def __init__(self, cfg, sSurname, xpCtxt):
        DecoBBXYWH.__init__(self, cfg, sSurname, xpCtxt)
        self.xpLineColor = cfg.get(sSurname, "xpath_LineColor")
        self.xpLineWidth = cfg.get(sSurname, "xpath_LineWidth")
        
    def __str__(self):
        s = "%s="%self.__class__
        s += DecoBBXYWH.__str__(self)
        return s
    
    def beginPage(self, node):
        """called before any sequnce of draw for a given page"""
        self.bInit = False

    def draw(self, wxh, node):
        """draw itself using the wx handle
        return a list of created WX objects"""

        lo = DecoBBXYWH.draw(self, wxh, node)
        x,y,w,h,inc = self.runXYWHI(node)
        sLineColor = self.xpathToStr(node, self.xpLineColor, "BLACK")
    
        x, y = int(x + w/2.0), int(y + h/2.0)
        if self.bInit:
            #draw a line
            iLineWidth = self.xpathToInt(node, self.xpLineWidth, 1)
            obj = wxh.AddLine( [(self.prevX, -self.prevY), (x, -y)]
                                    , LineWidth=iLineWidth
                                    , LineColor=sLineColor)
            lo.append(obj)
        else:
            self.bInit = True
            iEllipseParam = min(w,h) / 2
            wxh.AddEllipse((x, -y), (iEllipseParam, -iEllipseParam), LineColor=sLineColor, LineWidth=5, FillStyle="Transparent")
        self.prevX, self.prevY = x, y
        return lo

class DecoLine(Deco):
    """A line from x1,y1 to x2,y2
    """
    
    def __init__(self, cfg, sSurname, xpCtxt):
        Deco.__init__(self, cfg, sSurname, xpCtxt)
        self.xpX1, self.xpY1 = cfg.get(sSurname, "xpath_x1"), cfg.get(sSurname, "xpath_y1")
        self.xpX2, self.xpY2 = cfg.get(sSurname, "xpath_x2"), cfg.get(sSurname, "xpath_y2")
        #now get the xpath expressions that let us find the rectangle line and fill colors
        self.xpLineWidth = cfg.get(sSurname, "xpath_LineWidth")
        self.xpLineColor = cfg.get(sSurname, "xpath_LineColor")
        self._node = None        

    def __str__(self):
        s = "%s="%self.__class__
        s += "+(x1=%s y1=%s x2=%s y2=%s)" % (self.xpX1, self.xpY1, self.xpX2, self.xpY2)
        return s

    def draw(self, wxh, node):
        """draw itself using the wx handle
        return a list of created WX objects"""
#        print node.serialize()
#        print self.xpX
#        for n in node.xpathEval(self.xpX): print n.serialize()
        iLARGENEG = -9999
        lo = Deco.draw(self, wxh, node)
        
        if self._node != node: 
            self._x1 = self.xpathToInt(node, self.xpX1, iLARGENEG)
            self._y1 = self.xpathToInt(node, self.xpY1, iLARGENEG)
            self._x2 = self.xpathToInt(node, self.xpX2, iLARGENEG)
            self._y2 = self.xpathToInt(node, self.xpY2, iLARGENEG)            
            self._node = node
        
        if self._x1 != iLARGENEG and self._y1 != iLARGENEG and self._x2 != iLARGENEG and self._y2 != iLARGENEG:
            sLineColor = self.xpathToStr(node, self.xpLineColor, "#000000")
            iLineWidth = self.xpathToInt(node, self.xpLineWidth, 1)
            #draw a line
            obj = wxh.AddLine( [(self._x1, -self._y1), (self._x2, -self._y2)]
                                        , LineWidth=iLineWidth
                                        , LineColor=sLineColor)
            lo.append(obj)
        return lo


class DecoREAD(Deco):
    """
    READ PageXml has a special way to encode coordinates.
    like:
        <Coords points="985,390 1505,390 1505,440 985,440"/>

    or
        <Baseline points="985,435 1505,435"/>
    """    
    def __init__(self, cfg, sSurname, xpCtxt):
        Deco.__init__(self, cfg, sSurname, xpCtxt)
        self.xpCoords = cfg.get(sSurname, "xpath_lxy")

    def _getCoordList(self, node):
        sCoords = self.xpathToStr(node, self.xpCoords, "")
        try:
            ltXY = []
            for _sPair in sCoords.split(' '):
                (sx, sy) = _sPair.split(',')
                ltXY.append((int(sx), int(sy)))
        except Exception, e:
            print "ERROR: polyline coords are bad: '%s'"%sCoords
            raise e        
        return ltXY
    
    def _coordList_to_BB(self, ltXY):
        """
        return (x1, y1), (x2, y2)
        """
        lX = [_x for _x,_y in ltXY]
        lY = [_y for _x,_y in ltXY]
        return (min(lX), max(lY)), (max(lX), min(lY))
    
    
class DecoREADTextLine(DecoREAD):
    """A TextLine as defined by the PageXml format of the READ project
    <TextLine id="line_1551946877389_284" custom="readingOrder {index:0;} Item-name {offset:0; length:11;} Item-price {offset:12; length:2;}">
        <Coords points="985,390 1505,390 1505,440 985,440"/>
        <Baseline points="985,435 1505,435"/>
        <TextEquiv>
            <Unicode>Salgadinhos    12</Unicode>
        </TextEquiv>
    </TextLine>
    """
    def __init__(self, cfg, sSurname, xpCtxt):
        DecoREAD.__init__(self, cfg, sSurname, xpCtxt)
        self.xpContent  = cfg.get(sSurname, "xpath_content")
        self.xpFontColor = cfg.get(sSurname, "xpath_font_color")
        self.xpFit = cfg.get(sSurname, "xpath_fit_text_size").lower()

    def __str__(self):
        s = "%s="%self.__class__
        return s
    
    def _getFontSize(self, node, ltXY, txt, Family=wx.FONTFAMILY_TELETYPE):
        """
        compute the font size so as to fit the polygon
        
        and the extent of the 'x' character for this font size
        return iFontSize, ExtentX, ExtentY
        """
        (x1, y1), (x2, y2) = self._coordList_to_BB(ltXY)
        
        dc = wx.ScreenDC()
        # compute for font size of 24 and do proportional
        dc.SetFont(wx.Font(24, Family, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        Ex, Ey = dc.GetTextExtent("x")
        iFontSizeX = 24 * abs(x2-x1) / Ex / len(txt)
        iFontSizeY = 24 * abs(y2-y1) / Ey
        sFit = self.xpathToStr(node, self.xpFit, 'xy', bShowError=False)
        if sFit == "x":
            iFontSize = iFontSizeX
        elif sFit == "y":
            iFontSize = iFontSizeY
        else:
            iFontSize = min(iFontSizeX, iFontSizeY)
        dc.SetFont(wx.Font(iFontSize, Family, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        Ex, Ey = dc.GetTextExtent("x")
        del dc
        
        return iFontSize, Ex, Ey

    def draw(self, wxh, node):
        """draw itself using the wx handle
        return a list of created WX objects"""

        lo = []
        
        #add the text itself
        txt = self.getText(wxh, node)
        
        sFontColor = self.xpathToStr(node, self.xpFontColor, 'BLACK')

        # Position and computation of font size
        ltXY = self._getCoordList(node)
        
        iFontSize, Ex, Ey = self._getFontSize(node, ltXY, txt, Family=wx.FONTFAMILY_TELETYPE)
        
        x, y = ltXY[0] 
        obj = wxh.AddScaledText(txt, (x, -y+iFontSize/6), Size=iFontSize
                                       , Family=wx.FONTFAMILY_TELETYPE
                                       , Position='tl'
                                       , Color=sFontColor)
        lo.append(obj)
        return lo
    
    def getText(self, wxh, node):
        return self.xpathToStr(node, self.xpContent, "")


class READ_custom:
    """
    Everything related to the PageXML custom attribute
    """
    @classmethod
    def parseCustomAttr(cls, s, bNoCase=True):
        """
        The custom attribute contains data in a CSS style syntax.
        We parse this syntax here and return a dictionary of list of dictionary
        
        Example:
        parseCustomAttr( "readingOrder {index:4;} structure {type:catch-word;}" )
            --> { 'readingOrder': [{ 'index':'4' }], 'structure':[{'type':'catch-word'}] }
        """
        dic = defaultdict(list)
        
        s = s.strip()
        lChunk = s.split('}')
        if lChunk:
            for chunk in lChunk:    #things like  "a {x:1"
                chunk = chunk.strip()
                if not chunk: continue
                
                try:
                    sNames, sValues = chunk.split('{')   #things like: ("a,b", "x:1 ; y:2")
                except Exception:
                    raise ValueError("Expected a '{' in '%s'"%chunk)
                
                #the dictionary for that name
                dicValForName = dict()
                
                lsKeyVal = sValues.split(';') #things like  "x:1"
                for sKeyVal in lsKeyVal:
                    if not sKeyVal.strip(): continue  #empty
                    try:
                        sKey, sVal = sKeyVal.split(':')
                    except Exception:
                        raise ValueError("Expected a comma-separated string, got '%s'"%sKeyVal)
                    sKey = sKey.strip().lower() if bNoCase else sKey.strip()
                    dicValForName[sKey] = sVal.strip()
                
                lName = sNames.split(',')
                for name in lName:
                    name = name.strip().lower() if bNoCase else name.strip()
                    dic[name].append(dicValForName)
        return dic
        
class DecoREADTextLine_custom_offset(DecoREADTextLine, READ_custom):
    """
    Here we show the annotation by offset found in the custom attribute
    """
    def __init__(self, cfg, sSurname, xpCtxt):
        DecoREADTextLine.__init__(self, cfg, sSurname, xpCtxt)
        self.xpLabel     = cfg.get(sSurname, "xpath_label")
        self.xpLineColor = cfg.get(sSurname, "xpath_LineColor")
        self.xpBackgroundColor = cfg.get(sSurname, "xpath_background_color")

    def draw(self, wxh, node):
        """
        draw itself using the wx handle
        return a list of created WX objects
        """

        lo = []
        
        #add the text itself
        txt = self.getText(wxh, node)
        
        sFontColor = self.xpathToStr(node, self.xpFontColor, 'BLACK')
        sLineColor = self.xpathToStr(node, self.xpLineColor, "#000000")
        sBackgroundColor = self.xpathToStr(node, self.xpBackgroundColor, "#000000")

        # Position and computation of font size
        ltXY = self._getCoordList(node)
        
        iFontSize, Ex, Ey = self._getFontSize(node, ltXY, txt
                                              , Family=wx.FONTFAMILY_TELETYPE)
        
        dCustom = self.parseCustomAttr(node.get("custom"), bNoCase=True)
#         try:
#             _ldLabel = dCustom[self.xpathToStr(node, self.xpLabel, "").lower()]
#             iOffset = int(_dLabel["offset"])
#             iLength = int(_dLabel["length"])
#         except KeyError:
#             iOffset = 0
#             iLength = 0
# 
#         # some annotation ?
#         if iLength > 0:        
#             x, y = ltXY[0] 
#             x += Ex * iOffset
#         
#             obj = wxh.AddScaledTextBox(txt[iOffset:iOffset+iLength]
#                                        , (x, -y+iFontSize/6)
#                                        , Size=iFontSize
#                                        , Family=wx.FONTFAMILY_TELETYPE
#                                        , Position='tl'
#                                        , Color=sFontColor
#                                        , LineColor=sLineColor
#                                        , BackgroundColor=sBackgroundColor)
#             lo.append(obj)
        try:
            x0, y0 = ltXY[0] 
            _ldLabel = dCustom[self.xpathToStr(node, self.xpLabel, "").lower()]
            for _dLabel in _ldLabel:
                try:
                    iOffset = int(_dLabel["offset"])
                    iLength = int(_dLabel["length"])
                    x = x0 + Ex * iOffset
                    y = -y0+iFontSize/6
                    obj = wxh.AddScaledTextBox(txt[iOffset:iOffset+iLength]
                                       , (x, y)
                                       , Size=iFontSize
                                       , Family=wx.FONTFAMILY_TELETYPE
                                       , Position='tl'
                                       , Color=sFontColor
                                       , LineColor=sLineColor
                                       , BackgroundColor=sBackgroundColor)
                    lo.append(obj)
                except KeyError:
                    pass
        except KeyError:
            pass
        return lo
    

class DecoPolyLine(DecoREAD):
    """A polyline along 
    x1,y1,x2,y2, ...,xn,yn
        or
    x1,y1 x2,y2 .... xn,yn
    
    Example of config:
        [TextLine]
        type=DecoPolyLine
        xpath=.//TextLine/Coords
        xpath_lxy=@points
        xpath_LineColor="RED"
        xpath_FillStyle="Solid"
    
    JL Meunier - March 2016
    """
    
    def __init__(self, cfg, sSurname, xpCtxt):
        DecoREAD.__init__(self, cfg, sSurname, xpCtxt)
        #now get the xpath expressions that let us find the rectangle line and fill colors
        self.xpLineWidth = cfg.get(sSurname, "xpath_LineWidth")
        self.xpLineColor = cfg.get(sSurname, "xpath_LineColor")
        #cached values
        self._node = None         
        self._lxy = None        

    def __str__(self):
        s = "%s="%self.__class__
        s += "+(coords=%s)" % (self.xpCoords)
        return s

    def draw(self, wxh, node):
        """draw itself using the wx handle
        return a list of created WX objects"""
#        print node.serialize()
#        print self.xpX
#        for n in node.xpathEval(self.xpX): print n.serialize()
        lo = DecoREAD.draw(self, wxh, node)
        
        if self._node != node: 
            self._lxy = self._getCoordList(node)
            self._node = node
        
        if self._lxy:
            sLineColor = self.xpathToStr(node, self.xpLineColor, "#000000")
            iLineWidth = self.xpathToInt(node, self.xpLineWidth, 1)
            for (x1, y1), (x2, y2) in zip(self._lxy, self._lxy[1:]):
                #draw a line
                obj = wxh.AddLine( [(x1, -y1), (x2, -y2)]
                                            , LineWidth=iLineWidth
                                            , LineColor=sLineColor)
                lo.append(obj)             
        return lo


class DecoClosedPolyLine(DecoPolyLine):
    """A polyline that closes automatically the shape
    JL Meunier - September 2016
    """
    def __init__(self, cfg, sSurname, xpCtxt):
        DecoPolyLine.__init__(self, cfg, sSurname, xpCtxt)
        
    def _getCoordList(self, node):
        lCoord = DecoPolyLine._getCoordList(self, node)
        if lCoord: lCoord.append(lCoord[0])
        return lCoord

        
class DecoTextPolyLine(DecoPolyLine, DecoText):
    """A polyline that closes automatically the shape
    JL Meunier - September 2016
    """
    def __init__(self, cfg, sSurname, xpCtxt):
        DecoPolyLine.__init__(self, cfg, sSurname, xpCtxt)
        DecoText          .__init__(self, cfg, sSurname, xpCtxt)
        self.xpX_Inc = cfg.get(sSurname, "xpath_x_incr")  #to shift the text
        self.xpY_Inc = cfg.get(sSurname, "xpath_y_incr")  #to shift the text
            
    def draw(self, wxh, node):

        lo = Deco.draw(self, wxh, node)
        
        if self._node != node: 
            self._lxy = self._getCoordList(node)
            self._node = node
        
        #lo = DecoClosedPolyLine.draw(self, wxh, node)
        
        #add the text itself
        x, y = self._lxy[0]
        
        x_inc = self.xpathToInt(node, self.xpX_Inc, 0, False)        
        y_inc = self.xpathToInt(node, self.xpY_Inc, 0, False)        
        
        txt = self.xpathToStr(node, self.xpContent, "")
        iFontSize = self.xpathToInt(node, self.xpFontSize, 8)
        sFontColor = self.xpathToStr(node, self.xpFontColor, 'BLACK')
        
        obj = wxh.AddScaledTextBox(txt, (x+x_inc, -y-y_inc),
                                   Size=iFontSize,
                                   Family=wx.ROMAN, Position='tl',
                                   Color=sFontColor, PadSize=0, LineColor=None)
        lo.append(obj)    
        return lo

    
class DecoLink(Deco):
    """A link from x1,y1 to x2,y2
    """
    
    def __init__(self, cfg, sSurname, xpCtxt):
        Deco.__init__(self, cfg, sSurname, xpCtxt)
        self.xpX1, self.xpY1 = cfg.get(sSurname, "xpath_x1"), cfg.get(sSurname, "xpath_y1")
        #the following expression must be evaluated twice
        self.xpEvalX2, self.xpEvalY2 = cfg.get(sSurname, "eval_xpath_x2"), cfg.get(sSurname, "eval_xpath_y2")
        self.xpDfltX2, self.xpDfltY2 = cfg.get(sSurname, "xpath_x2_default"), cfg.get(sSurname, "xpath_y2_default")
        #now get the xpath expressions that let us find the rectangle line and fill colors
        self.xpLineWidth = cfg.get(sSurname, "xpath_LineWidth")
        self.xpLineColor = cfg.get(sSurname, "xpath_LineColor")
        self._node = None        

    def __str__(self):
        s = "%s="%self.__class__
        s += "+(x1=%s y1=%s x2=%s y2=%s)" % (self.xpX1, self.xpY1, self.xpEvalX2, self.xpEvalY2)
        return s

    def draw(self, wxh, node):
        """draw itself using the wx handle
        return a list of created WX objects"""
#        print node.serialize()
#        print self.xpX
#        for n in node.xpathEval(self.xpX): print n.serialize()
        iLARGENEG = -9999
        lo = Deco.draw(self, wxh, node)
        
        if self._node != node: 
            self._x1 = self.xpathToInt(node, self.xpX1, iLARGENEG)
            self._y1 = self.xpathToInt(node, self.xpY1, iLARGENEG)
            
            #double evaluation, and a default value if necessary
            xpX2 = self.xpathToStr(node, self.xpEvalX2, '""')
            self._x2 = self.xpathToInt(node, xpX2, iLARGENEG, False) #do not show any error
            if self._x2 == iLARGENEG: self._x2 = self.xpathToInt(node, self.xpDfltX2, iLARGENEG)
            
            xpY2 = self.xpathToStr(node, self.xpEvalY2, '""')
            self._y2 = self.xpathToInt(node, xpY2, iLARGENEG, False) #do not show any error
            if self._y2 == iLARGENEG: self._y2 = self.xpathToInt(node, self.xpDfltY2, iLARGENEG)
            self._node = node
        
        if self._x1 != iLARGENEG and self._y1 != iLARGENEG and self._x2 != iLARGENEG and self._y2 != iLARGENEG:
            sLineColor = self.xpathToStr(node, self.xpLineColor, "#000000")
            iLineWidth = self.xpathToInt(node, self.xpLineWidth, 1)
            #draw a line
            obj = wxh.AddLine( [(self._x1, -self._y1), (self._x2, -self._y2)]
                                        , LineWidth=iLineWidth
                                        , LineColor=sLineColor)
            lo.append(obj)
        return lo
    
         
class DecoClickableRectangleSetAttr(DecoBBXYWH):
    """A rectangle
    clicking on it add/remove an attribute
    the rectangle color is indicative of the presence/absence of the attribute
    """
    
    def __init__(self, cfg, sSurname, xpCtxt):
        DecoBBXYWH.__init__(self, cfg, sSurname, xpCtxt)
        #now get the xpath expressions that let us find the rectangle line and fill colors
        self.xpLineColor = cfg.get(sSurname, "xpath_LineColor")
        self.xpLineWidth = cfg.get(sSurname, "xpath_LineWidth")
        self.xpFillColor = cfg.get(sSurname, "xpath_FillColor")
        self.xpFillStyle = cfg.get(sSurname, "xpath_FillStyle")        
        
        self.xpAttrName  = cfg.get(sSurname, "xpath_AttrName")        
        self.xpAttrValue = cfg.get(sSurname, "xpath_AttrValue")
        self.dInitialValue = {} 
        self.xpLineColorSlctd = cfg.get(sSurname, "xpath_LineColor_Selected")
        self.xpLineWidthSlctd = cfg.get(sSurname, "xpath_LineWidth_Selected")
        self.xpFillColorSlctd = cfg.get(sSurname, "xpath_FillColor_Selected")
        self.xpFillStyleSlctd = cfg.get(sSurname, "xpath_FillStyle_Selected")        
        
        
    def __str__(self):
        s = "%s="%self.__class__
        s += DecoBBXYWH.__str__(self)
        return s
    
    def isActionable(self):
        return True

    def draw(self, wxh, node):
        """draw itself using the wx handle
        return a list of created WX objects"""
        lo = DecoBBXYWH.draw(self, wxh, node)
        x,y,w,h,inc = self.runXYWHI(node)
        sAttrName  = self.xpathToStr(node, self.xpAttrName , None)
        sAttrValue = self.xpathToStr(node, self.xpAttrValue, None)
        if sAttrName and sAttrValue != None:
            if node.prop(sAttrName) == sAttrValue:
                sLineColor = self.xpathToStr(node, self.xpLineColorSlctd, "#000000")
                iLineWidth = self.xpathToInt(node, self.xpLineWidthSlctd, 1)
                sFillColor = self.xpathToStr(node, self.xpFillColorSlctd, "#000000")
                sFillStyle = self.xpathToStr(node, self.xpFillStyleSlctd, "Solid")
            else:            
                sLineColor = self.xpathToStr(node, self.xpLineColor, "#000000")
                iLineWidth = self.xpathToInt(node, self.xpLineWidth, 1)
                sFillColor = self.xpathToStr(node, self.xpFillColor, "#000000")
                sFillStyle = self.xpathToStr(node, self.xpFillStyle, "Solid")
            obj = wxh.AddRectangle((x, -y), (w, -h), 
                                                 LineWidth=iLineWidth,
                                                 LineColor=sLineColor,
                                                 FillColor=sFillColor,
                                                 FillStyle=sFillStyle)
            lo = [obj] + lo
        return lo
    
    def act(self, obj, node):
        """
        Toggle the attribute value
        """
        s = "do nothing"
        sAttrName  = self.xpathToStr(node, self.xpAttrName , None)
        sAttrValue = self.xpathToStr(node, self.xpAttrValue, None)
        if sAttrName and sAttrValue != None:
            try:
                initialValue = self.dInitialValue[node]
            except KeyError:
                initialValue = node.prop(sAttrName) #first time
                self.dInitialValue[node] = initialValue
            if node.get(sAttrName) == sAttrValue:
                #back to previous value
                if initialValue == None or initialValue == sAttrValue:
                    #very special case: when an attr was set, then saved, re-clicking on it wil remove it.
                    del node.attrib[sAttrName]
                    s = "Removal of @%s"%sAttrName
                else:
                    node.set(sAttrName, initialValue)
                    s = '@%s := "%s"'%(sAttrName,initialValue)
            else:
                if not sAttrValue:
                    del node.attrib[sAttrName]
                    s = "Removal of @%s"%sAttrName
                else:
                    node.set(sAttrName, sAttrValue)
                    s = '@%s := "%s"'%(sAttrName,sAttrValue)
        return s            
        
class DecoClickableRectangleJump(DecoBBXYWH):
    """A rectangle
    clicking on it jump to a node
    """
    
    def __init__(self, cfg, sSurname, xpCtxt):
        DecoBBXYWH.__init__(self, cfg, sSurname, xpCtxt)
        #now get the xpath expressions that let us find the rectangle line and fill colors
        self.xpLineColor = cfg.get(sSurname, "xpath_LineColor")
        self.xpLineWidth = cfg.get(sSurname, "xpath_LineWidth")
        self.xpFillColor = cfg.get(sSurname, "xpath_FillColor")
        self.xpFillStyle = cfg.get(sSurname, "xpath_FillStyle")    
        
        self.xp_xTo = cfg.get(sSurname, "xpath_xTo")
        self.xp_yTo = cfg.get(sSurname, "xpath_yTo")
        self.xp_wTo = cfg.get(sSurname, "xpath_wTo")
        self.xp_hTo = cfg.get(sSurname, "xpath_hTo")
        
        self.xpAttrToId  = cfg.get(sSurname, "xpath_ToId")        
        self.config = cfg.jl_hack_cfg #HACK
        
    def __str__(self):
        s = "%s="%self.__class__
        s += DecoBBXYWH.__str__(self)
        return s
    
    def isActionable(self):
        return True

    def draw(self, wxh, node):
        """draw itself using the wx handle
        return a list of created WX objects"""
        lo = DecoBBXYWH.draw(self, wxh, node)
        x,y,w,h,inc = self.runXYWHI(node)
        sLineColor = self.xpathToStr(node, self.xpLineColor, "#000000")
        iLineWidth = self.xpathToInt(node, self.xpLineWidth, 1)
        sFillColor = self.xpathToStr(node, self.xpFillColor, "#000000")
        sFillStyle = self.xpathToStr(node, self.xpFillStyle, "Solid")
        obj = wxh.AddRectangle((x, -y), (w, -h), LineWidth=iLineWidth,
                                                 LineColor=sLineColor,
                                                 FillColor=sFillColor,
                                                 FillStyle=sFillStyle)
        lo = [obj] + lo
        return lo
    
    def act(self, obj, node):
        """
        return the page number of the destination
        or None on error
        """
        sPageTag = self.config.getPageTag()
        sPageNumberAttr = self.config.getPageNumberAttr()
        number = None
        x,y,w,h = None, None, None, None
        bbHighlight = None
        sToId = self.xpathToStr(node, self.xpAttrToId , None)
        if sToId:
            ln = self.xpathEval(node.doc.getroot(), '//*[@id="%s"]'%sToId.strip())
            if ln:
                #find the page number
                ndTo = nd = ln[0]
                #while nd and nd.name != "PAGE": nd = nd.parent
                while nd and nd.name != sPageTag: nd = nd.parent
                try:
                    #number = max(0, int(nd.prop("number")) - 1)
                    number = max(0, self.xpathToInt(nd, sPageNumberAttr, 1, True) - 1)
                    
                    #maybe we can also indicate the precise arrival point?
                    if self.xp_xTo and self.xp_yTo and self.xp_hTo and self.xp_wTo:
                        x = self.xpathToInt(ndTo, self.xp_xTo, None)
                        y = self.xpathToInt(ndTo, self.xp_yTo, None)
                        w = self.xpathToInt(ndTo, self.xp_wTo, None)
                        h = self.xpathToInt(ndTo, self.xp_hTo, None)
                        if x==None or y==None or w==None or h==None:
                            x,y,w,h = None, None, None, None
                except:
                    pass
        return number,x,y,w,h
    

class DecoClickableRectangleJumpToPage(DecoBBXYWH):
    """A rectangle
    clicking on it jump to a page
    """
    
    def __init__(self, cfg, sSurname, xpCtxt):
        DecoBBXYWH.__init__(self, cfg, sSurname, xpCtxt)
        #now get the xpath expressions that let us find the rectangle line and fill colors
        self.xpLineColor = cfg.get(sSurname, "xpath_LineColor")
        self.xpLineWidth = cfg.get(sSurname, "xpath_LineWidth")
        self.xpFillColor = cfg.get(sSurname, "xpath_FillColor")
        self.xpFillStyle = cfg.get(sSurname, "xpath_FillStyle")    
        
        self.xpAttrToPageNumber  = cfg.get(sSurname, "xpath_ToPageNumber")        
        
    def __str__(self):
        s = "%s="%self.__class__
        s += DecoBBXYWH.__str__(self)
        return s
    
    def isActionable(self):
        return True

    def draw(self, wxh, node):
        """draw itself using the wx handle
        return a list of created WX objects"""
        lo = DecoBBXYWH.draw(self, wxh, node)
        x,y,w,h,inc = self.runXYWHI(node)
        sLineColor = self.xpathToStr(node, self.xpLineColor, "#000000")
        iLineWidth = self.xpathToInt(node, self.xpLineWidth, 1)
        sFillColor = self.xpathToStr(node, self.xpFillColor, "#000000")
        sFillStyle = self.xpathToStr(node, self.xpFillStyle, "Solid")
        obj = wxh.AddRectangle((x, -y), (w, -h), LineWidth=iLineWidth,
                                                 LineColor=sLineColor,
                                                 FillColor=sFillColor,
                                                 FillStyle=sFillStyle)
        lo = [obj] + lo
        return lo
    
    def act(self, obj, node):
        """
        return the page number of the destination
        or None on error
        """
        index,x,y,w,h = None,None,None,None,None
        sToPageNum = self.xpathToStr(node, self.xpAttrToPageNumber , None)
        if sToPageNum:
            index = int(sToPageNum) - 1
        return index,x,y,w,h
        
        
