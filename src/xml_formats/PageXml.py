# -*- coding: utf-8 -*-

'''
Created on 21 Nov 2016


Various utilities to deal with PageXml format

@author: meunier
'''
import os
import datetime

import libxml2


class PageXml:
    '''
    Various utilities to deal with PageXml format
    '''
    
    #Namespace for PageXml
    NS_PAGE_XML         = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
    
    NS_XSI ="http://www.w3.org/2001/XMLSchema-instance"
    XSILOCATION ="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd"  

    #Schema for Transkribus PageXml
    XSL_SCHEMA_FILENAME = "pagecontent.xsd"

    #XML schema loaded once for all
    cachedValidationContext = None  
    
    sMETADATA_ELT   = "Metadata"
    sCREATOR_ELT        = "Creator"
    sCREATED_ELT        = "Created"
    sLAST_CHANGE_ELT    = "LastChange"
    sCOMMENTS_ELT       = "Comments"

    sCUSTOM_ATTR = "custom"
    
    sEXT = ".pxml"

    # ---  Schema -------------------------------------            

    def validate(cls, doc):
        """
        Validate against the PageXml schema used by Transkribus
        
        Return True or False
        """
#         schDoc = cls.getSchemaAsDoc()
        if not cls.cachedValidationContext: 
            schemaFilename = cls.getSchemaFilename()
            buff = open(schemaFilename).read()
            prsrCtxt = libxml2.schemaNewMemParserCtxt(buff, len(buff))
            schema = prsrCtxt.schemaParse()
            cls.cachedValidationContext = schema.schemaNewValidCtxt()
            del buff, prsrCtxt

        res = cls.cachedValidationContext.schemaValidateDoc(doc)

        return res == 0
         
    validate = classmethod(validate)

    def getSchemaFilename(cls):
        """
        Return the path to the schema, built from the path of this module.
        """
        filename = os.path.join(os.path.dirname(__file__), cls.XSL_SCHEMA_FILENAME)
        return filename
    getSchemaFilename = classmethod(getSchemaFilename)
    
    # ---  Metadata  -------------------------------------
    """
    <complexType name="MetadataType">
        <sequence>
            <element name="Creator" type="string"></element>
            <element name="Created" type="dateTime">
                <annotation>
                    <documentation>The timestamp has to be in UTC (Coordinated Universal Time) and not local time.</documentation></annotation></element>
            <element name="LastChange" type="dateTime">
                <annotation>
                    <documentation>The timestamp has to be in UTC (Coordinated Universal Time) and not local time.</documentation></annotation></element>
            <element name="Comments" type="string" minOccurs="0"
                maxOccurs="1"></element>
        </sequence>
    </complexType>
    """
    
    def getMetadata(cls, doc=None, domNd=None):
        """
        Parse the metadata of the PageXml DOM or of the given Metadata node
        return a Metadata object
        """
        _, ndCreator, ndCreated, ndLastChange, ndComments = cls._getMetadataNodes(doc, domNd)
        return Metadata(ndCreator.getContent()
                        , ndCreated.getContent()
                        , ndLastChange.getContent()
                        , None if not ndComments else ndComments.getContent())
    getMetadata = classmethod(getMetadata)

    def setMetadata(cls, doc, domNd, Creator, Comments=None):
        """
        Pass EITHER a DOM or a Metadat DOM node!! (and pass None for the other)
        Set the metadata of the PageXml DOM or of the given Metadata node
        
        Update the Created and LastChange fields.
        Either update the Comments fields or delete it.
        
        You MUST indicate the Creator (a string)
        You MAY give a Comments (a string)
        The Created field is kept unchanged
        The LastChange field is automatically set.
        The Comments field is either updated or deleted.
        return the Metadata DOM node
        """
        ndMetadata, ndCreator, ndCreated, ndLastChange, ndComments = cls._getMetadataNodes(doc, domNd)
        ndCreator.setContent(Creator)
        #The schema seems to call for GMT date&time  (IMU)
        #ISO 8601 says:  "If the time is in UTC, add a Z directly after the time without a space. Z is the zone designator for the zero UTC offset."
        #Python seems to break the standard unless one specifies properly a timezone by sub-classing tzinfo. But too complex stuff
        #So, I simply add a 'Z' 
        ndLastChange.setContent(datetime.datetime.utcnow().isoformat()+"Z") 
        if Comments != None:
            if not ndComments: #we need to add one!
                ndComments = ndMetadata.newChild(None, cls.sCOMMENTS_ELT, Comments)
            ndComments.setContent(Comments)
        return ndMetadata
    setMetadata = classmethod(setMetadata)        
    
    # ---  Xml stuff -------------------------------------
    def getChildByName(cls, elt, sChildName):
        """
        look for all child elements having that name in PageXml namespace!!!
            Example: lNd = PageXMl.getChildByName(elt, "Baseline")
        return a DOM node
        """
        ctxt = elt.doc.xpathNewContext()
        ctxt.xpathRegisterNs("pc", cls.NS_PAGE_XML)  
        ctxt.setContextNode(elt)
        lNd = ctxt.xpathEval(".//pc:%s"%sChildName)
        ctxt.xpathFreeContext()
        return lNd
    getChildByName = classmethod(getChildByName)
    
    def getCustomAttr(cls, nd, sAttrName, sSubAttrName=None):
        """
        Read the custom attribute, parse it, and extract the 1st or 1st and 2nd key value
        e.g. getCustomAttr(nd, "structure", "type")     -->  "catch-word"
        e.g. getCustomAttr(nd, "structure")             -->  {'type':'catch-word', "toto", "tutu"} 
        return a dictionary if no 2nd key provided, or a string if 1st and 2nd key provided
        Raise KeyError is one of the attribute does not exist
        """
        ddic = cls.parseCustomAttr( nd.prop( cls.sCUSTOM_ATTR) )
        
        #First key
        dic2 = ddic[sAttrName]
        if sSubAttrName:
            return dic2[sSubAttrName]
        else:
            return dic2
    getCustomAttr = classmethod(getCustomAttr)

    def setCustomAttr(cls, nd, sAttrName, sSubAttrName, sVal):
        """
        Change the custom attribute by setting the value of the 1st+2nd key in the DOM
        return the value
        Raise KeyError is one of the attribute does not exist
        """
        ddic = cls.parseCustomAttr( nd.prop(cls.sCUSTOM_ATTR) )
        try:
            ddic[sAttrName][sSubAttrName] = str(sVal)
        except KeyError:
            ddic[sAttrName] = dict()
            ddic[sAttrName][sSubAttrName] = str(sVal)
            
        sddic = cls.formatCustomAttr(ddic)
        nd.setProp(cls.sCUSTOM_ATTR,sddic)
        return sVal
    setCustomAttr = classmethod(setCustomAttr)
    
    def parseCustomAttr(cls, s):
        """
        The custom attribute contains data in a CSS style syntax.
        We parse this syntax here and return a dictionary of dictionary
        
        Example:
        parseCustomAttr( "readingOrder {index:4;} structure {type:catch-word;}" )
            --> { 'readingOrder': { 'index':'4' }, 'structure':{'type':'catch-word'} }
        """
        dic = dict()
        
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
                    dicValForName[sKey.strip()] = sVal.strip()
                
                lName = sNames.split(',')
                for name in lName:
                    dic[name.strip()] = dicValForName
        return dic
    parseCustomAttr = classmethod(parseCustomAttr)
    
    def formatCustomAttr(cls, ddic):
        """
        Format a dictionary of dictionary of string in the "custom attribute" syntax 
        e.g. custom="readingOrder {index:1;} structure {type:heading;}"
        """
        s = ""
        for k1, d2 in ddic.items():
            if s: s += " "
            s += "%s"%k1
            s2 = ""
            for k2, v2 in d2.items():
                if s2: s2 += " "
                s2 += "%s:%s;"%(k2,v2)
            s += " {%s}"%s2
        return s
    formatCustomAttr = classmethod(formatCustomAttr)
        
        
    def makeText(cls, nd):
        """
        build the text of a sub-tree by considering that textual nodes are tokens to be concatenated, with a space as separator
        return None if no textual node found
        """
        ctxt = nd.doc.xpathNewContext()
        ctxt.setContextNode(nd)
        lnText = ctxt.xpathEval('.//text()')
        s = None
        for ntext in lnText:
            stext = ntext.content.strip()
            try:
                if stext: s = s + " " + stext
            except TypeError:
                s = stext
        ctxt.xpathFreeContext()
        return s
    makeText = classmethod(makeText)


    def addPrefix(cls, sPrefix, nd, sAttr="id"):
        """
        Utility to add a addPrefix to a certain attribute of a sub-tree.

        By default works on the 'id' attribute
                
        return the number of modified attributes
        """
        sAttr = sAttr.strip()
        ctxt = nd.doc.xpathNewContext()
        ctxt.setContextNode(nd)
        lAttrNd = ctxt.xpathEval(".//@%s"%sAttr)
        ret = len(lAttrNd)
        for ndAttr in lAttrNd:
            sNewValue = sPrefix+ndAttr.getContent().decode('utf-8')
            ndAttr.setContent(sNewValue.encode('utf-8)'))
        ctxt.xpathFreeContext()   
        return ret
    addPrefix = classmethod(addPrefix)
                
    def rmPrefix(cls, sPrefix, nd, sAttr="id"):
        """
        Utility to remove a addPrefix from a certain attribute of a sub-tree.

        By default works on the 'id' attribute
                
        return the number of modified attributes        
        """
        sAttr = sAttr.strip()
        ctxt = nd.doc.xpathNewContext()
        ctxt.setContextNode(nd)
        lAttrNd = ctxt.xpathEval(".//@%s"%sAttr)
        n = len(sPrefix)
        ret = len(lAttrNd)
        for ndAttr in lAttrNd:
            sValue = ndAttr.getContent().decode('utf-8')
            assert sValue.startswith(sPrefix), "Prefix '%s' from attribute '@%s=%s' is missing"%(sPrefix, sAttr, sValue)
            sNewValue = sValue[n:]
            ndAttr.setContent(sNewValue.encode('utf-8)'))
        ctxt.xpathFreeContext()   
        return ret
    rmPrefix = classmethod(rmPrefix)

    def _getMetadataNodes(cls, doc=None, domNd=None):
        """
        Parse the metadata of the PageXml DOM or of the given Metadata node
        return a 4-tuple:
            DOM nodes of Metadata, Creator, Created, Last_Change, Comments (or None if no COmments)
        """
        assert bool(doc) != bool(domNd), "Internal error: pass either a DOM or a Metadata node"  #XOR
        if doc:
            lNd = cls.getChildByName(doc.getRootElement(), cls.sMETADATA_ELT)
            if len(lNd) != 1: raise ValueError("PageXml should have exactly one %s node"%cls.sMETADATA_ELT)
            domNd = lNd[0]
            assert domNd.name == cls.sMETADATA_ELT
        nd1 = domNd.firstElementChild()
        if nd1.name != cls.sCREATOR_ELT: raise ValueError("PageXMl mal-formed Metadata: Creator element must be 1st element")
        nd2 = nd1.nextElementSibling()
        if nd2.name != cls.sCREATED_ELT: raise ValueError("PageXMl mal-formed Metadata: Created element must be 2nd element")
        nd3 = nd2.nextElementSibling()
        if nd3.name != cls.sLAST_CHANGE_ELT: raise ValueError("PageXMl mal-formed Metadata: LastChange element must be 3rd element")
        nd4 = nd3.nextElementSibling()
        if nd4:
            if nd4.name != cls.sCOMMENTS_ELT: raise ValueError("PageXMl mal-formed Metadata: LastChange element must be 3rd element")
        return domNd, nd1, nd2, nd3, nd4
    _getMetadataNodes = classmethod(_getMetadataNodes)

    # ---  Geometry -------------------------------------            
    def getPointList(cls, data):
        """
        get either an XML node of a PageXml object
              , or the content of a points attribute
        
        return the list of (x,y) of the polygone of the object - ( it is a list of int tuples)
        """
        try:
            lsPair = data.split(' ')
        except:
            ctxt = data.doc.xpathNewContext()
            ctxt.xpathRegisterNs("pc", cls.NS_PAGE_XML)  
            ctxt.setContextNode(data)
            lndPoints = ctxt.xpathEval("(.//@points)[1]")  #no need to collect all @points below!
            sPoints = lndPoints[0].getContent()
            lsPair = sPoints.split(' ')
            ctxt.xpathFreeContext()
        lXY = list()
        for sPair in lsPair:
            (sx,sy) = sPair.split(',')
            lXY.append( (int(sx), int(sy)) )
        return lXY
    getPointList = classmethod(getPointList)


    def setPoints(cls, nd, lXY):
        """
        set the points attribute of that node to reflect the lXY values
        if nd is None, only returns the string that should be set to the @points attribute
        return the content of the @points attribute
        """
        sPairs = " ".join( ["%d,%d"%(int(x), int(y)) for x,y in lXY] )
        if nd: nd.setProp("points", sPairs)
        return sPairs
    setPoints = classmethod(setPoints)

    def getPointsFromBB(cls, x1,y1,x2,y2):
        """
        get the polyline of this bounding box
        return a list of int tuples
        """
        return [ (x1,y1), (x2,y1), (x2,y2), (x1,y2), (x1,y1) ]
    getPointsFromBB = classmethod(getPointsFromBB)
        
        
        
    @classmethod
    # --- Creation -------------------------------------
    def createPageXmlDocument(cls,creatorName='XRCE',filename=None,imgW=0, imgH=0):
        """
            create a new PageXml document
        """
        xmlPageDoc = libxml2.newDoc("1.0")
    
        xmlPAGERoot = libxml2.newNode('PcGts')
        xmlPageDoc.setRootElement(xmlPAGERoot)
        pagens = xmlPAGERoot.newNs(cls.NS_PAGE_XML,None)
        xmlPAGERoot.setNs(pagens)
        XSINs = xmlPAGERoot.newNs(cls.NS_XSI, "xsi")
        xmlPAGERoot.setNsProp(XSINs,"schemaLocation",cls.XSILOCATION)    
        
        
        metadata= libxml2.newNode(cls.sMETADATA_ELT)
        metadata.setNs(pagens)
        xmlPAGERoot.addChild(metadata)
        creator=libxml2.newNode(cls.sCREATOR_ELT)
        creator.setNs(pagens)
        creator.addContent(creatorName)
        created=libxml2.newNode(cls.sCREATED_ELT)
        created.setNs(pagens)
        created.addContent(datetime.datetime.now().isoformat())
        lastChange=libxml2.newNode(cls.sLAST_CHANGE_ELT)
        lastChange.setNs(pagens)
        lastChange.setContent(datetime.datetime.utcnow().isoformat()+"Z") 
        metadata.addChild(creator)
        metadata.addChild(created)
        metadata.addChild(lastChange)
        
        
        pageNode= libxml2.newNode('Page')
        pageNode.setNs(pagens)
        pageNode.setProp('imageFilename',filename )
        pageNode.setProp('imageWidth',str(imgW))
        pageNode.setProp('imageHeight',str(imgH))
    
        xmlPAGERoot.addChild(pageNode)
        
        bValidate = cls.validate(xmlPageDoc)
        assert bValidate, 'new file not validated by schema'
        
        return xmlPageDoc, pageNode
    
    @classmethod
    def createPageXmlNode(cls,nodeName,ns):
        """
            create a PageXMl element
        """
        node=libxml2.newNode(nodeName)
        
        #ns
        node.setNs(ns)        

        return node
    
       
            
# ---  Multi-page PageXml -------------------------------------            
            
class MultiPageXml(PageXml):          
    XSL_SCHEMA_FILENAME = "multipagecontent.xsd"
    sEXT = ".mpxml"
    
    
    @classmethod
    def makeMultiPageXmlMemory(cls,lDom):
        """
            create a MultiPageXml from a list of dom PageXml
        """
        
        assert lDom, "ERROR: empty list of DOM PageXml"
        pnum = 1
        doc = lDom.pop(0)
        rootNd = doc.getRootElement()
        #Let's addPrefix all IDs with a page number...
        cls.addPrefix("p%d_"%pnum, rootNd, "id")
        
        while lDom:
            pnum += 1
            _doc = lDom.pop(0)
            _rootNd = _doc.getRootElement()
            assert _rootNd.name == "PcGts", "Data error: expected a root element named 'PcGts' in %d th dom" %pnum

            ndChild = _rootNd.children
            sPagePrefix = "p%d_"%pnum
            while ndChild:
                if ndChild.type == "element": 
                    cls.addPrefix(sPagePrefix, ndChild, "id")
                rootNd.addChild(ndChild.copyNode(1))  #1=recursive copy (properties, namespaces and children when applicable)
                ndChild = ndChild.next 
            _doc.freeDoc()
        
        return doc
        
        
        
    def makeMultiPageXml(cls, lsXmlDocFilename):
        """
        We concatenate sequence of PageXml files into a multi-page (non-standard) PageXml
        
        Take a list of filenames,
        return a DOM
        """
        assert lsXmlDocFilename, "ERROR: empty list of filenames"
        
        pnum = 1
        sXmlFile = lsXmlDocFilename.pop(0)
        doc = libxml2.parseFile(sXmlFile)
        rootNd = doc.getRootElement()
        #Let's addPrefix all IDs with a page number...
        cls.addPrefix("p%d_"%pnum, rootNd, "id")
        
        while lsXmlDocFilename:
            pnum += 1
            sXmlFile = lsXmlDocFilename.pop(0)
            _doc = libxml2.parseFile(sXmlFile)
            _rootNd = _doc.getRootElement()
            assert _rootNd.name == "PcGts", "Data error: expected a root element named 'PcGts' in %s"%sXmlFile

            ndChild = _rootNd.children
            sPagePrefix = "p%d_"%pnum
            while ndChild:
                if ndChild.type == "element": 
                    cls.addPrefix(sPagePrefix, ndChild, "id")
                rootNd.addChild(ndChild.copyNode(1))  #1=recursive copy (properties, namespaces and children when applicable)
                ndChild = ndChild.next 
            _doc.freeDoc()
        
        return doc
    makeMultiPageXml = classmethod(makeMultiPageXml)

    def splitMultiPageXml(cls, doc, sToDir, sFilenamePattern, bIndent=False, bInPlace=True):
        """
        Split a multipage PageXml into multiple PageXml files
        
        Take a folder name and a filename pattern containing a %d
        
        if bInPlace, the input doc is split in-place, to this function modifies the input doc, which must no longer be used by the caller.
        
        PROBLEM: 
            We have redundant declaration of the default namespace. 
            I don't know how to clean them, ax xmllint does with its --nsclean option.
        
        return a list of filenames
        """
        lXmlFilename = list()
        
        if not( os.path.exists(sToDir) and os.path.isdir(sToDir)): raise ValueError("%s is not a folder"%sToDir)
        
        for pnum, newDoc in cls._iter_splitMultiPageXml(doc, bInPlace):
            #dump the new XML into a file in target folder
            name = sFilenamePattern%pnum
            sFilename = os.path.join(sToDir, name)
            newDoc.saveFormatFileEnc(sFilename, "UTF-8", bIndent)
            lXmlFilename.append(sFilename)

        return lXmlFilename
    splitMultiPageXml = classmethod(splitMultiPageXml)

    # ---  Metadata  -------------------------------------
    def getMetadata(cls, doc=None, lDomNd=None):
        """
        Parse the metadata of the MultiPageXml DOM or of the given Metadata nodes
        return a list of Metadata object
        """
        lDomNd = cls._getMetadataNodeList(doc, lDomNd)
        return [PageXml.getMetadata(None, domNd) for domNd in lDomNd]
    getMetadata = classmethod(getMetadata)

    def setMetadata(cls, doc, lDomNd, Creator, Comments=None):
        """
        Pass EITHER a DOM or a Metadata DOM node list!! (and pass None for the other)
        Set the metadata of the PageXml DOM or of the given Metadata node
        
        Update the Created and LastChange fields.
        Either update the Comments fields or delete it.
        
        You MUST indicate the Creator (a string)
        You MAY give a Comments (a string)
        The Created field is kept unchanged
        The LastChange field is automatically set.
        The Comments field is either updated or deleted.
        return the Metadata DOM node
        """
        lDomNd = cls._getMetadataNodeList(doc, lDomNd)
        return [PageXml.setMetadata(None, domNd, Creator, Comments) for domNd in lDomNd]
    setMetadata = classmethod(setMetadata)        

    # ---  Internal  ------------------------------
    def _getMetadataNodeList(cls, doc=None, lDomNd=None):
        """
        Return the list of Metadata node
        return a non-empty list of DOM nodes 
        """
        assert bool(doc) != bool(lDomNd), "Internal error: pass either a DOM or a Metadata node list"  #XOR
        if doc:
            lDomNd = cls.getChildByName(doc.getRootElement(), cls.sMETADATA_ELT)
            if not lDomNd: raise ValueError("PageXml should have at least one %s node"%cls.sMETADATA_ELT)
        return lDomNd
    _getMetadataNodeList = classmethod(_getMetadataNodeList)
    
    def _iter_splitMultiPageXml(cls, doc, bInPlace=True):
        """
        iterator that splits a multipage PageXml into multiple PageXml DOM
        
        Take a MultiPageXMl DOM
        
        Yield a tupe (<pnum>, DOM)  for each PageXMl of each page. pnum is an integer in [1, ...]
        
        those DOMs are automatically freed at end of iteration
        
        if bInPlace, the input doc is split in-place, to this function modifies the input doc, which must no longer be used by the caller.
        
        PROBLEM: 
            We have redundant declaration of the default namespace. 
            I don't know how to clean them, ax xmllint does with its --nsclean option.
        
        yield DOMs
        """
        rootNd = doc.getRootElement()
        
        ctxt = doc.xpathNewContext()
        ctxt.xpathRegisterNs("a", cls.NS_PAGE_XML)
        ctxt.setContextNode(rootNd)
        
        lMetadataNd = ctxt.xpathEval("/a:PcGts/a:Metadata")
        if not lMetadataNd: raise ValueError("Input multi-page PageXml should have at least one page and therefore one Metadata element")
        
        lDocToBeFreed = []
        pnum = 0
        for metadataNd in lMetadataNd:
            pnum += 1
            
            #create a DOM
            newDoc = libxml2.newDoc("1.0")
            newRootNd = rootNd.copyNode(2) #2 copy properties and namespaces (when applicable)
            newDoc.setRootElement(newRootNd)
            
            #to jump to the PAGE sibling node (we do it now, defore possibly unlink...)
            node = metadataNd.next

            #Add a copy of the METADATA node and sub-tree
            if bInPlace:
                metadataNd.unlinkNode()
                newRootNd.addChild(metadataNd)
            else:
                newMetadataNd = metadataNd.copyNode(1)
                newRootNd.addChild(newMetadataNd)
            
#             #jump to the PAGE sibling node
#             node = metadataNd.next
            while node:
                if node.type == "element": break
                node = node.next
            if node.name != "Page": raise ValueError("Input multi-page PageXml for page %d should have a PAGE node after the METADATA node."%pnum)
            
            #Add a copy of the PAGE node and sub-tree
            if bInPlace:
                node.unlinkNode()
                newNode = newRootNd.addChild(node)
            else:
                newPageNd = node.copyNode(1)
                newNode = newRootNd.addChild(newPageNd)
           
            #Remove the prefix on the "id" attributes
            sPagePrefix = "p%d_"%pnum
            nb = cls.rmPrefix(sPagePrefix, newNode, "id")
            
            newRootNd.reconciliateNs(newDoc)
            
            yield pnum, newDoc

            lDocToBeFreed.append(newDoc)
#             newDoc.freeDoc()
            
        ctxt.xpathFreeContext()
        for doc in lDocToBeFreed: doc.freeDoc()
           
        raise StopIteration
    _iter_splitMultiPageXml = classmethod(_iter_splitMultiPageXml)

# ---  Metadata of PageXml  --------------------------------            
class Metadata:
    
    """
    <complexType name="MetadataType">
        <sequence>
            <element name="Creator" type="string"></element>
            <element name="Created" type="dateTime">
                <annotation>
                    <documentation>The timestamp has to be in UTC (Coordinated Universal Time) and not local time.</documentation></annotation></element>
            <element name="LastChange" type="dateTime">
                <annotation>
                    <documentation>The timestamp has to be in UTC (Coordinated Universal Time) and not local time.</documentation></annotation></element>
            <element name="Comments" type="string" minOccurs="0"
                maxOccurs="1"></element>
        </sequence>
    </complexType>
    """
    
    def __init__(self, Creator, Created, LastChange, Comments=None):
        self.Creator    = Creator           # a string
        self.Created    = Created           # a string
        self.LastChange = LastChange        # a string
        self.Comments   = Comments          #None or a string
        
    
    
