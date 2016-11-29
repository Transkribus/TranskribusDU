# -*- coding: utf-8 -*-

'''
Created on 21 Nov 2016


Various utilities to deal with PageXml format

@author: meunier
'''
import os

import libxml2

class PageXml:
    '''
    Various utilities to deal with PageXml format
    '''
    
    #Namespace for PageXml
    NS_PAGE_XML         = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
    
    #Schema for Transkribus PageXml
    XSL_SCHEMA_FILENAME = "pagecontent.xsd"

    #XML schema loaded once for all
    cachedValidationContext = None  

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
    
    
    # ---  PageXml -------------------------------------            
    def parse_custom_attr(cls, s):
        """
        The custom attribute contains data in a CSS style syntax.
        We parse this syntax here and return a dictionary of dictionary
        
        Example:
        parse_custom_attr( "readingOrder {index:4;} structure {type:catch-word;}" )
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
    parse_custom_attr = classmethod(parse_custom_attr)
    



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


# ---  Multi-page PageXml -------------------------------------            
            
class MultiPageXml(PageXml):          
    XSL_SCHEMA_FILENAME = "multipagecontent.xsd"
    
    def makeMultiPageXml(cls, lsXmlDocFilename):
        """
        We concatenate sequence of PageXml files into a multi-page (non-standard) PageXml
        
        Take a list of filenames,
        return a DOM
        """
        assert lsXmlDocFilename, "ERROR: empty list of filenames"
        
        pnum = 1
        sXmlFile = lsXmlDocFilename.pop()
        doc = libxml2.parseFile(sXmlFile)
        rootNd = doc.getRootElement()
        #Let's addPrefix all IDs with a page number...
        cls.addPrefix("p%d_"%pnum, rootNd, "id")
        
        while lsXmlDocFilename:
            pnum += 1
            sXmlFile = lsXmlDocFilename.pop()
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
            
            #dump the new XML into a file in target folder
            name = sFilenamePattern%pnum
            sFilename = os.path.join(sToDir, name)
            newDoc.saveFormatFileEnc(sFilename, "UTF-8", bIndent)
            lXmlFilename.append(sFilename)

            lDocToBeFreed.append(newDoc)
#             newDoc.freeDoc()
            
        ctxt.xpathFreeContext()
        for doc in lDocToBeFreed: doc.freeDoc()
           
        return lXmlFilename
    splitMultiPageXml = classmethod(splitMultiPageXml)

