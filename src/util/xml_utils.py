#
# Various XML related utilities
#
# JL Meunier - May 2004
# SP Kruger - October 2005
#
# Copyright Xerox 2004
#
#
#from xml.sax import saxutils

################################
# minidom implementation
################################
## import xml.dom.minidom
## from xml import xpath
## def createDOM(root) :
##  	document = xml.dom.minidom.parseString("<"+root+"/>")
##  	return document

## def query(document, xpath) :
##	res = xpath.Evaluate(xpath, document):
## 	return res

## def addElement(document,root,elementName) :
##  	node = document.createElement(elementName)
##  	root.appendChild(node)
##  	return node

## def addTextElement(document,root,elementName,text) :
##  	text_node = addElement(document, root, elementName)
##  	try:
##  		text_data = document.createTextNode(saxutils.escape(text.encode("utf-8")))
##  	except UnicodeDecodeError :
##  		print "Error encoding to utf-8"
##  		text_data = document.createTextNode(saxutils.escape(text))
## 		text_node.appendChild(text_data)
## 		root.appendChild(text_node)
## 	return text_node

## def addAttribute(element, attName, attValue) :
##  	element.setAttribute(attName,attValue.encode("utf-8"))

## def getAttribute(element, attName) :
##  	element.getAttribute(attName)
	
## def toFile(document,fileName):
## 	try:
## 		f = file(fileName, 'w')
## 		document.writexml(f)
## 		f.close()
##  	except IOError:
##  		print("ERROR: %s cannot be opened for writing",fileName)

## def getRootElement(document):
## 	return document.documentElement

## def free(document):
##  	document.unlink()

## def escape(text):
##  	return saxutils.escape(text)

################################
# libxml2 implementation
################################
import libxml2
from __builtin__ import file
def createDOM(root) :
	document = libxml2.newDoc('1.0')
	documentroot = libxml2.newNode(root)
	document.setRootElement(documentroot)
	return document

def query(document, xpath) :
	ctxt = document.xpathNewContext()
	res = ctxt.xpathEval(xpath)
	ctxt.xpathFreeContext()
	return res

#xpath query relative to a node
def queryRel(document, node, xpath ):
	ctxt = document.xpathNewContext()
	ctxt.setContextNode( node )
	res =  ctxt.xpathEval( xpath )
	ctxt.xpathFreeContext()
	return res
		
def addElement(document,root,elementName) :
	newElement = libxml2.newNode(elementName)
	root.addChild(newElement)
	return newElement

def addComment(document, node, sComment):
	newComment =libxml2.newComment(sComment)
	node.addChild(newComment)
	return newComment

def delElement(elt):
	elt.unlinkNode()
	elt.freeNode()
	
def addTextElement(document,root,elementName,text) :
	newElement = libxml2.newNode(elementName)
	try :
		newElement.setContent(document.encodeEntitiesReentrant(text.encode("utf-8")))
	except UnicodeEncodeError :
		print "Error encoding to utf-81",text
		newElement.setContent(saxutils.escape(text))
	root.addChild(newElement)
	return newElement


def addTextToElement(document,root,text) :
	try :
		root.setContent(document.encodeEntitiesReentrant(text.encode("utf-8")))
	except UnicodeEncodeError :
		print "Error encoding to utf-8xx"
		root.setContent(saxutils.escape(text))

def addAttribute(element,attName,attValue) :
	try:
		element.setProp(attName,attValue.encode('utf-8'))
	except UnicodeEncodeError:
		print "Error encoding to utf-8 a "
		 
	
				
	
# 	try:
# 		element.setProp(attName,attValue.encode("utf-8"))
# 	except:
# 		print "Error encoding to utf-8a",attName,attValue,type(attValue),element.setProp(attName,attValue.encode("utf-8"))
# 		#print attName,attValue

def getAttribute(element,attName) :
	return element.prop(attName)



def toFile(doc, filename, bIndent=False):
# 		if self.bZLib:
# 			#traceln("ZLIB WRITE")
# 			try:
# 				FIX_docSetCompressMode(doc, self.iZLibRatio)
# 			except Exception, e:
# 				traceln("WARNING: ZLib error in Component.py: cannot set the libxml2 in compression mode. Was libxml2 compiled with zlib? :", e)
	if bIndent:
		doc.saveFormatFileEnc(filename, "UTF-8", bIndent)
	else: 
		#JLM - April 2009 - dump does not support the compressiondoc.dump(self.getOutputFile())
		doc.saveFileEnc(filename,"UTF-8")
	return filename
	
# def toFile(document,fileName):
# 	#document can be either a document or a node, in case we save each division separately
# 	#the pb is how to store a node i na file, since the dump is not provided for an xmlNode
# 	# and since the saveTo does not work... I rely on serialize for now... :-(
# 	
# 	#JL's add to save each division separately, we serialize from a node
# 	if isinstance(document, libxml2.xmlDoc):
# 		try:
# 			fOutFile = open(fileName,"w")
# 		except IOError:
# 			print("ERROR: %s cannot be opened for writing",fileName)
# 			sys.exit(1)
# 		document.dump(fOutFile)
# 		fOutFile.close()
# 	else:
# 		#DOES NOT WORK!! document.saveTo(fileName)
# 		try: #workaround
# 			fOutFile = open(fileName,"w")
# 		except IOError:
# 			print("ERROR: %s cannot be opened for writing",fileName)
# 			sys.exit(1)
# 		fOutFile.write('<?xml version="1.0"?>\n')
# 		fOutFile.write(document.serialize())
# 		fOutFile.close()
		
def serializeDocument(document, encoding = None, format = 0):
	return document.serialize(encoding, format)
	
def getRootElement(document):
    return document.getRootElement() 

def free(document):
	document.freeDoc()
	
#load an XML file and return its DOM
def loadXML(filepath):
	document = libxml2.parseFile(filepath)
	return document

################################
# xml neutral implementation
################################

def queryCount(document, xpath) :
	return len(query(document, xpath))

def escape(text):
	return saxutils.escape(text)

dicSerializer = {'\n':"&#10;",
                 '\r':"&#13;",
                 '\t':"&#9;" ,
                 '"':"&quot;" ,
                 '<':"&lt;" ,
                 '>':"&gt;" ,
                 '&':"&amp;" ,
                 }
	
def serializeText(sText):
    """Basic serialization of some text
    It's incomplete - should do more stuff, look at xmlsave.c of libxml2 to
    covince yourself. (xmlAttrSerializeTxtContent function)
    """
    s = ""
    for c in sText:
        try:
            s = s + dicSerializer[c]
        except KeyError:
            s = s + c
    return s


def deserializeText(sText):
    """Basic de-serialization of some text

    """
    s = sText
    for k, v in dicSerializer.items():
        s = s.replace(v, k)
    return s



#----------- SELF-TESTS ----------
if __name__ == "__main__":


    #--- tests of serializeText ---
    for k,v in dicSerializer.items():
        stpl = "toto%stutu"
        print stpl%v == serializeText(stpl%k)

    s = '> complique que d"hab & < simple\n'
    print serializeText(s) == "&gt; complique que d&quot;hab &amp; &lt; simple&#10;"
    print deserializeText("&gt; complique que d&quot;hab &amp; &lt; simple&#10;") == s
    
    
