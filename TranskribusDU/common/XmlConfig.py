"""
XmlConfig class used to read and write the XML configuration file 

Sophie Andrieu - October 2006

Copyright Xerox XRCE 2006

"""
from __future__ import absolute_import
from __future__ import  print_function
from __future__ import unicode_literals

import config.ds_xml_def as ds_xml
from lxml import  etree

sFalse = "False"
sTrue = "True"
sNodeDefault = ["NODEFAULT", "('NO', 'DEFAULT')"]
sBoolean = "boolean"

## The <code>XmlConfig</code> class is used to manage the XML configuration file
#@version 1.3
#@date October 2006
#@author Sophie Andrieu - Copyright Xerox XRCE 2006
#@see Parameter

class XmlConfig:
    
    confFileName = ""
    progName = ""
    version = ""
    description = ""
    #extConf = "_conf.xml"
    params = []
    
    def __init__(self, progName, vers, desc, fileName):
        self.progName = progName
        self.version = vers
        self.description = desc
        self.confFileName = fileName

    ## Write the XML configuration file with informations resulting <code>OptionParser</code>
    #@param options The list of options command line
    #@param parser The parser which store options and arguments
    def writeXmlConfig(self, dArgs, dicOptionDef=None):
        root = etree.Element(ds_xml.sCONFIGURATION)
        doc = etree.ElementTree(root)
        
        # TOOL tag
        tool = etree.SubElement(root, ds_xml.sTOOL)
        nameTool = etree.SubElement(tool, ds_xml.sNAME)
        nameTool.text =  self.progName
        versionTool = etree.SubElement(tool, ds_xml.sVERSION)
        versionTool.text =  self.version

        newComment = etree.Comment("""We describe here the command line argument:
  - @form is a space separated seris of option syntaxic form, e.g. '-i' for the 'input' option
  - @default is the default value
  - @help is a textual help for the user
  - @type is the data type (optparse has six built-in option types: string, int, long, choice, float and complex.)
  - @action is the expected action of the option parser, e.g. store, store_true, store_false (See http://docs.python.org/lib/optparse-standard-option-actions.html ) 
  - @metavar is "Stand-in for the option argument(s) to use when printing help text."
  See http://docs.python.org/lib/module-optparse.html understand fully the meaning of the attributes.
  Note @dest and @name are the same thing
""")
        root.append(newComment)

        descriptionTool = etree.SubElement(tool, ds_xml.sDESCRIPTION)
        descriptionTool.text = self.description

        lItems = dArgs.items()
#         lItems.sort()
        for k, v in sorted(lItems):
            param =etree.SubElement(root,ds_xml.sPARAM)
            param.text = repr(v)
            param.set(ds_xml.sATTR_NAME, k)
            if dicOptionDef:
                #add all the details we have!
                (tForms, dKW) = dicOptionDef[k]
#                 param.set(ds_xml.sATTR_FORM, string.joinfields(tForms)) #e.g. "-i --input"
                param.set(ds_xml.sATTR_FORM, str.join(" ",tForms)) #e.g. "-i --input"

                for k, v in dKW.items():
                    param.set(k, str(v)) #e.g. store="store_true"
        
        doc.write(self.confFileName,xml_declaration=True,encoding='UTF-8',pretty_print=True)
#         doc.saveFormatFile(self.confFileName, 1)
        del(doc)
       
    ## Read the XML configuration file and store all options inforations into a list of <code>Parameter</code> 
    #@see Parameter
    def readXmlConfig(self, dic):
        configXML = etree.parse(self.confFileName)
                
        lNode = self.getListNode(configXML, configXML.getroot(),"/%s/%s" %(ds_xml.sCONFIGURATION, ds_xml.sPARAM))
        for node in lNode:
            name = node.get(ds_xml.sATTR_NAME)
            value = eval(node.text, {}, {})
            
            dic[name] = value
                  
    ## Get the node list for a context
    #@param context The context to get the node list
    #@param nodeContext The node context
    #@param xpathQuery The Xpath query used to get the node list
    #@return the list of node
    def getListNode(self, context, nodeContext, xpathQuery):
#         ctxt = context.xpathNewContext()
#         ctxt.setContextNode(nodeContext)
#         list = ctxt.xpathEval(xpathQuery)
#         ctxt.xpathFreeContext()
        return nodeContext.xpath(xpathQuery)
#         return list

    