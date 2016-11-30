#
# XML definitions for the DS software
# 
# JL Meunier - May 2004
#
# Copyright XRCE, 2004
#


#Note: there is a kind of variable naming convention here:
# a variable named sV is supposed to contain a string
# lV contains a list
# tV contains a tuple
# iV contains an integer
# fV contains a float
# Sometime composition can occur, like lsV for a list of strings 


## pdftoxml tag ames
sFontName = "font-name"
sFontSize =  "font-size"
sFontColor = "font-color"
sBold = "bold"
sItalic = "italic"

sRotation = "rotation"


#DS namespace: name and URI
# we forget about the DS namespace
## sDSNs    = "DS"
## sDSNsURI = "http://www.xrce.xerox.com/DS"
sDSNs    = 99.99       #to detect its usage
sDSNsURI = 99999.99999 #to detect its usage


sSVGNs = "svg"
sSVGNsURI = "http://www.w3.org/2000/svg"

#For uniquely identifying node
#sIdAttr = "Id"
sIdAttr = "id"

#For the TOC detector
sTocElt        = "Toc"      #a Toc tag name to encapsulate the TOC region
sTocLinkElt    = "TocLink"  #tag to encapsulate a TOC entry
sTocToAttr     = "TocTo"       #an attribute name for the TocLink node
sTocListToAttr = "ToclTo"      #an attribute name for the TocLink node
sTocText		= "TocLinkText"
sTocLevelAttr     = "TocLevel" #an attr name to mark a node referred by the TOC
sTocLevelValNone  = "None"     #None when the TOC hasn't been structured yet
sTocPARAM      = "TocPARAM"

#For the Header/Footer detector
sHF_HeaderAttr = "header"
sHF_FooterAttr = "footer"
sHF_SideAttr = "sider"
sHFPARAM ="HFPARAM"
sHFType = "HFType"

#Footnotes
sNoteAttr = "Note"
sNoteAnchorAttr= "NoteAnchor"

#For the textbox clusterer
sPagAttr      = "Pag"

#For the xy_cutter segmenter and reorderer
sXY_COL_Elt          = "COLUMN"       #id like "cPP_NN"
sXY_PARAG_Elt        = "PARAGRAPH"    #id like "pPP_NN"
sXY_LINE_Elt         = "LINE"         #id like "lPP_NN"
sXYPARAM             = "XYPARAM"
sXY_LineSpacing_Attr = "lineSpace"
#-- the 3 below are obsolete with xy_orderer
sXY_PrefTxt_Attr     = "prefTxt"
sXY_Prefix_Attr      = "prefix"
sXY_IvecoWarn_Attr   = "IvecoWarn"
#obsolete sXY_newLine_Attr     = "newLine"
#obsolete sXY_endOfLine_Attr   = "endOfLine"


sTypeArea   = "typeArea"
sXCal       = "xcal"
sYCal       = "ycal"
sMainY      = "YAxis"
sMainX      = "XAxis"


#--- PageNumber
#The attributes put by the PageNum detector
ks_pcnt 	= "pcnt"      #the page index in the current sub-sequence, in [1..N] with N <= doc_page_number
ks_pnum_flg = "pnum_flg"  #indicates that this node is a detected page number
ks_pnum 	= "pnum"      #the page number string
ks_pnum_ext = "pnum_ext"  #the page number string, normalized and possibly extrapolated fro mthe sequence

ks_pnum_extref = "pnum_ref-ext" #indicate which pnum_ext appeared here
ks_pnum_ref = "pnum_ref" #indicate the occurence of a page number referring to the Ith page, I being in [1...doc_page_number]
ks_pnum_seq = "pnum_seq" #this reference is part of a sequence (doc- or page-sequence)]
ks_pnum_pageseq = "pnum_pageseq" #same plus this pnum_ref is part of the longest non-strictly increasing sequence of reference on this page, same value
ks_pnum_docseq  = "pnum_docseq"  #same plus this pnum_ref is part of the longest non-strictly increasing sequence of reference in the document, same value


sTOKEN = "TOKEN"
sTEXT = "TEXT"
sPAGE = "PAGE"
sX = "x"
sY = "y"
sHeight = "height"
sWidth = "width"
sPageNumber = "number"
sXY_IMAGE_Elt = "IMAGE"
sXY_GROUP_Elt = "GROUP"
sGALLEY       = "GALLEY"
sGraphicalElt ="GRAPHELT"
sBaseline = "base"

sBLOCK = "BLOCK"
sGBLOCK = "GBLOCK"
sTABLE = "TABLE"
sROW   = "ROW"
sCELL  = "CELL"
sFRAME = "FRAME"
sGLINE ="GLINE"  # for graphical line

sUnderline = "underline"
## for Alignment
sAlign_ELT = "text-align"
sAlign_Justified = "justified"
sAlign_Center = "center"
sAlign_Left = "left"
sAlign_Right = "right"
# alignment wrt parent
sAlign_Parent = "position"

sIndent = "Indent"
sHanging = "Hanging"
sTAB = "TAB"

sCaption = "caption"
sIllustration  ="Illustration"

#For saliency/normalElement
sNormalElement = "normalElement"
sSaliency = "saliency"

## for break
sBreak = "BREAK"
sPageBreak="PageBreak"
sColBreak ="ColBreak"
#sHardBreak = "hard"
#sSoftBreak = "soft"
sHeadBreak = "head"
sTailBreak = "tail"
sSequenceItem = "ITEM"
sItemFather = "itemFather"
#sSequenceItem = "SequenceItem"
sLIST = "LIST"
sLISTITEM = "LISTITEM" # deprecated?
sPrevItem = "prevItem"
sNextItem = "nextItem"
sequenceItemHead = "head"
#sequenceItemHead = "itemHead"
sPattern ="sequencePattern"
sHBreakPARAM = "HBREAK"
sSBreakPARAM = "SBREAK"




# XML tag used in XML file configuration for each tool
sCONFIGURATION = "CONFIGURATION"
sTOOL = "TOOL"
sNAME = "NAME"
sVERSION = "VERSION"
sDESCRIPTION = "DESCRIPTION"
sPARAM = "PARAM"
sCMD = "CMD"
sOPTIONSHORT = "OPTIONSHORT"
sOPTIONLONG = "OPTIONLONG"
sARG = "ARG"
sTYPE = "TYPE"
sDEFAULTVALUE = "DEFAULTVALUE"
sATTR_NAME = "name"
sATTR_FORM = "form"

sXML_VERSION = "1.0"
sENCODING_UTF = "UTF-8"

sDOCUMENT = "DOCUMENT"
sMETADATA = "METADATA"
sPROCESS = "PROCESS"
sVERSION = "VERSION"
sCOMMENT = "COMMENT"
sCREATIONDATE = "CREATIONDATE"
sATTR_CMD = "cmd"
sATTR_CMD_ARG = "cmd_arg"
sATTR_VALUE = "value"

#XML tags and attributes used in XML outline file generated by pdftoxml
sTOCITEMS = "TOCITEMS" 
sTOCITEMLIST = "TOCITEMLIST"
sITEM = "ITEM"
sSTRING = "STRING"
sLINK = "LINK"

sATTR_NBPAGES = "nbPages"
sATTR_LEVEL = "level"
sATTR_IDITEMPARENT = "idItemParent"
sATTR_PAGE = "page"
sATTR_TOP = "top"
sATTR_BOTTOM = "bottom"
sATTR_LEFT = "left"
sATTR_RIGHT = "right"

#---- Utility

import libxml2

#I know it should not be here but it's so convenient!

#Register the DS namespace on the root node of a document
#   (and deal with the case it is already registered) 
#Return the libxml2 namespace object or None on error
def registerDSNameSpace(doc):
    root = doc.getRootElement()
    nsDs = None
    try:
        nsDS = root.newNs(sDSNsURI, sDSNs)
    except libxml2.treeError:
        try:
            nsDS = doc.searchNsByHref(root, sDSNsURI)
        except libxml2.treeError, s:
            return nsDS
    if not nsDS:
        nsDS = None
    return nsDS
