# -*- coding: utf-8 -*-

"""
Typically for use with ABP tables, to match the GT documents with their HTRed 
    counterpart.

We have:
- an input  collection obtained by downloading a Transkribus collection using 
    (PyClient) Transkribus_downloader.py
- a GT collection containing the definition of areas in each page. (Can be 
    table cells, or menu region, or whatever)

We want 
1 - to generate a new document, where the "elements of interest" (e.g TextLine) 
    of the input collection are matched against the GT areas by the location, 
    so that each element is either inserted in an area that matches or left 
    outside any area.
2 - (optionnally) to normalize the bounding area of the "element of interest"
    This is done by making a box of predefined height from the Baseline
    
Generate a new collection, with input documents enriched with GT areas.

Any input document without GT counterpart is ignored.

Created on 23 août 2019

Copyright NAVER LABS Europe 2019
@author: JL Meunier
"""

import sys, os
from optparse import OptionParser
from copy import deepcopy
from collections import defaultdict

from lxml import etree
from numpy import argmax as argmax
from shapely.affinity import translate   

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version
TranskribusDU_version

from common.trace import traceln, trace
from util.Shape import ShapeLoader as ShapeLoader

# ----------------------------------------------------------------------------
iNORMALIZED_HEIGHT  = 43
xpELEMENT1          = ".//pg:TextRegion"
xpELEMENT2          = ".//pg:TextLine"

xpAREA1             = ".//pg:TableRegion"
xpAREA2             = ".//pg:TableCell"

xpBASELINE          = ".//pg:Baseline"
dNS = {"pg":"http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
# ----------------------------------------------------------------------------

def main(sInputDir, sGTDir, sOutputDir
         , xpElement1, xpElement2
         , xpArea1, xpArea2
         , bNorm, iNorm, bNormOnly
         , bSep
         , lsRmId
         , bEval
         , bWarm
         , sExt = ".mpxml"
         , bVerbose=False):
    
    lSkippedFile = []
    
    # filenames without the path
    lsFilename = [os.path.basename(name) for name in os.listdir(sInputDir) if name.endswith(sExt) and not name.endswith("_du%s"%sExt)]
    traceln(" - %d %s files to process" % (len(lsFilename), sExt))
    for sMPXml in lsFilename:
        trace(" - %s FILE : " % sExt, sMPXml)
        if bVerbose: traceln()
        
        # -- find individual subfiles
        sSubDir = os.path.join(sInputDir, sMPXml[:-len(sExt)])
        if os.path.isdir(sSubDir):
            traceln("  (->  ", sSubDir, ")")
            lsPXml = [os.path.basename(name) for name in os.listdir(sSubDir) if name.endswith(".pxml")]
            if bVerbose: traceln("\t%d files to process"%len(lsPXml))
        else:
            sSubDir = sInputDir
            lsPXml = [sMPXml]
            if bVerbose: traceln("\tprocessing the %s file"%sExt)
        
        # -- find GT...
        for sInputXml in lsPXml:
            trace("\t", sMPXml, " -- ", sInputXml)
                
            sGTFN = os.path.join(sGTDir, sInputXml)
            if not os.path.isfile(sGTFN):
                # maybe it is also a folder downloaded from Transkribus?
                if os.path.isfile(os.path.join(sGTDir, sMPXml[:-len(".mpxml")], sInputXml)):
                    sGTFN = os.path.join(sGTDir, sMPXml[:-len(".mpxml")], sInputXml)
                else:
                    # hummm, maybe it is a mpxml instead... :-/
                    sGTFN = sGTFN[:-len(".pxml")] + ".mpxml"
                    if not os.path.isfile(sGTFN):
                        traceln("  *** NO GT *** file skipped ")
                        lSkippedFile.append(sInputXml)
                        continue
            # ok GT file found
            trace(" ...")

            # input Xml
            sInFN = os.path.join(sSubDir, sInputXml)
            sOutFN = os.path.join(sOutputDir, sInputXml)

            if bWarm and os.path.exists(sOutFN):
                # check existence and freshness
                t_in  = os.path.getmtime(sInFN)
                t_gt  = os.path.getmtime(sGTFN)
                t_out =  os.path.getmtime(sOutFN)
                if t_out > t_in and t_out > t_gt:
                    traceln("\t\t fresh output file found on disk: %s  - skipping it!"%sOutFN)
                    continue
            
            # 0 - load input file
            doc = etree.parse(sInFN)
            
            # 1 - normalize input elements
            if bNorm: 
                doc = normaliseDocElements(doc, xpElement2, iNorm)
            
            # 2 - project GT
            if not bNormOnly:
                gtdoc = etree.parse(sGTFN)
                if True:
                    doc = project_Elt_to_GT(gtdoc, doc
                                            , xpElement1, xpElement2
                                            , xpArea2, bSep, lsRmId, bEval)
                else:
                    doc = project_Areas_to_Input(gtdoc, doc
                                                 , xpElement1, xpElement2, xpArea1, xpArea2
                                                 , bSep, lsRmId, bEval)
            
            # 3 - save
            doc.write(sOutFN,
                      xml_declaration=True,
                      encoding="utf-8",
                      pretty_print=True
                      #compression=0,  #0 to 9
                      )        
                
            # done
            
            del doc
            traceln(" done")
            
    
    traceln(" - %d .pxml files skipped" % len(lSkippedFile))


# ---------------------------------------------------------------------------
# Normalizing the box of TextElement, by translating a copy of the Baseline
def normaliseDocElements(doc, xpElement, iNorm):
    for ndPage in doc.getroot().xpath("//pg:Page", namespaces=dNS):
        for ndElt in ndPage.xpath(xpElement, namespaces=dNS):
            try:
                normaliseElement(ndElt, iNorm)
            except NormaliseException as e:
                traceln(str(e))
                traceln("Removing this element")
                ndElt.getparent().remove(ndElt)

    return doc


class NormaliseException(Exception): 
    pass


def normaliseElement(nd, iNorm):
    try:
        ndBaseline = nd.xpath(xpBASELINE, namespaces=dNS)[0]
    except IndexError:
        raise NormaliseException("WARNING: skipped element normalisation: no Baseline: %s" % etree.tostring(nd))
        
    try:
        line = ShapeLoader.node_to_LineString(ndBaseline)
    except ValueError:
        raise NormaliseException("WARNING: skipped element normalisation: invalid Coords: %s" % etree.tostring(nd))
    topline = translate(line, yoff=-iNorm)
    
    # serialise both in circular sequence
    spoints = ' '.join("%s,%s"%(int(x[0]),int(x[1])) for x in line.coords)
    lp=list(topline.coords)
    lp.reverse()
    spoints = spoints+ ' ' +' '.join("%s,%s"%(int(x[0]),int(x[1])) for x in lp)    

    # ad-hoc way of setting the element coodinates
    ndCoords = nd.xpath(".//pg:Coords", namespaces=dNS)[0]
    ndCoords.set("points",spoints)
    
    return
    
# ---------------------------------------------------------------------------
# projection of the GT area onto the doc

class GTProjectionException(Exception): pass

def project_Elt_to_GT(gtdoc, doc
                      , xpElement1, xpElement2
                      , xpArea2
                      , bSep, lsRmId, bEval
                      , fTH=0.5):
    """
    Here we take the element out of the production file to put them in the GT
    doc
    
    WE IGNORE xpArea1 (no need for it)
    
    We return the GT doc
    """
    gtroot = gtdoc.getroot()

    # Evaluation
    # we build a table of list of TextLineId from the GT to check this SW
    # table_id -> row -> col -> list of element id
    dTable = defaultdict(lambda : defaultdict(lambda : defaultdict(list)))
    nOk, nTot = 0, 0
    
    if lsRmId:
        nbEltRemoved = 0
        for sRmId in lsRmId:
            # for _nd in gtroot.xpath('//pg:*[@id="%s"]'%sRmId, namespaces=dNS):
            for _nd in gtroot.xpath('//*[@id="%s"]'%sRmId):
                _nd.getparent().remove(_nd)
                nbEltRemoved += 1
        trace(" (Rm by ID: %d elements removed)" % nbEltRemoved)

    # remove all elements of interest from GT
    # inside TableRegion, we have TextLine, outside we have TextRegion
    if xpElement1 != xpArea2:
        for ndElt in gtroot.xpath(xpElement1, namespaces=dNS):
            if bEval:
                for ndElt2 in ndElt.xpath(xpElement2, namespaces=dNS):
                    dTable[None][None][None].append(ndElt2.get("id"))
            ndElt.getparent().remove(ndElt)
    for ndElt in gtroot.xpath(xpElement2, namespaces=dNS):
        ndCell = ndElt.getparent()
        if bEval: dTable[ndCell.getparent().get("id")][ndCell.get("row")][ndCell.get("col")].append(ndElt.get("id")) 
        ndCell.remove(ndElt)
    if bEval: traceln("\npEvaluation mode")
    
    if bSep:
        nbSepRemoved, nbSepAdded = 0, 0
        for _nd in gtroot.xpath('//pg:SeparatorRegion', namespaces=dNS):
            _nd.getparent().remove(_nd)
            nbSepRemoved += 1
        trace(" (Separators: %d removed" % nbSepRemoved)
                    
    # project the GT areas, page by page
    lNdPage   = doc.getroot().xpath("//pg:Page", namespaces=dNS)
    lNdPageGT =        gtroot.xpath("//pg:Page", namespaces=dNS)
    if len(lNdPage) != len(lNdPageGT):
        raise GTProjectionException("GT and input have different numbers of pages")
    assert len(lNdPage) > 0, "No page??"

    uniqID = 1
    for ndPage, ndPageGT in zip(lNdPage, lNdPageGT):
        print(xpArea2)
        import lxml
        print(lxml.etree.tostring(ndPageGT))
        lNdArea2 = ndPageGT.xpath(xpArea2, namespaces=dNS)
        loArea2 = [ShapeLoader.node_to_Polygon(nd) for nd in lNdArea2]

        for ndElt in ndPage.xpath(xpElement2, namespaces=dNS):
            oElt = ShapeLoader.node_to_Polygon(ndElt)
            
            lOvrl = [oElt.intersection(o).area for o in loArea2]
            iMax = argmax(lOvrl) if lOvrl else None
            vMax = -1 if iMax is None else lOvrl[iMax]
            
            # where to add it?
            if vMax > 0 and vMax / oElt.area > fTH:
                # ok, this is a match
                ndCell = lNdArea2[iMax]
                # add it directly to the area2 (TableCell)
                ndCell.append(deepcopy(ndElt))
                if bEval:
                    if ndElt.get("id") in dTable[ndCell.getparent().get("id")][ndCell.get("row")][ndCell.get("col")]: 
                        nOk += 1
                    else:
                        try: traceln('FAILED:in table: id="%s" "%s"' % (ndElt.get("id"), ndElt.xpath(".//pg:Unicode", namespaces=dNS)[0].text))
                        except IndexError:traceln('FAILED:in table: id="%s" NOTEXT"' % (ndElt.get("id")))
                    
            else:
                # add it outside of any area
                bestNd = ndPageGT
                # add it in its own TextRegion
                ndTR = etree.Element("TextRegion")
                ndTR.set("id", "prjct_region_%d" % uniqID)
                uniqID += 1
                ndTR.set("custom", "")
                ndTR.append(deepcopy(ndElt.xpath("./pg:Coords", namespaces=dNS)[0]))
                ndTR.append(deepcopy(ndElt))
                bestNd.append(ndTR)
                if bEval:
                    if ndElt.get("id") in dTable[None][None][None]: 
                        nOk += 1
                    else:
                        try: traceln('FAILED:in table: id="%s" "%s"' % (ndElt.get("id"), ndElt.xpath(".//pg:Unicode", namespaces=dNS)[0].text))
                        except IndexError:traceln('FAILED:in table: id="%s" NOTEXT"' % (ndElt.get("id")))                        
                        
            nTot += 1
            
        if bSep:
            for _nd in ndPage.xpath('//pg:SeparatorRegion', namespaces=dNS):
                ndPageGT.append(deepcopy(_nd))
                nbSepAdded += 1
    if bSep: trace(", %d added.)  " % nbSepAdded)
        
    if bEval:
        traceln("-"*40)
        trace(" - evaluation: %d ok out of %d = %.2f%%\n" % (nOk, nTot, 100*nOk / (nTot+0.0001)))
        
    return gtdoc


def project_Areas_to_Input(gtdoc, doc, xpElement, xpArea1, xpArea2, bSep, lsRmId, bEval):
    """
    Here we extract teh areas and put them in the input file
    The element must be moved to the right areas
    we return the doc
    """
    raise GTProjectionException("Not implemented")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    
    version = "v.01"
    sUsage="""
Typically for use with ABP tables, to match the GT documents with their HTRed 
    counterpart.
We want to extract the HTRed text and , optionally, the separators from a
    Transkribus processed collection, and inject them in a GT collection, to
    replace the GT text, (and possibly the GT separators).

We have:
- an input  collection obtained by downloading a Transkribus collection using 
    (PyClient) Transkribus_downloader.py
- a GT collection containing the definition of nested areas in each page. 
    (Can be table cells in a table region, or whatever)
    The nesting has 2 levels for now.

In term of nesting, we assume:
    [not CURRENTLY  - xpArea1         are under Page XML element  (xpArea1 is IGNORED and USELESS)
    - xpArea2 (TableCell) are nested under xpArea1 (TableRegion)
    - xpElement1      are under Page XML element
    - xpElement2 (TextLine) are either under xpElement1 (TextRegion) or under xpArea2 (TableCell)
    - SeparatorRegion are under PAGE XML element

We want 
1 - to generate a new document, where the "elements of interest" (e.g TextLine) 
    of the input collection are matched against the GT areas by the location, 
    so that each element is either inserted in an area that matches or left 
    outside any area.
2 - (optionnally) to normalize the bounding area of the "element of interest"
    This is done by making a box of predefined height from the Baseline, which
    becomes the bottom side of the box.
3 - (optionnaly) to discard SeparatorRegion from the GT and get instead those
    from Transkribus.

This is done page by page, for each document.
    
Generate a new collection, with input documents enriched with GT areas.

Any input document without GT counterpart is ignored.

Usage: %s <sInputDir> <sGTDir> <sOutputDir> 
    [--normalize                     (%d          above the Baseline)
    [--normalize_height = <height>   (this height above the Baseline)
    [--normalize-only]
    [--separator]  replace GT SeparatorRegion by those from input.
    [--xpElement1        = <xpath>]   (defaults to "%s")
    [--xpElement2        = <xpath>]   (defaults to "%s")
    [--xparea1           = <xpath>]   (defaults to "%s") (CURRENTLY IGNORED and USELESS)
    [--xparea2           = <xpath>]   (defaults to "%s")
    [--eval]
    
""" % (sys.argv[0], iNORMALIZED_HEIGHT
       , xpELEMENT1, xpELEMENT2
       , xpAREA1, xpAREA1)

    parser = OptionParser(usage=sUsage)
    parser.add_option("--xpElement1", dest='xpElement1',  action="store", type="string"
                      , help="xpath of the elements lvl1"
                      , default=xpELEMENT1)   
    parser.add_option("--xpElement2", dest='xpElement2',  action="store", type="string"
                      , help="xpath of the elements lvl2 to project"
                      , default=xpELEMENT2)   
    parser.add_option("--xpArea1", dest='xpArea1',  action="store", type="string"
                      , help="xpath of the areas level 1 in GT"
                      , default=xpAREA1)   
    parser.add_option("--xpArea2", dest='xpArea2',  action="store", type="string"
                      , help="xpath of the areas level 2 (nested) in GT"
                      , default=xpAREA2)   
    parser.add_option("--normalize", dest='bNorm',  action="store_true"
                      , help="normalise the box of elements of interest")   
    parser.add_option("--separator", dest='bSep',  action="store_true"
                      , help="replace any separator by those from the Transkribus collection")   
    parser.add_option("--normalize_height", dest='iNormHeight',  action="store", type="int"
                      , help="normalise the box of elements of interest")   
    parser.add_option("--normalize-only", dest='bNormOnly',  action="store_true"
                      , help="only normalize, does not project GT")   
    parser.add_option("--rm_by_id", dest='lsRmId',  action="append"
                      , help="Remove those elements from the output XML")   
    parser.add_option("--eval", dest='bEval',  action="store_true"
                      , help="evaluation mode, pass GT as input!!")   
    parser.add_option("--warm", dest='bWarm',  action="store_true"
                      , help="Warm mode: skipped input files with a fresh output already there")   
    parser.add_option("--pxml", dest='bPXml',  action="store_true"
                      , help="Look directly for .pxml files")
    parser.add_option("-v", "--verbose", dest='bVerbose',  action="store_true"
                      , help="Verbose mode")
    (options, args) = parser.parse_args()
    
    try:
        sInputDir, sGTDir, sOutputDir = args
    except ValueError:
        sys.stderr.write(sUsage)
        sys.exit(1)
    
    # ... normalization
    bNorm = bool(options.bNorm) or bool(options.iNormHeight) or bool(options.bNormOnly)
    iNorm = options.iNormHeight if bool(options.iNormHeight) else iNORMALIZED_HEIGHT

    # ... checking folders
    #if not os.path.normpath(sInputDir).endswith("col")  : sInputDir = os.path.join(sInputDir, "col")
    #if not os.path.normpath(sGTDir).endswith("col")     : sGTDir = os.path.join(sGTDir, "col")
    if os.path.isdir(sInputDir) and os.path.isdir(sGTDir):
        # create the output fodlers if required
        if os.path.normpath(sOutputDir).endswith("col") :
            pass  # we expect the user knows what s/he does
        else:
            # try to create them
            try: os.mkdir(sOutputDir);
            except: pass
            sOutputDir = os.path.join(sOutputDir, "col")
            try: os.mkdir(sOutputDir);
            except: pass
    # all must be ok by now
    lsDir = [sInputDir, sGTDir, sOutputDir]
    if not all(os.path.isdir(s) for s in lsDir):
        for s in lsDir:
            if not os.path.isdir(s): sys.stderr.write("Not a directory: %s\n"%s)
        sys.exit(2)

    sExt = ".pxml" if options.bPXml else ".mpxml"

    # ok, go!
    traceln("Input is : ", os.path.abspath(sInputDir))
    traceln("GT is in : ", os.path.abspath(sGTDir))
    traceln("Ouput in : ", os.path.abspath(sOutputDir))
    traceln("Elements lvl 1: ", repr(options.xpElement1))
    traceln("Elements lvl 2: ", repr(options.xpElement2))
    traceln("GT areas lvl 1          : " , repr(options.xpArea1))
    traceln("GT areas lvl 2 (nested) : " , repr(options.xpArea2))
    traceln("Normalise elements  : ", bNorm)
    traceln("Normalise to height : ", iNorm)
    traceln("Get separators : ", options.bSep)
    traceln("Remove elements with @id: ", options.lsRmId)
    traceln("File extension: ", sExt)

    if os.listdir(sOutputDir): traceln("WARNING: *** output folder NOT EMPTY ***")

    main(sInputDir, sGTDir, sOutputDir
         , options.xpElement1, options.xpElement2
         , options.xpArea1, options.xpArea2
         , bNorm, iNorm, options.bNormOnly
         , options.bSep
         , options.lsRmId
         , options.bEval
         , options.bWarm
         , sExt=sExt
         , bVerbose=options.bVerbose)
    
    traceln("Done.")