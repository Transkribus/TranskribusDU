# -*- coding: utf-8 -*-
"""
    Data Augmentation: 
    
    generate Layout annotated data 
    
    copyright Naver Labs 2019
    READ project 

    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
    @author: H. Déjean
"""

import os,sys, optparse
import numpy as np
from lxml import etree
from random import gauss, random
from shapely.geometry import LineString
from shapely.affinity import translate
 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))
sys.path.append(os.path.dirname(os.path.abspath(sys.argv[0])))

def skewing(lCoordDom,angle):
    """
        skewing  (0,0) top left corner
        
        need a skewing for double page, skewing from the binding??
        
        need also to adapt the page size!
    """        

    theta= np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c,-s), (s, c)))  
    
    lSkewed= []
    for coords in lCoordDom:
        xx,yy = np.dot(R,[coords[0],coords[1]])
        lSkewed.append((int(round(xx)),int(round(yy))))
    return lSkewed
    
    
def rotateV180(lCoordDom,pageWidth):
    """
        rotate the page 180 vertically
    """ 
    vaxis =  pageWidth * 0.5
    lRot = []
    for x,y in lCoordDom:
        d = abs(vaxis - x )
        if x > vaxis:
            x = vaxis - d 
        else: x = vaxis + d
        lRot.append((int(round(x)),int(round(y))))
    return lRot


def linePolygon(baseline,param):
    lList= [(x) for xy  in baseline.attrib['points'].split(' ') for x in xy.split(',')]
    lcoord =[ (float(x),float(y)) for x,y  in zip(lList[0::2],lList[1::2]) ]
    try: line=LineString(lcoord)
    except ValueError: return  # LineStrings must have at least 2 coordinate tuples
    topline=translate(line,yoff=-param)
    spoints = ' '.join("%s,%s"%(int(x[0]),int(x[1])) for x in line.coords)
    lp=list(topline.coords)
    lp.reverse()
    spoints =spoints+ ' ' +' '.join("%s,%s"%(int(x[0]),int(x[1])) for x in lp) 
    baseline.getparent()[0].set('points',spoints)

def loadFile(file):
    return etree.parse(file)
     
def processPageXml(xmlfile,foo,param=None):
    """
    """
    if param ==None:
        NS_PAGE_XML         = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"    
        xpath  = "//a:%s" % ("Page")
        lP= xmlfile.xpath(xpath,namespaces ={'a':NS_PAGE_XML})
        assert len(lP) == 1
        param= int(lP[0].attrib['imageWidth'])
    
    xpath  = "//*[@%s]" % ("points")
    lNodes= xmlfile.xpath(xpath)
    for node in lNodes:
        spoints=node.attrib['points']
        lList= [(x) for xy  in spoints.split(' ') for x in xy.split(',')]
        lp =[ (float(x),float(y)) for x,y  in zip(lList[0::2],lList[1::2]) ]
        lsk= foo(lp,param)
        node.attrib['points']=" ".join("%s,%s"%(int(round(x)),int(round(y))) for x,y in lsk)
    

def textlineLocalSkewing(xmlfile,param):    
    
    NS_PAGE_XML         = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"    
    xpath  = "//a:TextLine/a:Coords"    
    lNodes = xmlfile.xpath(xpath,namespaces ={'a':NS_PAGE_XML})
    for node in lNodes:
        angle=gauss(0,param)
        # Coords
        spoints=node.attrib['points']  
        lList= [(x) for xy  in spoints.split(' ') for x in xy.split(',')]
        lp =[ (float(x),float(y)) for x,y  in zip(lList[0::2],lList[1::2]) ]
        lsk = skewing(lp,angle)
        node.attrib['points']=" ".join("%s,%s"%(x,y) for x,y in lsk)
        #baseline
        baselinecoord= node.getnext()
        try:
            spoints=baselinecoord.attrib['points']  
            lList= [(x) for xy  in spoints.split(' ') for x in xy.split(',')]
            lp =[ (float(x),float(y)) for x,y  in zip(lList[0::2],lList[1::2]) ]
            lsk = skewing(lp,angle)
            baselinecoord.attrib['points']=" ".join("%s,%s"%(x,y) for x,y in lsk)
        except: # no baseline???? 
            pass
        
        
        
        
def deleteElement(xmlfile,param):
    """
        randomly delete TextLine
        @param: param ([0,1]) : percentage of  deleted elements
    """
    NS_PAGE_XML         = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"    
    xpath  = "//a:TextLine"    
    lNodes = xmlfile.xpath(xpath,namespaces ={'a':NS_PAGE_XML})
    for node in lNodes:
        if random()< param: node.getparent().remove(node)
        
def textlineNormalisation(xmlfile,param):    
    NS_PAGE_XML         = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"    
    xpath  = "//a:TextLine/a:Baseline"
    lNodes= xmlfile.xpath(xpath,namespaces ={'a':NS_PAGE_XML})
    for node in lNodes:        
        linePolygon(node,param)

if __name__ == "__main__":

    parser = optparse.OptionParser(usage="")
    
    parser.add_option("--informat", dest='inputFormat',  action="store", type='string'
                      , help="input format: PageXml (default) or DSXML")    
    parser.add_option("--skew", dest='bSkew',  action="store"
                      , help="perform skewing deformation")    
    parser.add_option("--perpective", dest='extension',  action="store", default=''
                      , help="perform perspective deformation")  
    (options, args) = parser.parse_args()

    try:
        lsDir = args
        lsDir[0]
    except:
        parser.print_help()
        parser.exit(1, "")
        
    try: 
        inputFile = sys.argv[1]
        outFile = sys.argv[2]
    except IndexError as e:
        print("usage: dataAugmentation.py INPUTFILE OUTFILE")
        sys.exit() 
        
    xml = loadFile(inputFile)
    
    for a in [-1,-0.5,0.5,1]:
        xml = loadFile(inputFile)
        processPageXml(xml, skewing,a)
        xml.write("%s_%s_%s"%(None,a,outFile))
#         for h in [30,50]:
#             textlineNormalisation(xml, h)
#             xml.write("%s_%s_%s"%(h,a,outFile))
#     for h in [30,50]:
#         xml = loadFile(inputFile)
#         processPageXml(xml,rotateV180)
#         textlineNormalisation(xml, h)
#         textlineLocalSkewing(xml,0.5)
#         xml.write("%s_%s_%s"%(h,180,outFile))
    
    var = 0.5
    xml = loadFile(inputFile)
    textlineLocalSkewing(xml, var)
    xml.write("%s_%s_%s"%(var,'locals',outFile))

#     var = 0.5
#     deleteElement(xml,var)
#     
#     xml.write("%s_%s_%s"%(var,'del',outFile))
        