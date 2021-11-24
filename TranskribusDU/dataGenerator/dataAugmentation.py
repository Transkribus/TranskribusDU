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
import glob 
import numpy as np
from lxml import etree
from random import gauss, random, shuffle
from shapely.geometry import LineString
from shapely.geometry import Polygon
from shapely.affinity import translate

from copy import deepcopy

from util.Shape import ShapeLoader

from xml_formats.PageXml import PageXml
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))
#sys.path.append(os.path.dirname(os.path.abspath(sys.argv[0])))


NS_PAGE_XML         = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"    


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


def translateElt(lCoordDom,offset):
    return [(x+offset,y) for (x,y) in lCoordDom]

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
        
        
        
def translatePage(xmlfile,param):    
    
    xpath  = "//a:TextLine/a:Coords"    
    pageNode=xml.getroot()
    lNodes = xmlfile.xpath(xpath,namespaces ={'a':NS_PAGE_XML})
    lNewNodes=[]
    for node in lNodes:
        cpyNode=node #deepcopy(node)
        # add cpyNonde to the page
        # Coords
        spoints=cpyNode.attrib['points']  
        lList= [(x) for xy  in spoints.split(' ') for x in xy.split(',')]
        lp =[ (float(x),float(y)) for x,y  in zip(lList[0::2],lList[1::2]) ]
        try: line=LineString(lp)
        except ValueError: return  # LineStrings must have at least 2 coordinate tuples
        trans = translate(line,param)
        cpyNode.attrib['points']=" ".join("%s,%s"%(x,y) for x,y in trans.coords)
        #baseline        
        baselinecoord= cpyNode.getnext()
        try:
            spoints=baselinecoord.attrib['points']  
            lList= [(x) for xy  in spoints.split(' ') for x in xy.split(',')]
            lp =[ (float(x),float(y)) for x,y  in zip(lList[0::2],lList[1::2]) ]
            try: line=LineString(lp)
            except ValueError: return  # LineStrings must have at least 2 coordinate tuples            
            trans = translate(line,param)
            baselinecoord.attrib['points']=" ".join("%s,%s"%(x,y) for x,y in trans.coords)
        except: # no baseline???? 
            pass        
        lNewNodes.append(cpyNode)
        pageNode.append(cpyNode)

    return lNewNodes
        
def deleteElement(xmlfile,param):
    """
        randomly delete TextLine
        @param: param ([0,1]) : percentage of  deleted elements
    """
    xpath  = "//a:TextLine"    
    lNodes = xmlfile.xpath(xpath,namespaces ={'a':NS_PAGE_XML})
    for node in lNodes:
        if random()< param: node.getparent().remove(node)
        

def splitTextLine(imgW,tl):
    """
        cut a tl into two tls (randomly)
        assume horizontal line!
        
        create the 'gap' and make the difference with tl
        
        need to update baseline, text 
    """
    poly= ShapeLoader.node_to_Polygon(tl, bValid=True)
    #get gap
    x1,y1,x2,y2 = poly.bounds
    length= abs(x2-x1)
    
    r= random()
    gaplength= length * r
    if gaplength < imgW*.1: return #too small
    
    gap=Polygon([(x1+gaplength,y1),(x1+gaplength+10,y1),(x1+gaplength+10,y2),(x1+gaplength,y2)])
    l = list(poly.difference(gap))
    if  len(l)!=2: return
    
    #baseline # todo!!
#     xpath  = ".//a:Baseline"    
#     bl = tl.xpath(xpath,namespaces ={'a':NS_PAGE_XML})[0]        
    
    # srto by x1:
    l.sort(key=lambda x:x.bounds[0])
    tl2=deepcopy(tl)
    scoords = ShapeLoader.getCoordsString(l[0])
    xpath  = ".//a:Coords"    
    coords = tl.xpath(xpath,namespaces ={'a':NS_PAGE_XML})[0]    
    coords.set('points',scoords)
    xpath  = ".//a:Unicode"
    uni = tl.xpath(xpath,namespaces ={'a':NS_PAGE_XML})[0]
    text = uni.text[:] 
    textl= int(round(len(text)*r))
    uni.text = text[:textl]
    
    xpath  = ".//a:Coords"    
    coords = tl2.xpath(xpath,namespaces ={'a':NS_PAGE_XML})[0]  
    coords.set('points',ShapeLoader.getCoordsString(l[1]))
    
    xpath  = ".//a:Unicode"    
    uni = tl2.xpath(xpath,namespaces ={'a':NS_PAGE_XML})[0] 
    uni.text = text[textl:]
    
    tl2.set('id',tl.get('id')+'_dup')
    tl.getparent().append(tl2)
            
def splitTextLines(xmlfile,proba):
    """
        split aTextline (p>param), and add a small gap  
    """
    page = xmlfile.findall(f"//pc:Page", {"pc":PageXml.NS_PAGE_XML})
    imgW = int(page[0].get('imageWidth'))
    
    xpath  = "//a:TextLine"
    lNodes= xmlfile.xpath(xpath,namespaces ={'a':NS_PAGE_XML})
    for node in lNodes:      
        if  random() > proba:
            splitTextLine(imgW,node)    
        
        
        
def textlineNormalisation(xmlfile,param):    
    xpath  = "//a:TextLine/a:Baseline"
    lNodes= xmlfile.xpath(xpath,namespaces ={'a':NS_PAGE_XML})
    for node in lNodes:        
        linePolygon(node,param)

def replaceCharacter(s,th):
    """
         add noise  (replace char) to pureGen
    """
        
    ns=""
    for i in range(len(s)):
        generateProb = random.uniform(0,100)
        if generateProb < th:
            ns+=chr(int(random.uniform(65,240)))
        else: ns+=s[i]
    return ns
       

def duplicatePage(xml):
    """
        create a double page with the initial page
    """
    page = xml.findall(f"//pc:Page", {"pc":PageXml.NS_PAGE_XML})
    imgW = page[0].get('imageWidth')
#     imgH = page.get('imageHeigth')
    # update imgW!
    page[0].set('imageWidth',str(int(imgW)*2))
    
    xpath  = "//a:Page/*"    
    lNodes = xml.xpath(xpath,namespaces ={'a':NS_PAGE_XML})
    lcpy=[]
    [lcpy.append(etree.fromstring(etree.tostring(n))) for n in lNodes]
    processPageXml(xml,translateElt,float(imgW))
    # update ids
    xpath  = "//*[@id]"    
    lnodes = xml.xpath(xpath,namespaces ={'a':NS_PAGE_XML})
    for n in lnodes:
        n.set('id','dup_%s'%n.get('id'))
    
    #update menu-item number!!!
    xpath  = "//*[@menu_item]"    
    lnodes = xml.xpath(xpath,namespaces ={'a':NS_PAGE_XML})
    for n in lnodes:
        n.set('menu_item','dup_%s'%n.get('menu_item'))
    xpath  = "//*[@section]"    
    lnodes = xml.xpath(xpath,namespaces ={'a':NS_PAGE_XML})
    for n in lnodes:
        n.set('section','dup_%s'%n.get('section'))           
    
    xpath  = "//a:Page"    
    lPages = xml.xpath(xpath,namespaces ={'a':NS_PAGE_XML})    
    [lPages[0].append(n) for n in lcpy]
    
    return xml

def permuteText(xml,tag='Word'):
    """
        permute words from the same category
    """
    lTokens = xml.findall(f"//pc:{tag}", {"pc":PageXml.NS_PAGE_XML})
    dWordLabel=dict()
    for token in lTokens:
        try:dWordLabel[token.get('type')].append(token)
        except: dWordLabel[token.get('type')]=[token]
    [shuffle(l) for l in dWordLabel.values()]
    print(dWordLabel.keys())
    for token in lTokens:
        unicode=token.findall(f".//pc:Unicode", {"pc":PageXml.NS_PAGE_XML})[0]
#         oldtext = unicode.text
#         print (unicode.text)
        new= dWordLabel[token.get('type')].pop()
        newtext=new.findall(f".//pc:Unicode", {"pc":PageXml.NS_PAGE_XML})[0]
        unicode.text= newtext.text
#         print (f'{oldtext} --> {unicode.text}')
        
if __name__ == "__main__":

    parser = optparse.OptionParser(usage="<INPUT FILENAME> <OUTPUT FOLDER>  <OUTPUT FILENAME>" )
    
#     parser.add_option("--informat", dest='inputFormat',  action="store", type='string'
#                       , help="input format: PageXml (default) or DSXML")    
#     parser.add_option("--skew", dest='bSkew',  action="store"
#                       , help="perform skewing deformation")    
    parser.add_option("--attribute", dest='attribute',  action="store", default=None
                      , help="for building label dictionary (Word level only?)")  
    (options, args) = parser.parse_args()



    try:
        lsDir = args
        lsDir[0]
    except:
        parser.print_help()
        parser.exit(1, "")
        
    try: 
        inputFile = sys.argv[1]
        outFolder = sys.argv[2]
        outFile = sys.argv[3]
    except IndexError as e:
        print("usage: dataAugmentation.py INPUTFILE OUTFILE")
        sys.exit() 
    
    if os.path.isdir(inputFile):
        lsFile = sorted([s for s in glob(os.path.join(inputFile, '*'))]) #if s.endswith(options.extension)])
    else:
        lxml = [inputFile]
        
#     # get textual dict
#     if options.attribute is not None:
#         buildDictionnary(lsFile,options.attribute)
    
    for xmlfile in lxml:
        xml = loadFile(xmlfile)
#         permuteText(xml)
#         xml.write(os.path.join(outFolder,"dup_%s"%(outFile)))

        duplicatePage(xml) 
#         splitTextLines(xml,0.66)
        xml.write(os.path.join(outFolder,"dup_%s"%(outFile)))
#         for a in [-0.5,0.5]:
#             xml = loadFile(inputFile)
#             processPageXml(xml, skewing,a)
#             xml.write(os.path.join(outFolder,"%s_%s_%s"%(None,a,outFile)))
    #         for h in [30,50]:
    #             textlineNormalisation(xml, h)
    #             xml.write("%s_%s_%s"%(h,a,outFile))
    #     for h in [30,50]:
    #         xml = loadFile(inputFile)
#         xml = loadFile(xmlfile)
#         processPageXml(xml,rotateV180)
    #         textlineNormalisation(xml, h)
    #         textlineLocalSkewing(xml,0.5)
#         xml.write("%s_%s"%(180,outFile))

    ### ADD TEXT NOISE ???
#     var = 0.5
#     xml = loadFile(inputFile)
#     textlineLocalSkewing(xml, var)
#     xml.write(os.path.join(outFolder,"%s_%s_%s"%(var,'locals',outFile)))





#     var = 0.5
#     deleteElement(xml,var)
#     
#     xml.write("%s_%s_%s"%(var,'del',outFile))
        