# -*- coding: utf-8 -*-
"""


    Build a table grid from cells

     H. Déjean
    

    copyright Naver 2019
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
"""

from lxml import etree
import os,sys
# from sklearn.pipeline import Pipeline, FeatureUnion
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.base import BaseEstimator, TransformerMixin
import sys, os.path
sys.path.append (os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))) + os.sep+'TranskribusDU')

# from contentProcessing.taggerChrono import DeepTagger
# class Transformer(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         BaseEstimator.__init__(self)
#         TransformerMixin.__init__(self)
#         
#     def fit(self, l, y=None):
#         return self
# 
#     def transform(self, l):
#         assert False, "Specialize this method!"
#         
# class SparseToDense(Transformer):
#     def __init__(self):
#         Transformer.__init__(self)
#     def transform(self, o):
#         return o.toarray()    
#     
# class NodeTransformerTextEnclosed(Transformer):
#     """
#     we will get a list of block and need to send back what a textual feature extractor (TfidfVectorizer) needs.
#     So we return a list of strings  
#     """
#     def transform(self, lw):
#         return map(lambda x: x, lw) 

def getSequenceOfNConsecutive(n,lListe):
    """
        filter elements which have less than N consecutive elements
        
        pi-1: x or x-1
        pi : X  : max of possible years
        pj : X or x+1
        
        
        start with max of a page; and delete other in neighbourhood
    """
def processDS(infile):
    """
        Extract year candidates per page
        
        Tag dates as well: all content
            add 6 Uhr Morgens   ....
            
        enrich at page level
    """
    lDates= [str(x) for x in range(1845,1880)]
#     print('file... %s'%(infile))
    et = etree.parse(infile)
    xpath  = "//%s" % ("PAGE")
    ltr= et.xpath(xpath)
    lAllPages=[]
    for it,page in enumerate(ltr):
        lDocDates= []
#         lDocDates= []
        xpath  = ".//%s" % ("TEXT")
        llines= page.xpath(xpath)
        for i ,elem in enumerate(llines):
            for date in lDates:
                if elem.text and elem.text.find(date)>=0:
                    lDocDates.append(int(date))
        #filter and sort
#         page.attrib['years']= "-".join(list(lDocDates))
        lAllPages.append(lDocDates[:])
#         print("%s\t%d\t%s"%(infile,it,lDocDates))
    if lAllPages ==[]: print('NONE!!',infile,len(ltr))
    pok=0
    curyear=0
    for i in range(1,len(lAllPages)-1):
#         print (i,lAllPages[i] ,lAllPages[i-1],  lAllPages[i+1])
        # check with prev
        lDInPrev = []
        lDCur=[]
        lDInNext = []
        for d in  lAllPages[i]:
            if d in lAllPages[i-1]:
                lDInPrev.append(d)
                lDCur.append(d)
            if int(d)-1 in lDInPrev:
                lDCur.append(d)
            if d in lAllPages[i+1]:
                lDInNext.append(d)
            if int(d)+1 in lDInNext:
                lDCur.append(d)
                lDInNext.append(int(d)+1)                
#         print (i, lDCur,lDInPrev,lDInNext)
        if lDCur != []:
            year = max(set(lDCur), key = lDCur.count)
            if  int(year) >=curyear:
                print (i,year)
                ltr[i].set('computedyear',str(year))
                curyear=year
            
                
#     print (infile,len(lAllPages),pok)


    et.write(infile)
        
def getPageBreak(infile):
    """
        DB format?? 
        
        collect month name and appy taggerchrono -> normalisation  yes
        add date?
        if break add info  -> finf first record of next year
    """
    if True or False:
        from contentProcessing.taggerChrono import DeepTagger
        tagger=DeepTagger()
        tagger.sModelName="chrono_lstmdense_64_128_32_3_512"
        tagger.dirName="."
        #load
        tagger.loadModels()
        
    print('file... %s'%(infile))
    et = etree.parse(infile)
    xpath  = "//%s" % ("PAGE")
    ltr= et.xpath(xpath)
    print(len(ltr))
    ## concat all
    highest=0
    curYear=None
    lBreak=[]
    lYears=[]
    for it,page in enumerate(ltr):
        lBreak[it]=False
        lYears[it] = page.get('computedyear')
        lmonth=[]
        year=page.get('years')
        xpath  = "./%s" % ("RECORD")
        llines= page.xpath(xpath)
        for line in llines:
            try:
                # monthdaydategenerator as well 
                line.attrib['deathDate']
                lmonth.append(line.attrib['deathDate'])
            except KeyError:pass
        print (lmonth)
        # process lmonth with taggerchrono
        if tagger and lmonth != []: 
            lres = tagger.predict_multiptype(lmonth)
            for res  in lres: 
                monthNum = int(res[0][0][2][0])
                monthNumProba = res[0][0][2][1]
                print (monthNum,monthNumProba)
                if monthNum < highest and monthNumProba > 0.1:
                    bPageBreak=True
                    print ('BREAK', highest, monthNum,monthNumProba)
                    lBreak[it]=True
                elif monthNum >highest:highest=monthNum
            # compare with highest month: if lower: breakpage 
        # replace deathDate by deathMonthNumber
    
    lYearWOPrev=[]
    lastyear=""
    for i,y in lYears[1:]:
        if y != "":
            lastyear= i
            if lYears[i-1] == "":lYearWOPrev.append(i)
    
    #last forward:
    if lYears[-1] == "":
        pass
        #forward from lastyear
    
    ## fora a page wo year: take the nearest page with year and the number of break in between
    ## take pages with year (and previous wo year) : go back and update each page/record    
    
def processDocument(infile,outfile):
    """
        associate a date to a page
        detect year increment first?
    """
    

    print('file... %s'%(infile))
    et = etree.parse(infile)
    xpath  = "//%s" % ("PAGE")
    ltr= et.xpath(xpath)
    print(len(ltr))
    for it,page in enumerate(ltr):
        sYear=page.get('years')
        lyear=sYear.split('-')
        myyear=lyear[0]
        parish=page.get('pagenum')
        if parish:
            location=parish.split('_')[0]
        xpath  = "./%s" % ("RECORD")
        llines= page.xpath(xpath)
        for i ,elem in enumerate(llines):
            elem.attrib['year'] =myyear
            elem.attrib['parish']= location 
    et.write(outfile)


if __name__ == "__main__":
    processDS(sys.argv[1])
#     getPageBreak(sys.argv[1])
#     processDocument(sys.argv[1], sys.argv[2])
    
    
    