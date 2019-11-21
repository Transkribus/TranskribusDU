# -*- coding: utf-8 -*-
"""
    Build a table grid from cells

     H. Déjean

    copyright Naver 2019
    READ project 

    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
"""

from lxml import etree
import os,sys
import re
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import sys, os.path
sys.path.append (os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))) + os.sep+'TranskribusDU')
 
from contentProcessing.taggerChrono import DeepTagger
class Transformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)
         
    def fit(self, l, y=None):
        return self
 
    def transform(self, l):
        assert False, "Specialize this method!"
         
class SparseToDense(Transformer):
    def __init__(self):
        Transformer.__init__(self)
    def transform(self, o):
        return o.toarray()    
     
class NodeTransformerTextEnclosed(Transformer):
    """
    we will get a list of block and need to send back what a textual feature extractor (TfidfVectorizer) needs.
    So we return a list of strings  
    """
    def transform(self, lw):
        return map(lambda x: x, lw) 

def getSequenceOfNConsecutive(n,lListe):
    """
        filter elements which have less than N consecutive elements
        
        pi-1: x or x-1
        pi : X  : max of possible years
        pj : X or x+1
        
        
        start with max of a page; and delete other in neighbourhood
    """

lMappingFirstYear={
    20005:1862,
    
    }
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
        ltr[i].set('computedyear','')
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

def addAllFields(record):
        """
        <RECORD lastname="Krern" firstname="Anna Maria" religion="" occupation="H&#228;uslerin" location="Baugt&#246;tting" situation="Wittwer." deathreason="Marass. senil." doktor="Loter" monthdaydategenerator="29" monthdaydategenerator="Oktbr." monthdaydategenerator="" burialdate="Juni" buriallocation="geb." age="" ageunit=""/>
        """
        lFields =[
            'lastname','firstname','religion','occupation','location','situation','deathreason','doktor','monthdaydategenerator','deathdate','deathyear','burialdate','buriallocation', 'age','ageunit'
            ]
        
        for f in lFields:
            try: record.attrib[f]
            except KeyError:record.attrib[f]=''

def normalizeAgeUnit(unit):
    unit=unit.lower()
    if unit =="":
        return 0
    elif unit[0] == 'j' : return 1    
    elif unit[0] == 'm' : return 2 
    elif unit[0] == 'w' : return 3
    elif unit[0] == 't' : return 5
    elif unit[0] == 's' : return 6
    else: return 0  
    
def normalizeAgeValue(age):
    if age == "":
        return 0
    age = re.sub(r'\D+', '½',age)
    age = re.sub(r'\D+', '¼',age)    
    age = re.sub(r'\D+', '⅓',age)
    age = re.sub(r'\D+', '¾',age)      
    
    try:return int(age)
    except: return 0


    
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
#     print(len(ltr))
    ## concat all
    highest=0
    curYear=''
    ## forward then backward!
    backward = False
    for it,page in enumerate(ltr):
        parish=page.get('pagenum')
        # why do we have \col\None\S_Indersbach_003_0001
        #  col  None S_Indersbach_003_0001
        #            S Indersbach   003    0001
        parish = re.sub(r'_\d+', '',parish).replace('\\col\\None\\S_','')
        if page.get('years') !='' and it+1 <len(ltr) and ltr[it+1].get('years') !="":
            curYear =int(page.get('years'))
        lmonth=[]
        xpath  = "./%s" % ("RECORD")
        llines= page.xpath(xpath)
        for il,line in enumerate(llines):
            addAllFields(line)
            line.attrib['colid']  = "6285"
            line.attrib['docid']  = re.search('\d{5}',infile).group(0)
            line.attrib['pageid'] = page.attrib['number']
            line.attrib['parish']=parish
            line.attrib['int_ageunit'] = str(normalizeAgeUnit(line.attrib['ageunit']))
            line.attrib['int_age'] = str(normalizeAgeValue(line.attrib['age']))
            
            day = re.sub(r'\D+', '',line.attrib['monthdaydategenerator'])
            line.attrib['monthdaydategenerator'] =day
            line.attrib['int_monthdaydate'] = day
            # burial date is more chronological
            month= line.attrib['burialdate']
            if month =="": month= line.attrib['deathdate']
            month = month.replace("Jäner",'Januar')
            month = month.replace("än",'an')
            if day !="":xx ="%s %s "%(day ,month)
            else:xx ="10 %s "%(month) 
            xx =xx.strip()
            lmonth.append(xx) #,line.attrib['deathyear']))
#         print (line.attrib['pageid'],curYear,lmonth)
        # process lmonth with taggerchrono
        if tagger and lmonth != []: 
            lres = tagger.predict_multiptype(lmonth)
            prevmonthNumProba=0
            for res  in lres: 
                for irec,record in enumerate(res):
#                     print (record) 
                    monthNum = int(record[0][2][0])
                    monthNumProba = record[0][2][1]
                    dayNum = int(record[0][3][0])
                    dayProba = record[0][3][1]      
                    
                    try:nextProba= res[irec+1][0][2][1]
                    except IndexError:nextProba=0
#                     print (dayNum,dayProba,monthNum,monthNumProba)
#                     if llines[irec].attrib['int_monthdaydate'] != ''  and dayNum != int(llines[irec].attrib['int_monthdaydate']): print ("%s != %s " % (dayNum,llines[irec].attrib['int_monthdaydate']))
#                     print (monthNum,monthNumProba)
                    llines[irec].attrib['int_deathmonth']=str(monthNum)
#                     llines[irec].attrib['int_deathday']=monthNum
                    if monthNum < highest and monthNumProba > 0.5 and (prevmonthNumProba > 0.5 or nextProba >0.5):
                        bPageBreak=True
#                         print ('BREAK', curYear,highest, monthNum,monthNumProba)
                        if curYear !='':
                            if backward:curYear =- 1
                            else: curYear += 1
                        highest = monthNum
                    elif monthNum >highest and monthNumProba > 0.70  and prevmonthNumProba > 0.70:highest=monthNum
                    elif monthNumProba > 0.5  and (prevmonthNumProba > 0.5 or nextProba >0.5) :highest=monthNum
                    
                    if curYear !='':
                        llines[irec].attrib['year']=str(curYear)
#                         print (curYear)
                    prevmonthNumProba = monthNumProba
                    #print (etree.tostring(llines[irec]))    
            # compare with highest month: if lower: breakpage 
        # replace deathDate by deathMonthNumber
    curYear=''
    highest = 0
    backward = True
    ltr.reverse()
    for it,page in enumerate(ltr):
        if page.get('years') !='' and it+1 <len(ltr) and ltr[it+1].get('years') !="":
            curYear =int(page.get('years'))
        else:
            lmonth=[]
            xpath  = "./%s" % ("RECORD")
            llines= page.xpath(xpath)
            llines.reverse()
            for il,line in enumerate(llines):
                 
                month= line.attrib['deathdate']
                month = month.replace("Jäner",'Januar')
                month = month.replace("än",'an')
                xx ="%s %s "%(line.attrib['int_monthdaydate'] ,month)
                xx =xx.strip()
                lmonth.append(xx)
#             print (lmonth)
            # process lmonth with taggerchrono
            if tagger and lmonth != []: 
                lres = tagger.predict_multiptype(lmonth)
                prevmonthNumProba=0
                for res  in lres: 
                    for irec,record in enumerate(res):
    #                     print (record) 
                        monthNum = int(record[0][2][0])
                        monthNumProba = record[0][2][1]
#                         print (monthNum,monthNumProba)
                        if monthNum > highest and monthNumProba > 0.5 and prevmonthNumProba > 0.75:
                            bPageBreak=True
#                             print ('BREAK', curYear,highest, monthNum,monthNumProba)
                            if curYear !='':
                                if backward:curYear -= 1
                                else: curYear += 1
                            highest = monthNum
#                         elif monthNum >highest and monthNumProba > 0.75  and prevmonthNumProba > 0.5:highest=monthNum
                        elif  monthNumProba > 0.75  and prevmonthNumProba > 0.5:highest=monthNum
                           
                        if curYear !='':
                            try:llines[irec].attrib['year']
                            except KeyError:
                                llines[irec].attrib['YEAR']=str(curYear)
#                                 print (curYear)
                        prevmonthNumProba = monthNumProba  


    
    et.write("%s.out" %(infile),xml_declaration=True, encoding='UTF-8')
    
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
    if sys.argv[1] == 'year':
        processDS(sys.argv[2])
    elif sys.argv[1] == 'norm':
        getPageBreak(sys.argv[2])
    else:
        print("uage: %s [year|norm] INFILE"%sys.argv[0])
        
#     processDocument(sys.argv[1], sys.argv[2])
    
    
    