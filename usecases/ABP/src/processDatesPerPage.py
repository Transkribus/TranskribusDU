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
            # same as previous and same as next
            # same as previous
            #[1847, 1877] [1847, 1877] [1847, 1877] [1847]
            # -> add more weight 
            if d in lAllPages[i-1] and i < len(lAllPages)-1 and d in lAllPages[i+1]:
                lDCur.append(d)
            if d in lAllPages[i-1]:
                lDInPrev.append(d)
                lDCur.append(d)
            # next year
            if int(d)-1 in lDInPrev:
                lDCur.append(d)
            # same as next
            if int(d)-1 in lDInPrev:
                lDInNext.append(d)
            # next year in next page
            if int(d)+1 in lDInNext:
                lDCur.append(d)
                lDInNext.append(int(d)+1)                
#         print (i,lAllPages[i], lDCur,lDInPrev,lDInNext)
        if lDCur != []:
            year = max(set(lDCur), key = lDCur.count)
            print (i,year,curyear)
            if  int(year) >=curyear:
                print (i,year)
                ltr[i].set('computedyear',str(year))
                curyear=year
            
                
#     print (infile,len(lAllPages),pok)


#     et.write(infile+".year")
    et.write(infile)

def addAllFields(record):
        """
        <RECORD lastname="Krern" firstname="Anna Maria" religion="" occupation="H&#228;uslerin" location="Baugt&#246;tting" situation="Wittwer." deathreason="Marass. senil." doktor="Loter" monthdaydategenerator="29" monthdaydategenerator="Oktbr." monthdaydategenerator="" burialdate="Juni" buriallocation="geb." age="" ageunit=""/>
        """
        lFields =[
            'lastname','firstname','religion','occupation',
            'location','situation','deathreason','doktor','monthdaydategenerator',
            'deathdate','deathyear','burialdate','buriallocation', 'age','ageunit'
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
        if break add info  -> find first record of next year
        
        for a page : if previous year = cur = next: keep it 
        else:
            if prev = cur  and cur =next-1: find break
            if prev != cur: find break?
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
#     xpath  = "//%s[@number>= 310 and @number <= 312]" % ("PAGE")
    xpath  = "//%s" % ("PAGE")

    ltr= et.xpath(xpath)
#     print(len(ltr))
    ## concat all
    highest=0
    curYear=''
    ## forward then backward!
    backward = False
    nbElementsInCurrentYear=0
    llMonth=[]
    lenPage=len(ltr)
    dLineRecord={}
    for it,page in enumerate(ltr):
        print ('***'*10,it+1,curYear, page.get('number'))
        dLineRecord[it]={}
        parish=page.get('pagenum')
        # why do we have \col\None\S_Indersbach_003_0001
        #  col  None S_Indersbach_003_0001
        #            S Indersbach   003    0001
        parish = re.sub(r'_\d+', '',parish).replace('\\col\\None\\S_','')
        if page.get('years') !='':# and it+1 <len(ltr):# and ltr[it+1].get('years') !="":
            prev=curYear
            curYear =int(page.get('years'))
            #print ("-"*20,page.get('number'),curYear)
            if curYear !=prev: nbElementsInCurrentYear=0
        lmonth=[]
        xpath  = "./%s" % ("RECORD")
        llines= page.xpath(xpath)
        cpt=0
        for il,line in enumerate(llines):
            # for mapping month and lines!
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
            
            line.attrib['year']=str(curYear)
            
            # burial date is more chronological  but mroe noisy?
            month= line.attrib['deathdate']
            if month =="": month= line.attrib['burialdate']
            month = month.replace("Jäner",'Januar')
            month = month.replace("än",'an')
            if month:
                if day !="":xx ="%s %s "%(day ,month)
                else:xx =" %s "%(month) 
                xx =xx.strip()
                lmonth.append(xx) #,line.attrib['deathyear']))
                try:dLineRecord[it][cpt]=il
                except: dLineRecord[it]={};  dLineRecord[it][cpt]=il
                cpt+=1
            #if curYear:print (line.attrib['year'],curYear,lmonth)
        # process lmonth with taggerchrono
        llMonth.append(lmonth)
    # complete  if page i =X page + 2 =x   page i+1=X
    curYear=''
    prev=''
    nbElementsInCurrentYear=0
    for i, page in  enumerate(ltr):
        #print ('page %s'%page.get('number'))
        if page.get('years') !='':
            prev=curYear
            curYear =int(page.get('years'))
        if curYear !=prev: nbElementsInCurrentYear=0
        xpath  = "./%s" % ("RECORD")
        llines= page.xpath(xpath)
        
        lmonth = llMonth[i]
        if tagger and lmonth != []: 
            if i>0 and i< lenPage-1:
                iStart=0
                iEnd= len(lmonth)
                #threemonth=sum([llMonth[i-1],lmonth,llMonth[i+1]],[])
                threemonth=sum([lmonth,llMonth[i+1]],[])
                threemonth=threemonth[:29]
                lres = tagger.predict_multiptype(threemonth)
            else:
                threemonth=lmonth
                iStart=0
                iEnd=len(lmonth)
                lres = tagger.predict_multiptype(lmonth)
            prevmonthNumProba=0
            prevmonth=0
            #print (iStart, iEnd, lmonth,threemonth, threemonth [iStart:iEnd])
            lNbMonth=[]
            for ires,res  in enumerate(lres):
                # ires; iteration at "sentence" level: here one sentence!
                for irec,record in enumerate(res[iStart:iEnd]):
                    #print (curYear,record,len(res[iStart:iEnd])) 
                    nbElementsInCurrentYear+=1
                    monthNum = int(record[0][2][0])
                    monthNumProba = record[0][2][1]
                    dayNum = int(record[0][3][0])
                    dayProba = record[0][3][1]      
                    
                    ## gros hack for Juny/janer
                    if prevmonth == 12 and monthNum==6: monthNum=1
                    ##
                    try:nextProba= res[irec+1][0][2][1]
                    except IndexError:nextProba=0
                    try:nextMonth= int(res[irec+1][0][2][0])
                    except IndexError:nextMonth = 13                   
                    
                    #print (dayNum,dayProba,monthNum,monthNumProba,nextMonth,i,irec)
                    try:newiline= dLineRecord[i][irec]
                    except:
                        print(dLineRecord[i])
                        continue
                    llines[newiline].attrib['int_deathmonth']=str(monthNum)
                    lNbMonth.append(monthNum)
#                     llines[irec].attrib['int_deathday']=monthNum
                    if monthNum < highest and monthNumProba > 0.7 and (prevmonthNumProba > 0.5 or nextProba > 0.5):
                        bPageBreak=True
                        # not reliable enough
                        if (nextMonth and prevmonth == nextMonth):
                            #print ('NO BREAK',prevmonth,monthNum,nextMonth) 
                            continue
                        #print ('BREAK', curYear,highest, monthNum,monthNumProba,nbElementsInCurrentYear)
                        if curYear !='' and nbElementsInCurrentYear >3:
                            if backward:curYear =- 1
                            else: curYear += 1
                            nbElementsInCurrentYear=0
                        highest = monthNum
                    elif monthNum >highest and monthNumProba > 0.70  and prevmonthNumProba > 0.70:highest=monthNum
                    elif monthNumProba > 0.5  and (prevmonthNumProba > 0.5 or nextProba >0.5) :highest=monthNum
                    
                    if curYear !='':
                        llines[newiline].attrib['year']=str(curYear)
                        #print (curYear), etree.tostring(llines[newiline])
                    prevmonthNumProba = monthNumProba
                    prevmonth= monthNum
                mfmonth=  max(set(lNbMonth), key=lNbMonth.count)
                for line in llines:
                    try:line.attrib['int_deathmonth']
                    except:line.attrib['int_deathmonth']=str(mfmonth)
                    #hack : if no month: assign the most requent one in the page
        # replace deathDate by deathMonthNumber
    
    
    curYear=''
    highest = 0
    backward = True
    ltr.reverse()
    if False:
        for it,page in enumerate(ltr):
            if page.get('years') !='' and it+1 <len(ltr): # and ltr[it+1].get('years') !="":
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
                    if month:
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
                                    llines[irec].attrib['year']=str(curYear)
    #                                 print (curYear)
                            prevmonthNumProba = monthNumProba  


    
    et.write("%s.out" %(infile),xml_declaration=True, encoding='UTF-8')
    


if __name__ == "__main__":
    if sys.argv[1] == 'year':
        processDS(sys.argv[2])
    elif sys.argv[1] == 'norm':
        getPageBreak(sys.argv[2])
    else:
        print("usage: %s [year|norm] INFILE"%sys.argv[0])
        
    
    
    