import pickle
import os
import sys
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.getcwd()) ) )
print sys.path
import itertools
import pickle

import commands
from IPython import embed
import string



def getFilesList_baseline():
    listM = commands.getoutput('ls reports/*| grep -v chi2')
    files = string.split(listM,'\n')
    return files



def print_comparative_accuracies():
    B=[]
    C=[]

    files=getFilesList_baseline()
    for f in files:
        #print f
        tok=string.split(f,'_TEST')
        chi2fname_tok=[tok[0]+'chi2']+tok[1:]
        chi2fname = string.join(chi2fname_tok,'_TEST')
        #print(f,chi2fname)
        if os.path.exists(chi2fname):
            #print('\t FOUND')
            baselinereport=pickle.load(open(f))
            chi2report    =pickle.load(open(chi2fname))
            #print baselinereport,chi2report
            name = f.replace('reports/','')
            name = name.replace('.pickle','')
            #print name, baselinereport.fScore,chi2report.fScore
            #B.append(baselinereport.fScore)
            #C.append(chi2report.fScore)
            print name,baselinereport.fScore,chi2report.fScore
        else:
            print('Could not find the results for',f)



def print_comparative_average_precision():
    #TODO Make this generic
    files=getFilesList_baseline()
    for f in files:
        #print f
        tok=string.split(f,'_TEST')
        chi2fname_tok=[tok[0]+'chi2']+tok[1:]
        chi2fname = string.join(chi2fname_tok,'_TEST')
        #print(f,chi2fname)
        if os.path.exists(chi2fname):
            #print('\t FOUND')
            baselinereport=pickle.load(open(f))
            chi2report    =pickle.load(open(chi2fname))
            #print baselinereport,chi2report
            name = f.replace('reports/','')
            name = name.replace('.pickle','')
            print(name)
            if hasattr(baselinereport,'average_precision') and hasattr(chi2report,'average_precision'):
                for tuple_baseline,tuple_chi2 in zip(baselinereport.average_precision,chi2report.average_precision):
                    print(' '*3,tuple_baseline[0],tuple_baseline[1],tuple_chi2[0],tuple_chi2[1])
            else:
                print('-- Error Could not find average precision for baseline or chi2 selection ')
            print('-'*30)
        else:
            print('Could not find the results for',f)



if __name__=='__main__':
    if len(sys.argv)>=2:
        mode =sys.argv[1]
        if mode =='chi2_acc':
            print_comparative_accuracies()
        elif mode =='chi2_avgp':
            print_comparative_average_precision()
        elif mode=='print':
            report_file=sys.argv[2]
            report=pickle.load(open(report_file))
            print(report)
        else:
            raise Exception('Invalid Mode Selected')
    else:
        embed()
        print('Embed ...')