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
import Dodge_Tasks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def getFilesList_baseline():
    listM = commands.getoutput('ls reports/*| grep -v chi2')
    files = string.split(listM,'\n')
    return files

_M=[('tf',500),('tf',1000),('tf',10000),('chi2',500),('chi2',1000),('mi_rr',500),('mi_rr',1000),('chi2_rr',500),('chi2_rr',1000)]
#Deprecated
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


#Deprecated
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

def tasks():
    sel_collections = ['DVD1', 'DVD4',
               'Plans_for_Grenoble2',
               'Plans_for_Grenoble3',
               'Plans_for_Grenoble4',
               'Plans_for_Grenoble5',
               'Plans_for_Grenoble7',]

    AD=list(itertools.product(sel_collections,sel_collections,))
    AD=filter(lambda x : x[0]!=x[1],AD)

    #Models
    #M=[('tf',500),('tf',1000),('tf',10000),('chi2',500),('chi2',1000),('mi_rr',500),('mi_rr',1000)]
    #M=[('tf',500),('tf',1000),('tf',10000),('chi2',500),('chi2',1000),('mi_rr',500),('mi_rr',1000),('chi2_rr',500),('chi2_rr',1000)]
    M=_M
    #Measure Accuracy, Macro Average Precision Macro F1
    ALL_PERF=[]
    for i,c in enumerate(AD):
        train,test=c[0],c[1]
        task_perf =[i,train,test]
        for feat_select,nb_feat in M:
            m_perf = read_report(train,test,feat_select,nb_feat)
            for p in m_perf:
                task_perf.append(p)
        ALL_PERF.append(task_perf)
    return ALL_PERF

def read_report(train,test,feat_select,nb_feat,report_dir='./reports'):
    model_name=Dodge_Tasks.get_model_name(train,feat_select,nb_feat)

    repname='Train_'+model_name+'_TEST_'+test+'.pickle'

    rep_path=os.path.join(report_dir,repname)

    if os.path.exists(rep_path):
        report=pickle.load(open(rep_path))

        perf =[np.nan,np.nan,np.nan]

        perf[0]=report.fScore
        perf[1]=np.mean(report.average_precision)

        #Recompute f1
        confmat=report.aConfusionMatrix
        #rows are groundtruth
        #col are predictions
        eps=1e-8

        Precision = np.diag(confmat)/(eps+confmat.sum(axis=0))
        Recall    = np.diag(confmat)/(eps+confmat.sum(axis=1))
        print Precision
        print Recall

        F1        = 2*Precision*Recall/(Precision+Recall)
        perf[2]=np.mean(F1)
        return perf

    else:
        return [np.nan,np.nan,np.nan]


def get_headers_name():
    #AD=list(itertools.product([('tf',500),('tf',10000),('chi2',500),('chi2',1000),('mi_rr',500),('mi_rr',1000)],['ACC','MAP','F1']))
    AD=list(itertools.product(_M,['ACC','MAP','F1']))
    H=['id','train','test']
    for models,eval_metric in AD:
        H.append(str(models[0])+'_'+str(models[1])+':'+eval_metric )

    return H


def plot_diff(df_selection):

    df_selection.plot(kind='bar',ylim=[0.6,df_selection.max().max()+0.1])
    ri =np.isfinite(df_selection.values.sum(axis=1))

    m=np.mean(df_selection.values[ri],axis=0)

    plt.xlabel('Task index')
    plt.ylabel('Macro Average of F1')

    plt.title('Mean over Task:'+str(m))
    plt.grid(True)
    plt.show()






if __name__=='__main__':
    if len(sys.argv)>=2:
        mode =sys.argv[1]
        if mode =='chi2_acc':
            #Deprecated
            print_comparative_accuracies()
        elif mode =='chi2_avgp':
            #Deprecated
            print_comparative_average_precision()
        elif mode=='print':
            report_file=sys.argv[2]
            report=pickle.load(open(report_file))
            print(report)

        elif mode=='genreport':
            A=tasks()
            H=get_headers_name()
            df=pd.DataFrame(A,columns=H)
            f=open('/home/sclincha/Desktop/exp_dodge.csv','w')
            df.to_csv(f)
            f.close()

        else:
            raise ValueError('Invalid Mode')


    else:
        A=tasks()
        H=get_headers_name()
        df=pd.DataFrame(A,columns=H)
        embed()
        print('Embed ...')