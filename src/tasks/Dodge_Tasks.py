



import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.getcwd()) ) )
print sys.path
import itertools
import pickle
import string

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    #sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    sys.path.append( os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd()) )) )
    import TranskribusDU_version


import DU_Dodge
#This is a class method ...
# Dodge_Graph should herit from this actually ...
#Or split the things where graph should not be load differently
#print("- classes: ", DU_Dodge.DU_DODGE_GRAPH.getLabelNameList())

#print('After')
#print sys.argv[0]

from DU_BL_Task import DU_BL_Task
#print("- classes: ", DU_Dodge.DU_DODGE_GRAPH.getLabelNameList())

from IPython import embed
from DU_CRF_Task import DU_CRF_Task

from crf.FeatureDefinition_PageXml_FeatSelect import FeatureDefinition_PageXml_FeatSelect
import commands

_prefix_root = '/opt/project/dodge/data/plan/'

dodge_collections = ['DVD1', 'DVD2', 'DVD3', 'DVD4',
               'Plans_for_Grenoble2',
               'Plans_for_Grenoble3',
               'Plans_for_Grenoble4',
               'Plans_for_Grenoble5',
               'Plans_for_Grenoble6',
               'Plans_for_Grenoble7',]



def picklefile(fname):
    return pickle.load(open(fname))

def get_collection(collection_name,prefix=_prefix_root):
    path_collection =os.path.join(prefix,collection_name,'test_gl','out')
    ls_res=commands.getoutput('ls '+path_collection+'/GraphML_R33*ds.xml')
    return ls_res.split('\n')


def getDodgeFileList0():
    return ['/opt/project/dodge/data/plan/DVD1/test_gl/out/GraphML_R33_v06a_2298385_000_SP_ds.xml',
            '/opt/project/dodge/data/plan/DVD1/test_gl/out/GraphML_R33_v06a_2304654_000_SP_ds.xml',
            '/opt/project/dodge/data/plan/DVD1/test_gl/out/GraphML_R33_v06a_2305189_000_SP_ds.xml',
            ]


def getDodgeFileList1():
    '''
    Represent a first test set
    :return:
    '''
    return ['/opt/project/dodge/data/plan/DVD1/test_gl/out/GraphML_R33_v06a_2298618_000_SP_ds.xml',
            '/opt/project/dodge/data/plan/DVD1/test_gl/out/GraphML_R33_v06a_2304956_000_SP_ds.xml',
            #'/opt/project/dodge/data/plan/DVD1/test_gl/out/GraphML_R33_v06a_2293321.xml',
            ]


    #return ['/opt/project/dodge/data/plan/DVD1/test_gl/out/GraphML_R33_v06a_2298618_000_SP_ds.xml',
    #        ]


dFeatureConfig_Baseline = {'n_tfidf_node': 10000
    , 't_ngrams_node': (2, 4)
    , 'b_tfidf_node_lc': False
    , 'n_tfidf_edge': 250
    , 't_ngrams_edge': (2, 4)
    , 'b_tfidf_edge_lc': False,
                           }


dFeatureConfig_FeatSelect = {'n_tfidf_node': 500
    , 't_ngrams_node': (2, 4)
    , 'b_tfidf_node_lc': False
    , 'n_tfidf_edge': 250
    , 't_ngrams_edge': (2, 4)
    , 'b_tfidf_edge_lc': False
    , 'feat_select':'chi2'
                             }


dLearnerConfig={
                            'C': .1
                            , 'njobs': 4
                            , 'inference_cache': 50
                            , 'tol': .1
                            , 'save_every': 50  # save every 50 iterations,for warm start
                            , 'max_iter': 250
                        }


class DU_BL_Dodge(DU_BL_Task):
    sXmlFilenamePattern = "GraphML_R33*_ds.xml"

    # === CONFIGURATION ====================================================================
    def __init__(self, sModelName, sModelDir, feat_select=None,sComment=None):

        if feat_select=='chi2':
            DU_BL_Task.__init__(self, sModelName, sModelDir,
                                DU_Dodge.DU_DODGE_GRAPH,
                                dFeatureConfig=dFeatureConfig_FeatSelect,
                                dLearnerConfig=dLearnerConfig,
                                sComment=sComment
                                )
        else:
            DU_BL_Task.__init__(self, sModelName, sModelDir,
                                DU_Dodge.DU_DODGE_GRAPH,
                                dFeatureConfig=dFeatureConfig_Baseline,
                                dLearnerConfig=dLearnerConfig,
                                sComment=sComment
                                )

        self.addFixedLR()



def simpleTest():
    listTrain = getDodgeFileList0()
    modelName = 'baselineDodge_3'
    modeldir = 'UT_model'
    #Baseline Model
    doer = DU_BL_Dodge(modelName, modeldir)
    doer.addFixedLR()

    #doer = DU_Dodge.DU_Dodge(modelName, modeldir)




    #First Check Test with Dodge
    #doer = DU_Dodge.DU_Dodge(modelName,modeldir)
    # Remove Previous file in anya
    doer.rm()

    listTest = getDodgeFileList1()

    print("- classes: ", DU_Dodge.DU_DODGE_GRAPH.getLabelNameList())

    report=doer.train_save_test(listTrain, listTest, bWarm=False, filterFilesRegexp=False)
    print(report)
    node_transformer,edge_transformer =doer._mdl.getTransformers()

    print(node_transformer)

    #The Feature definition object is instantiated but not saved after
    #Only the node_transformer and the edge transformer
    ngrams_selected=FeatureDefinition_PageXml_FeatSelect.getNodeTextSelectedFeatures(node_transformer)
    print('########  Tokens Selected')



def train_dodge_collection(collection_name,feat_select=None):
    #Baseline Model
    listTrain = get_collection(collection_name)
    if feat_select:
        if feat_select=='chi2':
            doer = DU_BL_Dodge(collection_name+'chi2', 'DODGE_TRAIN',feat_select='chi2')
    else:
        doer = DU_BL_Dodge(collection_name, 'DODGE_TRAIN')


    #doer.addFixedLR()
    doer.addBaseline_LogisticRegression()
    report=doer.train_save_test(listTrain, [], bWarm=False, filterFilesRegexp=False)
    print(report)



def test_model(model_name,model_dir,target_collection):
    listTest = get_collection(target_collection)
    doer = DU_BL_Dodge(model_name,model_dir)
    doer.load()
    tstReport = doer.test(listTest,filterFilesRegexp=False)
    rep=tstReport[0]

    #I only save the first report ....
    rep.name='Train_'+model_name+'_TEST_'+target_collection
    if os.path.exists('reports') is False:
        os.mkdir('reports')

    print('Saving files in reports')
    f=open(os.path.join('reports',rep.name+'.pickle'),'w')
    pickle.dump(rep,f)
    f.close()

    return tstReport


def create_training_plan_pickle():
    #create a pickle file with all the list of test files
    #seems to have troubles with 'DVD2', 'DVD3', 'Plans_for_Grenoble6',
    sel_collections = ['DVD1', 'DVD4',
               'Plans_for_Grenoble2',
               'Plans_for_Grenoble3',
               'Plans_for_Grenoble4',
               'Plans_for_Grenoble5',
               'Plans_for_Grenoble7',]

    AD=list(itertools.product(sel_collections,['_tf','_chi2','_mi']))
    AD_tmp=[]
    for coll,fs in AD:
        if fs=='_tf':
            for k  in [500,1000,10000]:
                AD_tmp.append((coll,fs,k))
        else:
            for k in [500,1000]:
                AD_tmp.append((coll,fs,k))

    AD=AD_tmp
    AD=sorted(AD,key= lambda x : x[0])
    DATASETSID={}
    i=0
    for da in AD:
        DATASETSID[i]=da
        i+=1
    print('Generated',i,' Tasks')
    outname='dodge_train_plan.pickle'
    f=open(outname,'w')
    pickle.dump(DATASETSID,f)
    f.close()


def create_test_plan_pickle():
    #create a pickle file with all the list of test files
    #seems to have troubles with 'DVD2', 'DVD3', 'Plans_for_Grenoble6',
    sel_collections = ['DVD1', 'DVD4',
               'Plans_for_Grenoble2',
               'Plans_for_Grenoble3',
               'Plans_for_Grenoble4',
               'Plans_for_Grenoble5',
               'Plans_for_Grenoble7',]

    AD=list(itertools.product(sel_collections,['','chi2'],sel_collections,))
    AD=filter(lambda x : x[0]!=x[2],AD)
    AD=sorted(AD,key= lambda x : x[0])
    DATASETSID={}
    i=0
    for da in AD:
        DATASETSID[i]=da
        i+=1
    print('Generated',i,' Tasks')
    outname='dodge_test_plan.pickle'
    f=open(outname,'w')
    pickle.dump(DATASETSID,f)
    f.close()





import time
def qsub_test_plan(taskid):
    #source /usr/local/grid/XRCE/common/settings.sh
    exec_path="/opt/MLS_db/usr/sclincha/Transkribus/src/tasks/make_dodge_test_task.sh"
    exp_name = 'DT'+str(taskid)
    #cmd_str='qsub -l h='(floriad|alabama|alaska|ontario)  -o /opt/scratch/MLS/sclincha/sge_logs/ -e /opt/scratch/MLS/sclincha/sge_logs/ -m a -cwd -N ' +exp_name+ ' -l vf=32G,h_vmem=32G '\
    #cmd_str='qsub -l h=\'(florida|alabama|ohio|alaska|ontario|arizona|california|nevada|chichet|oregon|montana|colorado|kansas|iowa|indiana)\'  -o /opt/scratch/MLS/sclincha/sge_logs/ -e /opt/scratch/MLS/sclincha/sge_logs/ -m a -cwd -N ' +exp_name + ' -l vf=48G,h_vmem=48G ' +exec_path +' '+str(taskid)
    cmd_str='qsub  -o /opt/scratch/MLS/sclincha/sge_logs/ -e /opt/scratch/MLS/sclincha/sge_logs/ -m a -cwd -N ' +exp_name + ' -l vf=48G,h_vmem=48G ' +exec_path +' '+str(taskid)
    print cmd_str
    os.system(cmd_str)

#TEST Feature selection ..





if __name__ == '__main__':

    if len(sys.argv)>=2:
        mode =sys.argv[1]
        if mode =='scratch':
            simpleTest()
        elif mode =='dodge_train':
            train_collection = sys.argv[2]
            if train_collection.endswith('_chi2'):
                train_collection = string.split(train_collection,'_chi2')[0]
                train_dodge_collection(train_collection,'chi2')
            else:
                train_dodge_collection(train_collection)

        elif mode =='dodge_train_chi2':
            tmp = sys.argv[2]
            #remove the _chi2
            train_collection=tmp[:-5:]
            train_dodge_collection(train_collection,'chi2')
        elif mode=='dodge_test':
            model_name= sys.argv[2]
            model_dir =sys.argv[3]
            target_collection =sys.argv[4]
            test_model(model_name,model_dir,target_collection)
        elif mode=='make_test':
            pickle_fname =sys.argv[2]
            task_index   =int(sys.argv[3])
            Z=pickle.load(open(pickle_fname))
            model_src,feat_select,test_collection=Z[task_index]
            model_name=model_src+feat_select
            model_dir='DODGE_TRAIN'
            res=test_model(model_name,model_dir,test_collection)
            print(res)
        else:
            raise Exception('Invalid Mod Selected')
    else:
        embed()
        print('Embed ...')


'''
import Dodge_Tasks
model_name='Plans_for_Grenoble3'
model_dir='DODGE_TRAIN/'
doer = DU_BL_Dodge(model_name,model_dir)
doer = Dodge_Tasks.DU_BL_Dodge(model_name,model_dir)
doer
doer.load()
doer.mdl
doer._mdl
A=doer._mdl.getTransformers()
A
listTest = get_collection(model_name)
listTest = Dodge_Tasks.get_collection(model_name)
listTest
tstReport = doer.test(listTest,filterFilesRegexp=False)
tstReport
tstReport.toStr()
tstReport[0].tstReport
str(tstReport[0])
print(tstReport[0])
'''

#Memory
#TFIDF Vectorizer takes the most frequents and then apply a TFiDF transformer
#so it is the logic of a CountVectorizer
#Use a random Forest
#Maybe Do a real model with Grid Search ....
#TODO Test this on READ ...