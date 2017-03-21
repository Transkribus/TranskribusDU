
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
from DU_CRF_Task import DU_CRF_Task,DU_CRF_FS_Task
import dodge_graph
from DU_Dodge_c import DU_Dodge_c
#print("- classes: ", DU_Dodge.DU_DODGE_GRAPH.getLabelNameList())

try:
    from IPython import embed
except:
    print('Could not Load Ipython')

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
    return [#'/opt/project/dodge/data/plan/DVD1/test_gl/out/GraphML_R33_v06a_2298385_000_SP_ds.xml',
            #'/opt/project/dodge/data/plan/DVD1/test_gl/out/GraphML_R33_v06a_2304654_000_SP_ds.xml',
            '/opt/project/dodge/data/plan/DVD1/test_gl/out/GraphML_R33_v06a_2305189_000_SP_ds.xml',
            ]


def getDodgeFileList1():
    '''
    Represent a first test set
    :return:
    '''
    return ['/opt/project/dodge/data/plan/DVD1/test_gl/out/GraphML_R33_v06a_2298618_000_SP_ds.xml',
            #'/opt/project/dodge/data/plan/DVD1/test_gl/out/GraphML_R33_v06a_2304956_000_SP_ds.xml',
            #'/opt/project/dodge/data/plan/DVD1/test_gl/out/GraphML_R33_v06a_2293321.xml',
            ]


    #return ['/opt/project/dodge/data/plan/DVD1/test_gl/out/GraphML_R33_v06a_2298618_000_SP_ds.xml',
    #        ]


dFeatureConfig_Baseline = {'n_tfidf_node': 500
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
    def __init__(self, sModelName, sModelDir, feat_select=None,nb_feat=500,sComment=None):

        if feat_select is not None and feat_select!='tf':
            paramsFeatSelect=dict(dFeatureConfig_FeatSelect)
            paramsFeatSelect['feat_select']=feat_select
            paramsFeatSelect['n_tfidf_node']=nb_feat

            DU_BL_Task.__init__(self, sModelName, sModelDir,
                                dodge_graph.DU_GRAPH,
                                dFeatureConfig=paramsFeatSelect,
                                dLearnerConfig=dLearnerConfig,
                                sComment=sComment
                                )
        else:
            paramsBaseline=dict(dFeatureConfig_Baseline)
            paramsBaseline['n_tfidf_node']=nb_feat

            DU_BL_Task.__init__(self, sModelName, sModelDir,
                                dodge_graph.DU_GRAPH,
                                dFeatureConfig=paramsBaseline,
                                dLearnerConfig=dLearnerConfig,
                                sComment=sComment
                                )

        self.addFixedLR()

class DU_CRF_FS_Dodge(DU_CRF_FS_Task):
    sXmlFilenamePattern = "GraphML_R33*_ds.xml"

    # === CONFIGURATION ====================================================================
    def __init__(self, sModelName, sModelDir, feat_select=None,nb_feat=500,sComment=None):

        if feat_select is not None and feat_select!='tf':
            paramsFeatSelect=dict(dFeatureConfig_FeatSelect)
            paramsFeatSelect['feat_select']=feat_select
            paramsFeatSelect['n_tfidf_node']=nb_feat

            DU_CRF_FS_Task.__init__(self, sModelName, sModelDir,
                                dodge_graph.DU_GRAPH,
                                dFeatureConfig=paramsFeatSelect,
                                dLearnerConfig=dLearnerConfig,
                                sComment=sComment
                                )
        else:
            paramsBaseline=dict(dFeatureConfig_Baseline)
            paramsBaseline['n_tfidf_node']=nb_feat

            DU_CRF_FS_Task.__init__(self, sModelName, sModelDir,
                                dodge_graph.DU_GRAPH,
                                dFeatureConfig=paramsBaseline,
                                dLearnerConfig=dLearnerConfig,
                                sComment=sComment
                                )






def simpleTest():
    listTrain = getDodgeFileList0()
    modelName = 'baselineDodge_3'
    modeldir = 'UT_model'
    #Baseline Model

    use_bl_model=True
    use_crf_model=True

    if use_bl_model:

        doer = DU_BL_Dodge(modelName, modeldir,'chi2_rr',500)
        #doer.addBaseline_LogisticRegression()
        doer.addFixedLR()
        doer.rm()
        #doer.load()
        #doer = DU_Dodge.DU_Dodge(modelName, modeldir)

        #By Default Do Feature Selection with chi2
        #doer = DU_Dodge.DU_Dodge_CRF_FS(modelName, modeldir)
        #First Check Test with Dodge
        #doer = DU_Dodge.DU_Dodge(modelName,modeldir)
        #Remove Previous file in anya
        #doer.rm()
        #doer.load()

        listTest = getDodgeFileList1()

        print("- classes: ", dodge_graph.DU_GRAPH.getLabelNameList())

        report=doer.train_save_test(listTrain, [], bWarm=False, filterFilesRegexp=False)
        #report = doer.test(listTest,filterFilesRegexp=False)

        #This is a copy of the test function ... not clean that
        repname='Debug_Report'
        tstReport=doer.test(listTrain,filterFilesRegexp=False,test_sequential=False)
        rep=tstReport[0]
        rep.name=repname
        print rep


    if use_crf_model:
        doer = DU_Dodge.DU_Dodge(modelName,modeldir)
        #Remove Previous file in anya
        doer.rm()
        #doer.load()

        listTest = getDodgeFileList1()

        print("- classes: ", dodge_graph.DU_GRAPH.getLabelNameList())

        report=doer.train_save_test(listTrain, [], bWarm=False, filterFilesRegexp=False)
        #report = doer.test(listTest,filterFilesRegexp=False)

        #This is a copy of the test function ... not clean that
        repname='Debug_Report_CRF'

        tstReport=doer.test(listTrain,filterFilesRegexp=False)

        #rep=tstReport[0]
        rep=tstReport
        rep.name=repname
        print(rep)

    #Ytrue,Reports=doer.test_seq(listTest,filterFilesRegexp=False)

    #embed()
    #print Ytrue,Reports
    #print Reports[0]

    #print(report)

    #node_transformer,edge_transformer =doer._mdl.getTransformers()
    #print(node_transformer)

    #The Feature definition object is instantiated but not saved after
    #Only the node_transformer and the edge transformer
    #ngrams_selected=FeatureDefinition_PageXml_FeatSelect.getNodeTextSelectedFeatures(node_transformer)
    #print('########  Tokens Selected')



def train_dodge_collection(collection_name,feat_select=None,nb_feat=500,model_id=''):
    '''

    :param collection_name:
    :param feat_select:
    :param nb_feat:
    :param model_id: '' or bl Baseline Model crf else accepted
    :return:
    '''
    #Baseline Model
    listTrain = get_collection(collection_name)
    print(feat_select,nb_feat,model_id)


    MODEL_DIR='DODGE_TRAIN'
    '''
    if feat_select:
        model_name=get_model_name(collection_name,feat_select,nb_feat,model_id)
        print("Training...",collection_name,feat_select,nb_feat)
        if model_id=='' or model_id=='bl':
            doer = DU_BL_Dodge(model_name, MODEL_DIR,feat_select=feat_select,nb_feat=nb_feat)

        elif model_id=='crf':
            #This is not fully supported
            doer = DU_CRF_FS_Dodge(model_name,MODEL_DIR,feat_select=feat_select,nb_feat=nb_feat)

        else:
            raise ValueError('Invalid Model')

    else:

        if model_id=='' or model_id=='bl':
            model_name=get_model_name(collection_name,'tf',500,model_id)
            doer = DU_BL_Dodge(model_name,MODEL_DIR)


        elif model_id=='crf':
            model_name=get_model_name(collection_name,'tf',500,model_id)
            #This is not fully supported
            doer = DU_Dodge.DU_Dodge(model_name,MODEL_DIR)


        else:
            raise ValueError('Invalid Model')
    '''
    doer = create_model(collection_name,model_id,feat_select=feat_select,nb_feat=nb_feat,model_dir=MODEL_DIR)
    doer.addBaseline_LogisticRegression()
    report=doer.train_save_test(listTrain, [], bWarm=False, filterFilesRegexp=False)
    print(report)


#Kind of Factory
def create_model(collection_name,model_id='',feat_select=None,nb_feat=500,model_dir='DODGE_TRAIN'):

    if model_id=='crf_c':
        model_name=get_model_name(collection_name,'tf',0,model_id)
        doer = DU_Dodge_c(model_name,model_dir)
        return doer
    else:

        if feat_select:
            model_name=get_model_name(collection_name,feat_select,nb_feat,model_id)
            print('Creating Model:',collection_name,model_id,feat_select,nb_feat)
            if model_id=='' or model_id=='bl':
                doer = DU_BL_Dodge(model_name, model_dir,feat_select=feat_select,nb_feat=nb_feat)

            elif model_id=='crf':
                #TODO FIX This is not fully supported with all feature selection
                #This is just chi2scare
                doer = DU_CRF_FS_Dodge(model_name,model_dir,feat_select=feat_select,nb_feat=nb_feat)
            else:
                raise ValueError('Invalid Model')

        else:
            if model_id=='' or model_id=='bl':

                model_name=get_model_name(collection_name,'tf',500,model_id)
                doer = DU_BL_Dodge(model_name, model_dir)

            elif model_id=='crf':
                model_name=get_model_name(collection_name,'tf',500,model_id)

                doer = DU_Dodge.DU_Dodge(model_name,model_dir,feat_select=feat_select,nb_feat=nb_feat)
        return doer



def train_crf_dodge_collection(collection_name,feat_select=None,nb_feat=500):
    #Baseline Model
    listTrain = get_collection(collection_name)

    if feat_select:
        model_name=get_model_name(collection_name,feat_select,nb_feat)
        print("Training...",collection_name,feat_select,nb_feat)
        #doer = DU_Dodge.DU_Dodge(model_name, 'DODGE_CRF_TRAIN',feat_select=feat_select,nb_feat=nb_feat)
        #TODO Fixing this
        doer = DU_Dodge.DU_CRF_FS_Task(model_name, 'DODGE_CRF_TRAIN',feat_select=feat_select,nb_feat=nb_feat)

    else:
        doer = DU_Dodge.DU_Dodge(collection_name, 'DODGE_CRF_TRAIN')

    #doer.addFixedLR()
    doer.addBaseline_LogisticRegression()
    report=doer.train_save_test(listTrain, [], bWarm=False, filterFilesRegexp=False)
    print(report)


def test_model(model_name,model_dir,target_collection,model_id=''):

    #I only save the first report ....
    repname='Train_'+model_name+'_TEST_'+target_collection

    if os.path.exists('reports') is False:
        os.mkdir('reports')

    report_fname=os.path.join('reports',repname+'.pickle')

    if os.path.exists(report_fname) is False:

        listTest = get_collection(target_collection)
        #IS this correct just to load
        #doer = DU_BL_Dodge(model_name,model_dir)
        collection_name,feat_select,nb_feat,model_id=parse_model_name(model_name)
        doer = create_model(collection_name,model_id=model_id,feat_select=feat_select,nb_feat=nb_feat,model_dir=model_dir)
        doer.load()
        #tstReport = doer.test(listTest,filterFilesRegexp=False)
        #TODO
        #This break the baseline Models ..
        #tstReport=doer.test_files(listTest,filterFilesRegexp=False)
        #Should Do the seq test for the baseline model
        if model_id=='':
            tstReport=doer.test(listTest,filterFilesRegexp=False,test_sequential=False)
            rep=tstReport[0]
            rep.name=repname

        else:
            #Do the sequential test here
            tstReport=doer.test(listTest,filterFilesRegexp=False)
            #FIXME this is broken here as I did not generate the baseline report here
            #rep=tstReport[0]
            rep=tstReport
            rep.name=repname

        print('Saving files in reports')
        f=open(report_fname,'w')
        pickle.dump(rep,f)
        f.close()
        return tstReport
    else:
        print('Report Already Generated')
        f=open(report_fname)
        return pickle.load(f)



def create_training_plan_pickle():
    #create a pickle file with all the list of test files
    #seems to have troubles with 'DVD2', 'DVD3', 'Plans_for_Grenoble6',
    sel_collections = ['DVD1', 'DVD4',
               'Plans_for_Grenoble2',
               'Plans_for_Grenoble3',
               'Plans_for_Grenoble5',
               'Plans_for_Grenoble7',]

    #AD=list(itertools.product(sel_collections,['tf','chi2','mi_rr','chi2_rr']))
    AD=list(itertools.product(sel_collections,['tf','chi2','chi2_rr']))
    AD_tmp=[]
    for coll,fs in AD:
        '''
        if fs=='tf':
            for k  in [500,1000]:

                AD_tmp.append((coll,fs,k))
        else:
            for k in [500,1000]:
                AD_tmp.append((coll,fs,k))
        '''
        if fs in ['tf','chi2']:
            for k in [500,1000]:
                for mid in ['','crf']:
                    AD_tmp.append((coll,fs,k,mid))
        else:
            for k in [500,1000]:
                for mid in ['']:
                    AD_tmp.append((coll,fs,k,mid))
        AD_tmp.append((coll,'tf','0','crf_c'))

    AD=AD_tmp
    AD=sorted(AD,key= lambda x : x[0])
    DATASETSID={}
    i=0
    for da in AD:
        DATASETSID[i]=da
        i+=1
    print('Generated',i,' Tasks')
    outname='dodge_train_crf_plan.pickle'
    f=open(outname,'w')
    pickle.dump(DATASETSID,f)
    f.close()
    return DATASETSID





def create_test_plan_pickle():
    #create a pickle file with all the list of test files
    #seems to have troubles with 'DVD2', 'DVD3', 'Plans_for_Grenoble6',
    sel_collections = ['DVD1', 'DVD4',
               'Plans_for_Grenoble2',
               'Plans_for_Grenoble3',
               'Plans_for_Grenoble5',
               'Plans_for_Grenoble7',]
    '''
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
    '''
    f=open('dodge_train_crf_plan.pickle')
    Models=pickle.load(f)
    f.close()

    AD=[]
    for m in Models.values():
        collection_name = m[0]
        for coll in sel_collections:
            if coll!=collection_name:
                AD.append((m[0],m[1],m[2],m[3],coll))

    AD=sorted(AD,key= lambda x : x[0])
    TasksID={}
    i=0
    for da in AD:
        TasksID[i]=da
        i+=1
    print('Generated',i,' Tasks')
    outname='dodge_test_plan.pickle'
    f=open(outname,'w')
    pickle.dump(TasksID,f)
    f.close()

    return TasksID


import time
def qsub_test_plan(taskid,mem="48G"):
    #source /usr/local/grid/XRCE/common/settings.sh
    exec_path="/opt/MLS_db/usr/sclincha/Transkribus/src/tasks/make_dodge_test_task.sh"
    exp_name = 'DT'+str(taskid)
    #cmd_str='qsub -l h='(floriad|alabama|alaska|ontario)  -o /opt/scratch/MLS/sclincha/sge_logs/ -e /opt/scratch/MLS/sclincha/sge_logs/ -m a -cwd -N ' +exp_name+ ' -l vf=32G,h_vmem=32G '\
    #cmd_str='qsub -l h=\'(florida|alabama|ohio|alaska|ontario|arizona|california|nevada|chichet|oregon|montana|colorado|kansas|iowa|indiana)\'  -o /opt/scratch/MLS/sclincha/sge_logs/ -e /opt/scratch/MLS/sclincha/sge_logs/ -m a -cwd -N ' +exp_name + ' -l vf=48G,h_vmem=48G ' +exec_path +' '+str(taskid)
    cmd_str='qsub  -o /opt/scratch/MLS/sclincha/sge_logs/ -e /opt/scratch/MLS/sclincha/sge_logs/ -m a -cwd -N ' +exp_name + ' -l vf=1G,h_vmem=32G ' +exec_path +' '+str(taskid)
    #cmd_str='qsub  -o /dev/null -e /dev/null -m a -cwd -N ' +exp_name + ' -l vf=1G,h_vmem=32G ' +exec_path +' '+str(taskid)
    print cmd_str
    os.system(cmd_str)


def qsub_train_plan(taskid):
    #source /usr/local/grid/XRCE/common/settings.sh
    exec_path="/opt/MLS_db/usr/sclincha/Transkribus/src/tasks/make_dodge_train_task.sh"
    exp_name = 'DS'+str(taskid)
    #cmd_str='qsub -l h='(floriad|alabama|alaska|ontario)  -o /opt/scratch/MLS/sclincha/sge_logs/ -e /opt/scratch/MLS/sclincha/sge_logs/ -m a -cwd -N ' +exp_name+ ' -l vf=32G,h_vmem=32G '\
    #cmd_str='qsub -l h=\'(florida|alabama|ohio|alaska|ontario|arizona|california|nevada|chichet|oregon|montana|colorado|kansas|iowa|indiana)\'  -o /opt/scratch/MLS/sclincha/sge_logs/ -e /opt/scratch/MLS/sclincha/sge_logs/ -m a -cwd -N ' +exp_name + ' -l vf=48G,h_vmem=48G ' +exec_path +' '+str(taskid)
    cmd_str='qsub  -o /opt/scratch/MLS/sclincha/sge_logs/ -e /opt/scratch/MLS/sclincha/sge_logs/ -m a -cwd -N ' +exp_name + ' -l vf=1G,h_vmem=64G ' +exec_path +' '+str(taskid)
    #cmd_str='qsub  -l h=\'(texas|dakota)\' -o /dev/null -e /dev/null -m a -cwd -N ' +exp_name + ' -l vf=200G,h_vmem=200G ' +exec_path +' '+str(taskid)
    #cmd_str='qsub  -l h=\'(texas|dakota|alerta|ontario|sirac)\' -o /dev/null -e /dev/null -m a -cwd -N ' +exp_name + ' -l vf=128G ' +exec_path +' '+str(taskid)
    #cmd_str='qsub   -o /dev/null -e /dev/null -m a -cwd -N ' +exp_name + ' -l vf=64G,h_vmem=64G ' +exec_path +' '+str(taskid)
    print cmd_str
    os.system(cmd_str)





def parse_model_name(model_name):
    tok=string.split(model_name,':')
    if len(tok)==3:
        collection=tok[0]
        feat_select =tok[1]
        nb_feat     =int(tok[2])

        return collection,feat_select,nb_feat,''
    elif len(tok)==4:
        collection =  tok[0]
        feat_select = tok[1]
        nb_feat     =int(tok[2])
        mid        =  tok[3]

        return collection,feat_select,nb_feat,mid
    else:
        #raise ValueError('Invalid Model Name',model_name)
        return model_name,None,500



def get_model_name(collection,feat_select,nb_feat,mid=''):
    if mid=='':
        return string.join([collection,feat_select,str(nb_feat)],':')
    else:
        return string.join([collection,feat_select,str(nb_feat),mid],':')





if __name__ == '__main__':

    if len(sys.argv)>=2:
        mode =sys.argv[1]
        if mode =='scratch':
            simpleTest()
        elif mode =='dodge_train':
            train_collection = sys.argv[2]
            '''
            if train_collection.endswith('_chi2'):
                train_collection = string.split(train_collection,'_chi2')[0]
                train_dodge_collection(train_collection,'chi2')
            else:
                train_dodge_collection(train_collection)
            '''
            collection,feat_select,nb_feat,model_id = parse_model_name(train_collection)
            train_dodge_collection(collection,feat_select,nb_feat,model_id)

        elif mode=='dodge_train_crf':
            #Temporary Fix
            train_collection = sys.argv[2]
            '''
            if train_collection.endswith('_chi2'):
                train_collection = string.split(train_collection,'_chi2')[0]
                train_dodge_collection(train_collection,'chi2')
            else:
                train_dodge_collection(train_collection)
            '''
            collection,feat_select,nb_feat = parse_model_name(train_collection)
            train_crf_dodge_collection(collection,feat_select,nb_feat)


        elif mode=='dodge_test':
            model_name= sys.argv[2]
            model_dir =sys.argv[3]
            target_collection =sys.argv[4]
            test_model(model_name,model_dir,target_collection)
        elif mode=='make_test':
            pickle_fname =sys.argv[2]
            task_index   =int(sys.argv[3])
            Z=pickle.load(open(pickle_fname))
            model_src,feat_select,nb_feat,mid,test_collection=Z[task_index]
            model_name=get_model_name(model_src,feat_select,nb_feat,mid)
            model_dir='DODGE_TRAIN'
            print (model_name)
            res=test_model(model_name,model_dir,test_collection)
            print(res)

        elif mode=="make_train":
            pickle_fname =sys.argv[2]
            task_index   =int(sys.argv[3])
            Z=pickle.load(open(pickle_fname))
            collection_name,feat_select,nb_feat,mid=Z[task_index]
            train_dodge_collection(collection_name,feat_select,nb_feat,model_id=mid)
        else:
            raise Exception('Invalid Mod Selected')
    else:
        embed()
        print('Embed ...')



#Memory
#TFIDF Vectorizer takes the most frequents and then apply a TFiDF transformer
#so it is the logic of a CountVectorizer
#Use a random Forest
#Maybe Do a real model with Grid Search ....
#TODO Test this on READ ...