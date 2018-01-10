from Dodge_Tasks import *
import pickle


def get_crf_jobid():
    Z=pickle.load(open('dodge_train_crf_plan.pickle'))
    L=[]
    for task_index in Z:
        model_src,feat_select,nb_feat,mid=Z[task_index]
        if mid=='crf':
            L.append(task_index)
    return L


def train_crf_jobs_on_machine(joblist):
    for taskid in joblist:
        os.system('python Dodge_Tasks.py make_train dodge_train_crf_plan.pickle '+str(taskid)+' >& /opt/scratch/MLS/sclincha/sge_logs/'+str(taskid)+'.log  &' )



def get_crf_test_jobid():
    Z=pickle.load(open('dodge_test_plan.pickle'))
    CRF_jobs = filter(lambda x : x[1][3]=='crf',Z.items())
    CRF_jobid=[x[0] for x in CRF_jobs]
    return CRF_jobid





def test_crf_jobs_on_machine(joblist):
    for taskid in joblist:
        print(taskid)
        os.system('python Dodge_Tasks.py make_test dodge_test_plan.pickle '+str(taskid)+'  >& logs/'+str(taskid)+'.log  &' )
        time.sleep(0.1)



def run_chi2_mi_exp():
    Z=pickle.load(open('dodge_train_plan.pickle'))
    for task_index in Z:
        model_src,feat_select,nb_feat=Z[task_index]
        model_name=get_model_name(model_src,feat_select,nb_feat)

        if feat_select=='chi2' or feat_select =='mi_rr':
            qsub_train_plan(task_index)
            time.sleep(0.1)

def run_train_tf(tf_feat=1000):
    Z=pickle.load(open('dodge_train_plan.pickle'))
    for task_index in Z:
        model_src,feat_select,nb_feat=Z[task_index]
        model_name=get_model_name(model_src,feat_select,nb_feat)

        if feat_select=='tf' and nb_feat ==tf_feat:
            qsub_train_plan(task_index)
            time.sleep(0.5)


def train_missing_models():
    Z=pickle.load(open('dodge_train_crf_plan.pickle'))
    for task_index in Z:
        model_src,feat_select,nb_feat,mid=Z[task_index]
        model_name=get_model_name(model_src,feat_select,nb_feat,mid=mid)
        if mid !='crf':
            model_file =os.path.join("./DODGE_TRAIN",model_name+'_baselines.pkl')
            print model_file
            if os.path.exists(model_file) is False:
                qsub_train_plan(task_index)
                time.sleep(0.1)
        else:
            model_file =os.path.join("./DODGE_TRAIN",model_name+'_model.pkl')
            print model_file
            if os.path.exists(model_file) is False:
                qsub_train_plan(task_index)
                time.sleep(0.1)


def retrain_rr_models():
    Z=pickle.load(open('dodge_train_crf_plan.pickle'))
    for task_index in Z:
        model_src,feat_select,nb_feat,mid=Z[task_index]
        model_name=get_model_name(model_src,feat_select,nb_feat,mid=mid)
        if feat_select.endswith('rr'):
            print(model_name)
            qsub_train_plan(task_index)
            time.sleep(0.1)



def run_missing_exp():
    #TODO Memory
    Z=pickle.load(open('dodge_test_plan.pickle'))
    for task_index in Z:
        model_src,feat_select,nb_feat,mid,test_collection=Z[task_index]
        model_name=get_model_name(model_src,feat_select,nb_feat,mid)
        #model_dir='DODGE_TRAIN'

        repname='Train_'+model_name+'_TEST_'+test_collection
        report_fname=os.path.join('reports',repname+'.pickle')
        print(model_name,' testing on ',test_collection)
        if os.path.exists(report_fname) is False:
            qsub_test_plan(task_index)
            time.sleep(0.5)



if __name__ == '__main__':
    print('Embed ...')
    S=pickle.load(open('dodge_train_crf_plan.pickle'))
    Z =pickle.load(open('dodge_test_plan.pickle'))
    embed()




#Memory
#TFIDF Vectorizer takes the most frequents and then apply a TFiDF transformer
#so it is the logic of a CountVectorizer
#Use a random Forest
#Maybe Do a real model with Grid Search ....
#TODO Test this on READ ...