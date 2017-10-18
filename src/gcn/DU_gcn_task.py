# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

import numpy as np
import tensorflow as tf

from sklearn.metrics import confusion_matrix
import os
import pickle
import os.path
from sklearn.metrics import classification_report
import random
import gcn_models
from gcn_datasets import GCNDataset

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train', '', "FilePath Train pickle file")
tf.app.flags.DEFINE_string('test', '', "FilePath for the pickle")
tf.app.flags.DEFINE_integer('fold', '1', "FilePath for the pickle")
tf.app.flags.DEFINE_string('out_dir', 'out_res', "outdirectory for saving the results")
tf.app.flags.DEFINE_integer('configid', 0, 'gridid')
tf.app.flags.DEFINE_bool('snake',False, 'whether to work on the snake dataset')

# Details of the training configuration.
tf.app.flags.DEFINE_float('learning_rate', 0.1, """How large a learning rate to use when training, default 0.1 .""")
tf.app.flags.DEFINE_integer('nb_iter', 3000, """How many training steps to run before ending, default 1.""")
tf.app.flags.DEFINE_integer('nb_layer', 1, """How many layers """)
tf.app.flags.DEFINE_integer('eval_iter', 256, """How often to evaluate the training results.""")
tf.app.flags.DEFINE_string('path_report', 'default', """Path for saving the results """)
tf.app.flags.DEFINE_string('grid_configs', '3_4_5', """Configs to be runned on all the folds """)
tf.app.flags.DEFINE_integer('qsub_taskid', -1, 'qsub_taskid')






#For Snake  python DU_gcn_task.py --snake=True --configid=22




def _make_grid_qsub(grid_qsub=0):

    if grid_qsub==0:
        tid=0
        C={}
        for fold_id in [1,2,3,4]:
            #for config in [3,4,5]:
            for config in [5]:
                C[tid]=(fold_id,config)
                tid+=1
        return C

    else:
        raise NotImplementedError



def get_config(config_id=0):
    config = {}

    if config_id == 0:
        config['nb_iter'] = 1000
        config['lr'] = 0.001
        config['stack_instead_add'] = True
        config['mu'] = 0.0
        config['num_layers'] = 1
        config['node_indim'] = -1
        config['nconv_edge'] = 1


    elif config_id==-1:
        #Debug Configuration with few iterations
        config['nb_iter'] = 10
        config['lr'] = 0.001
        config['stack_instead_add'] = True
        config['mu'] = 0.0
        config['num_layers'] = 2
        config['node_indim'] = -1
        config['nconv_edge'] = 10

    elif config_id==1:
        #config['nb_iter'] = 2000
        config['nb_iter'] = 1000
        config['lr'] = 0.001
        config['stack_instead_add'] = False
        config['mu'] = 0.0
        config['num_layers'] = 1
        config['node_indim'] = -1
        config['nconv_edge'] = 1

    elif config_id==2:
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['stack_instead_add'] = True
        config['mu'] = 0.0
        config['num_layers'] = 1
        config['node_indim'] = -1
        config['nconv_edge'] = 10

    elif config_id==3:
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['stack_instead_add'] = True
        config['mu'] = 0.0
        config['num_layers'] = 1
        config['node_indim'] = -1
        config['nconv_edge'] = 50

    elif config_id==4:
        #config['nb_iter'] = 2000
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['stack_instead_add'] = True
        config['mu'] = 0.0
        config['num_layers'] = 2
        config['node_indim'] = -1
        config['nconv_edge'] = 7

    elif config_id==5:
        #config['nb_iter'] = 2000
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['stack_instead_add'] = True
        config['mu'] = 0.0
        config['num_layers'] = 3
        config['node_indim'] = -1 #INDIM =2 not working here
        config['nconv_edge'] = 10

    #Projection
    elif config_id == 6:
        # config['nb_iter'] = 2000
        config['nb_iter'] = 1500
        config['lr'] = 0.001
        config['stack_instead_add'] = True
        config['mu'] = 0.1
        config['num_layers'] = 2
        config['node_indim'] = 20  # INDIM =2 not working here
        config['nconv_edge'] = 10

    #Config for snakes ..
    elif config_id == 7:
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['stack_instead_add'] = True
        config['mu'] = 0.1
        config['num_layers'] = 1
        config['node_indim'] = -1  # INDIM =2 not working here
        config['nconv_edge'] = 121
        #config['activation']=tf.tanh

    elif config_id == 8:
        config['nb_iter'] = 500
        config['lr'] = 0.001
        config['stack_instead_add'] = True
        config['mu'] = 0.0
        config['num_layers'] = 3
        config['node_indim'] = -1  # INDIM =2 not working here
        config['nconv_edge'] =10
        #config['snake']=True
        #config['activation'] = tf.tanh

    #Feature in snake. No way just to consider on neighbor , in table this is possible due to the type of feature , which are group
    ###########################################
    elif config_id == 9:
        config['nb_iter'] = 2000
        config['lr'] = 0.0005
        config['stack_instead_add'] = True
        config['mu'] = 0.0
        config['num_layers'] = 5
        config['node_indim'] = -1  # INDIM =2 not working here
        config['nconv_edge'] =4
        config['snake'] = True

    # Testing Regularization Effect ...
    # Back to Config 5 but with regularization
    elif config_id==10:
        #config['nb_iter'] = 2000
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['stack_instead_add'] = True
        config['mu'] = 0.001
        config['num_layers'] = 3
        config['node_indim'] = -1 #INDIM =2 not working here
        config['nconv_edge'] = 10

    elif config_id==11:
        #config['nb_iter'] = 2000
        config['nb_iter'] = 2000
        config['lr'] = 0.01
        config['stack_instead_add'] = True
        config['mu'] = 0.001
        config['num_layers'] = 3
        config['node_indim'] = -1 #INDIM =2 not working here
        config['nconv_edge'] = 10

    elif config_id==12:
        #config['nb_iter'] = 2000
        config['nb_iter'] = 2000
        config['lr'] = 0.1
        config['stack_instead_add'] = True
        config['mu'] = 0.001
        config['num_layers'] = 3
        config['node_indim'] = -1 #INDIM =2 not working here
        config['nconv_edge'] = 10

    #Config Deep
    elif config_id == 13:
        # config['nb_iter'] = 2000
        config['nb_iter'] = 2000
        config['lr'] = 0.1
        config['stack_instead_add'] = True
        config['mu'] = 0.0
        config['num_layers'] = 5
        config['node_indim'] = -1  # INDIM =2 not working here
        config['nconv_edge'] = 5

    #Test Residual Connection
    elif config_id == 14:
        # config['nb_iter'] = 2000
        config['nb_iter'] = 2000
        config['lr'] = 0.1
        config['stack_instead_add'] = True
        config['mu'] = 0.0
        config['num_layers'] = 3
        config['node_indim'] = -1  # INDIM =2 not working here
        config['nconv_edge'] = 10
        config['residual_connection']=True
    elif config_id == 15:
        # config['nb_iter'] = 2000
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['stack_instead_add'] = True
        config['mu'] = 0.0
        config['num_layers'] = 2
        config['node_indim'] = -1  # INDIM =2 not working here
        config['nconv_edge'] = 50
        config['shared_We']=True

    elif config_id == 16:
        # config['nb_iter'] = 2000
        config['nb_iter'] = 2000
        config['lr'] = 0.1
        config['stack_instead_add'] = True
        config['mu'] = 0.0
        config['num_layers'] = 3
        config['node_indim'] = -1  # INDIM =2 not working here
        config['nconv_edge'] = 10
        config['opti']=tf.train.AdagradOptimizer(config['lr'])

    elif config_id == 17:
        # config['nb_iter'] = 2000
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['stack_instead_add'] = True
        config['mu'] = 0.0
        config['num_layers'] = 3
        config['node_indim'] = -1  # INDIM =2 not working here
        config['nconv_edge'] = 10
        config['opti']=tf.train.RMSPropOptimizer(config['lr'])

    #Dropout Mode Test
    elif config_id==18:
        #config['nb_iter'] = 2000
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['stack_instead_add'] = True
        config['mu'] = 0.0
        config['num_layers'] = 3
        config['node_indim'] = -1 #INDIM =2 not working here
        config['nconv_edge'] = 10
        config['dropout_rate'] = 0.2 #means we keep with a proba of 0.8
        config['dropout_mode'] = 2

    #Dropout Edges..
    elif config_id==19:
        #config['nb_iter'] = 2000
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['stack_instead_add'] = True
        config['mu'] = 0.0
        config['num_layers'] = 3
        config['node_indim'] = -1 #INDIM =2 not working here
        config['nconv_edge'] = 10
        config['dropout_rate'] = 0.2 #means we keep with a proba of 0.8
        config['dropout_mode'] = 4

    elif config_id==20:
        #config['nb_iter'] = 2000
        config['nb_iter'] = 2000
        config['lr'] = 0.005
        config['stack_instead_add'] = True
        config['mu'] = 0.0
        config['num_layers'] = 3
        config['node_indim'] = -1 #INDIM =2 not working here
        config['nconv_edge'] = 10
        config['dropout_rate'] = 0.0
        config['dropout_mode'] = 0

    elif config_id==21:
        #config['nb_iter'] = 2000
        config['nb_iter'] = 2000
        config['lr'] = 0.0005
        config['stack_instead_add'] = True
        config['mu'] = 0.0
        config['num_layers'] = 3
        config['node_indim'] = -1 #INDIM =2 not working here
        config['nconv_edge'] = 10
        config['dropout_rate'] = 0.0
        config['dropout_mode'] = 0

    elif config_id == 22:
        config['nb_iter'] = 200
        config['lr'] = 0.001
        config['stack_instead_add'] = True
        config['mu'] = 0.0
        #config['num_layers'] = 2 #Mean Node  Accuracy 0.92
        #config['num_layers'] = 3 #Mean Node  Accuracy 0.9381
        config['num_layers'] = 9 # --> 9523 converges quickly
        config['node_indim'] = -1  # INDIM =2 not working here #Should add bias to convolutions, no ?
        config['nconv_edge'] =4  #Already by default
        config['snake']=True
        config['dropout_rate'] = 0.0
        config['dropout_mode'] = 0

    elif config_id == 23:
        config['nb_iter'] = 200
        config['lr'] = 0.0001
        config['stack_instead_add'] = True
        config['mu'] = 0.0
        #config['num_layers'] = 2 #Mean Node  Accuracy 0.92
        #config['num_layers'] = 3 #Mean Node  Accuracy 0.9381
        config['num_layers'] = 6  # --> 9523 converges quickly
        config['node_indim'] = -1  # INDIM =2 not working here #Should add bias to convolutions, no ?
        config['nconv_edge'] =4  #Already by default
        config['snake']=True
        config['dropout_rate'] = 0.1
        config['dropout_mode'] = 2

    elif config_id == 24:
        config['nb_iter'] = 800
        config['lr'] = 0.001
        config['stack_instead_add'] = True
        config['mu'] = 0.0
        config['num_layers'] = 9
        config['node_indim'] = -1  # INDIM =2 not working here
        config['nconv_edge'] =5
        config['dropout_rate'] = 0.0
        config['dropout_mode'] = 0
        #config['shared_We'] = True

    elif config_id == 25:
        config['nb_iter'] = 500
        config['lr'] = 0.001
        config['stack_instead_add'] = True
        config['mu'] = 0.0
        config['num_layers'] = 20
        config['node_indim'] = -1  # INDIM =2 not working here
        config['nconv_edge'] =2
        config['dropout_rate'] = 0.0
        config['dropout_mode'] = 0
        #config['shared_We'] = True


    else:
        raise NotImplementedError

    return config


def run_model(gcn_graph, config_params, gcn_graph_test,eval_iter=10):
    g_1 = tf.Graph()
    with g_1.as_default():

        node_dim = gcn_graph[0].X.shape[1]
        edge_dim = gcn_graph[0].E.shape[1] - 2.0
        nb_class = gcn_graph[0].Y.shape[1]


        if 'snake' in config_params:
            gcn_model = gcn_models.EdgeSnake(node_dim, edge_dim, nb_class,
                                                     num_layers=config_params['num_layers'],
                                                     learning_rate=config_params['lr'],
                                                     mu=config_params['mu'],
                                                     node_indim=config_params['node_indim'],
                                                     nconv_edge=config_params['nconv_edge'],
                                                     residual_connection=config_params[
                                                         'residual_connection'] if 'residual_connection' in config_params else False,
                                                     shared_We=config_params[
                                                         'shared_We'] if 'shared_We' in config_params else False,
                                                     dropout_rate=config_params[
                                                         'dropout_rate'] if 'dropout_rate' in config_params else 0.0,
                                                     dropout_mode=config_params[
                                                         'dropout_mode'] if 'dropout_mode' in config_params else 0,
                                                     )
        else:



            gcn_model = gcn_models.GCNModelGraphList(node_dim, edge_dim, nb_class,
                                                     num_layers=config_params['num_layers'],
                                                     learning_rate=config_params['lr'],
                                                     mu=config_params['mu'],
                                                     node_indim=config_params['node_indim'],
                                                     nconv_edge=config_params['nconv_edge'],
                                                     residual_connection=config_params[
                                                         'residual_connection'] if 'residual_connection' in config_params else False,
                                                     shared_We=config_params[
                                                         'shared_We'] if 'shared_We'  in config_params else False,
                                                     dropout_rate=config_params['dropout_rate'] if 'dropout_rate' in config_params else 0.0,
                                                     dropout_mode=config_params['dropout_mode'] if 'dropout_mode' in config_params else 0,
                                                     )
        if 'activation' in config_params:
            gcn_model.activation=config_params['activation']

        gcn_model.stack_instead_add = config_params['stack_instead_add']
        if 'opti' in config_params:
            gcn_model.optalg=config_params['opti']

        #print('################"')
        #print('## Decaying the learning Rate"')
        #print('################"')
        gcn_model.optim_mode=0
        gcn_model.create_model()


        train_acc=[]
        test_acc=[]


        with tf.Session() as session:
            session.run([gcn_model.init])
            for i in range(config_params['nb_iter']):
                random.shuffle(gcn_graph)
                if i % eval_iter == 0:
                    print('Epoch', i)
                    mean_acc = []
                    for g in gcn_graph:
                        #                        print('G Stats #node,#edge',g.X.shape[0],g.E.shape[0])
                        acc = gcn_model.test(session, g.X.shape[0], g.X, g.EA, g.Y, g.NA, verbose=False)
                        mean_acc.append(acc)
                    print('Mean Accuracy', np.mean(mean_acc))
                    train_acc.append(np.mean(mean_acc))


                    print('Test Error')
                    gcn_model.test_lG(session,gcn_graph_test)


                for g in gcn_graph:
                    gcn_model.train(session, g.X.shape[0], g.X, g.EA, g.Y, g.NA, n_iter=1)

                #print("Bigraph Train",'Adjacency Not set')
                #gcn_model.train_batch_lG(session,gcn_graph)


            mean_acc = []
            print('Training Error')
            for g in gcn_graph:
                print('G Stats #node,#edge', g.X.shape[0], g.E.shape[0])
                acc = gcn_model.test(session, g.X.shape[0], g.X, g.EA, g.Y, g.NA)
                mean_acc.append(acc)
            print('Mean Accuracy', np.mean(mean_acc))

            print('Test Error')
            mean_acc_test = []
            for g in gcn_graph_test:
                print('G Stats #node,#edge', g.X.shape[0], g.E.shape[0])
                acc = gcn_model.test(session, g.X.shape[0], g.X, g.EA, g.Y, g.NA)
                mean_acc_test.append(acc)
            print('Mean Accuracy', np.mean(mean_acc_test))
    return mean_acc_test





def run_model_train_val_test(gcn_graph,
                             config_params,
                             gcn_graph_test,
                             outpicklefname,
                             ratio_train_val=0.1
                             ):
    g_1 = tf.Graph()


    with g_1.as_default():

        node_dim = gcn_graph[0].X.shape[1]
        edge_dim = gcn_graph[0].E.shape[1] - 2.0
        nb_class = gcn_graph[0].Y.shape[1]

        gcn_model = gcn_models.GCNModelGraphList(node_dim, edge_dim, nb_class,
                                                 num_layers=config_params['num_layers'],
                                                 learning_rate=config_params['lr'],
                                                 mu=config_params['mu'],
                                                 node_indim=config_params['node_indim'],
                                                 nconv_edge=config_params['nconv_edge'],
                                                 residual_connection=config_params['residual_connection'] if 'residual_connection' in config_params else False

                                                 )

        gcn_model.stack_instead_add = config_params['stack_instead_add']

        gcn_model.create_model()


        #Split Training to get some validation
        #ratio_train_val=0.2
        split_idx= int(ratio_train_val*len(gcn_graph))
        random.shuffle(gcn_graph)
        gcn_graph_train=[]
        gcn_graph_val=[]

        gcn_graph_val.extend(gcn_graph[:split_idx])
        gcn_graph_train.extend(gcn_graph[split_idx:])

        with tf.Session() as session:
            session.run([gcn_model.init])

            R=gcn_model.train_with_validation_set(session,gcn_graph_train,gcn_graph_val,config_params['nb_iter'],eval_iter=10,patience=1000,graph_test=gcn_graph_test)

            f=open(outpicklefname,'wb')
            pickle.dump(R,f)
            f.close()



def main_fold(foldid,configid,outdir):
    pickle_train = '/opt/project/read/testJL/TABLE/abp_quantile_models/abp_CV_fold_' + str(
        foldid) + '_tlXlY_trn.pkl'
    pickle_test = '/opt/project/read/testJL/TABLE/abp_quantile_models/abp_CV_fold_' + str(foldid) + '_tlXlY_tst.pkl'

    train_graph = GCNDataset.load_transkribus_pickle(pickle_train)
    test_graph = GCNDataset.load_transkribus_pickle(pickle_test)

    config = get_config(configid)
    #acc_test = run_model(train_graph, config, test_graph)
    #print('Accuracy Test', acc_test)

    outpicklefname = os.path.join(FLAGS.out_dir, 'table_F' + str(FLAGS.fold) + '_C' + str(FLAGS.configid) + '.pickle')
    run_model_train_val_test(train_graph,
                             config,
                             test_graph,
                             outpicklefname)


    #To be improved ....the file result
    #mean_acc=np.mean(acc_test)
    #fresult_fname=os.path.join(outdir,'fold_'+str(foldid)+'_configid_'+str(configid)+'_acc_'+str(mean_acc))
    #os.system( 'touch '+fresult_fname)


def main(_):

    if FLAGS.snake is True:

        pickle_train = '/home/meunier/Snake/snake_tlXlY_edge_trn.pkl'
        pickle_test =  '/home/meunier/Snake/snake_tlXlY_edge_tst.pkl'

        #pickle_train = '/home/meunier/Snake/snake_tlXlY_trn.pkl'
        #pickle_test =  '/home/meunier/Snake/snake_tlXlY_tst.pkl'


        #pickle_train = '/home/meunier/Snake/snake_tlXlY_fixed_trn.pkl'
        #pickle_test =  '/home/meunier/Snake/snake_tlXlY_fixed_tst.pkl'


        #pickle_train='/home/meunier/Snake/snake_tlXlY_2_fixed_trn.pkl'
        #pickle_test='/home/meunier/Snake/snake_tlXlY_2_fixed_tst.pkl'

        train_graph = GCNDataset.load_snake_pickle(pickle_train)
        test_graph = GCNDataset.load_snake_pickle(pickle_test)

        config = get_config(FLAGS.configid)
        acc_test = run_model(train_graph, config, test_graph)
        print('Accuracy Test', acc_test)


    elif FLAGS.qsub_taskid >-1:

        GRID = _make_grid_qsub(0)

        try:
            fold_id,configid =GRID[FLAGS.qsub_taskid]
        except:
            print('Invalid Grid Parameters',FLAGS.qsub_taskid,GRID)
            return -1
        print('Experiement with FOLD',fold_id,' CONFIG',configid)
        pickle_train = '/opt/project/read/testJL/TABLE/abp_quantile_models/abp_CV_fold_' + str(
            fold_id) + '_tlXlY_trn.pkl'
        pickle_test = '/opt/project/read/testJL/TABLE/abp_quantile_models/abp_CV_fold_' + str(
            fold_id) + '_tlXlY_tst.pkl'

        train_graph = GCNDataset.load_transkribus_pickle(pickle_train)
        test_graph = GCNDataset.load_transkribus_pickle(pickle_test)

        config = get_config(configid)

        if os.path.exists(FLAGS.out_dir) is False:
            print('Creating Dir',FLAGS.out_dir)
            os.mkdir(FLAGS.out_dir)

        outpicklefname = os.path.join(FLAGS.out_dir, 'table_F' + str(fold_id) + '_C' + str(configid) + '.pickle')
        run_model_train_val_test(train_graph, config, test_graph, outpicklefname,ratio_train_val=0.1)


    else:

        if FLAGS.fold==-1:
            #Do it on all the fold for the specified configs
            FOLD_IDS=[1,2,3,4]
            sel_configs_ = FLAGS.grid_configs.split('_')
            sel_configs =  [int(x) for x in sel_configs_]
            print('GRID on FOLDS',FOLD_IDS)
            print('Model Configs', sel_configs)

            for cid in sel_configs:
                for fid in FOLD_IDS:
                    print('Running Fold',fid,'on Config',cid)
                    main_fold(fid,cid,FLAGS.out_dir)

        else:


            pickle_train = '/opt/project/read/testJL/TABLE/abp_quantile_models/abp_CV_fold_' + str(
                FLAGS.fold) + '_tlXlY_trn.pkl'
            pickle_test = '/opt/project/read/testJL/TABLE/abp_quantile_models/abp_CV_fold_' + str(FLAGS.fold) + '_tlXlY_tst.pkl'


            train_graph = GCNDataset.load_transkribus_pickle(pickle_train)
            print('Loaded Trained Graphs:',len(train_graph))
            test_graph = GCNDataset.load_transkribus_pickle(pickle_test)
            print('Loaded Test Graphs:', len(test_graph))

            config = get_config(FLAGS.configid)

            acc_test = run_model(train_graph, config, test_graph,eval_iter=1)
            print('Accuracy Test', acc_test)

            #outpicklefname=os.path.join(FLAGS.out_dir,'table_F'+str(FLAGS.fold)+'_C'+str(FLAGS.configid)+'.pickle')
            #run_model_train_val_test(train_graph,config,test_graph,outpicklefname)



if __name__ == '__main__':
    tf.app.run()
