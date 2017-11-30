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
tf.app.flags.DEFINE_string('out_dir', '', "outdirectory for saving the results")
tf.app.flags.DEFINE_integer('configid', 0, 'gridid')

# Details of the training configuration.
tf.app.flags.DEFINE_float('learning_rate', 0.1, """How large a learning rate to use when training, default 0.1 .""")
tf.app.flags.DEFINE_integer('nb_iter', 3000, """How many training steps to run before ending, default 1.""")
tf.app.flags.DEFINE_integer('nb_layer', 1, """How many layers """)
tf.app.flags.DEFINE_bool('stack_PN', False, "whehter to concator add the vector induced from the neighbors")
tf.app.flags.DEFINE_integer('eval_iter', 256, """How often to evaluate the training results.""")
tf.app.flags.DEFINE_string('path_report', 'default', """Path for saving the results """)
tf.app.flags.DEFINE_string('grid_configs', '0_1_2_3', """Configs to be runned on all the folds """)



def get_config(config_id=0):
    config = {}

    if config_id == 0:
        #1 Layer no dim reduction
        config['nb_iter'] = 1000
        config['lr'] = 0.001
        config['mu'] = 0.0
        config['num_layers'] = 1
        config['node_indim'] = -1
        config['dropout_p'] = 0.0
        config['dropout_mode'] = 0

    elif config_id==1:
        # 3 Layer no dim reduction
        config['nb_iter'] = 1000
        config['lr'] = 0.001
        config['mu'] = 0.0
        config['num_layers'] = 2
        config['node_indim'] = -1
        config['dropout_p'] = 0.0
        config['dropout_mode'] = 0

    elif config_id==2:
        # 3 Layer no dim reduction
        config['nb_iter'] = 1000
        config['lr'] = 0.001
        config['mu'] = 0.0
        config['num_layers'] = 3
        config['node_indim'] = -1
        config['dropout_p'] = 0.0
        config['dropout_mode'] = 0

    elif config_id==3:
        # 2 Layer dim reduction
        config['nb_iter'] = 1000
        config['lr'] = 0.001
        config['mu'] = 0.0
        config['num_layers'] = 2
        config['node_indim'] = 20
        config['dropout_p'] = 0.0
        config['dropout_mode'] = 0

    elif config_id==4:
        config['nb_iter'] = 3000
        config['lr'] = 0.001
        config['mu'] = 0.0
        config['num_layers'] = 5
        config['node_indim'] = -1
        config['dropout_p'] = 0.0
        config['dropout_mode'] = 0

    elif config_id==5:
        config['nb_iter'] = 1000
        config['lr'] = 0.001
        config['mu'] = 0.0
        config['num_layers'] = 7
        config['node_indim'] = -1
        config['dropout_p'] = 0.0
        config['dropout_mode'] = 0

    elif config_id==6:
        config['nb_iter'] = 1000
        config['lr'] = 0.001
        config['mu'] = 0.0
        config['num_layers'] = 12
        config['node_indim'] = -1
        config['dropout_p'] = 0.0
        config['dropout_mode'] = 0

    #Find the best config for BGCN on 1 fold and vary regu and dropout after
    #

    #Should test the convolve logit as well
    #and regularization

    else:
        raise NotImplementedError

    return config


def run_model(gcn_graph, config_params, gcn_graph_test):
    g_1 = tf.Graph()
    with g_1.as_default():

        node_dim = gcn_graph[0].X.shape[1]
        edge_dim = gcn_graph[0].E.shape[1] - 2.0
        nb_class = gcn_graph[0].Y.shape[1]

        gcn_model = gcn_models.GCNBaselineGraphList(node_dim, nb_class,
                                                 num_layers=config_params['num_layers'],
                                                 learning_rate=config_params['lr'],
                                                 mu=config_params['mu'],
                                                 node_indim=config_params['node_indim'],
                                                 dropout_rate=config_params['dropout_p'],
                                                 dropout_mode=config_params['dropout_mode']
                                                 )

        gcn_model.create_model()

        with tf.Session() as session:
            session.run([gcn_model.init])
            for i in range(config_params['nb_iter']):
                random.shuffle(gcn_graph)
                if i % 10 == 0:
                    print('Epoch', i)
                    mean_acc = []
                    for g in gcn_graph:
                        #                        print('G Stats #node,#edge',g.X.shape[0],g.E.shape[0])
                        acc = gcn_model.test(session, g.X.shape[0], g.X, g.Y, g.NA, verbose=False)
                        mean_acc.append(acc)
                    print('Mean Accuracy', np.mean(mean_acc))

                for g in gcn_graph:
                    gcn_model.train(session, g.X.shape[0], g.X, g.Y, g.NA, n_iter=1)

            mean_acc = []
            print('Training Error')
            for g in gcn_graph:
                print('G Stats #node,#edge', g.X.shape[0], g.E.shape[0])
                acc = gcn_model.test(session, g.X.shape[0], g.X, g.Y, g.NA)
                mean_acc.append(acc)
            print('Mean Accuracy', np.mean(mean_acc))

            print('Test Error')
            mean_acc_test = []
            for g in gcn_graph_test:
                print('G Stats #node,#edge', g.X.shape[0], g.E.shape[0])
                acc = gcn_model.test(session, g.X.shape[0], g.X, g.Y, g.NA)
                mean_acc_test.append(acc)
            print('Mean Accuracy', np.mean(mean_acc_test))
    return mean_acc_test


def main_fold(foldid,configid,outdir):
    pickle_train = '/opt/project/read/testJL/TABLE/abp_quantile_models/abp_CV_fold_' + str(
        foldid) + '_tlXlY_trn.pkl'
    pickle_test = '/opt/project/read/testJL/TABLE/abp_quantile_models/abp_CV_fold_' + str(foldid) + '_tlXlY_tst.pkl'

    train_graph = GCNDataset.load_transkribus_pickle(pickle_train)
    test_graph = GCNDataset.load_transkribus_pickle(pickle_test)

    config = get_config(configid)
    acc_test = run_model(train_graph, config, test_graph)
    print('Accuracy Test', acc_test)

    #To be improved ....the file result
    mean_acc=np.mean(acc_test)
    fresult_fname=os.path.join(outdir,'fold_'+str(foldid)+'_configid_'+str(configid)+'_acc_'+str(mean_acc))
    os.system( 'touch '+fresult_fname)


def main(_):

    if FLAGS.fold==-1:
        #Do it on all the fold for the specified configs
        #FOLD_IDS=[1,2,3,4]
        FOLD_IDS = [1]
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
        test_graph = GCNDataset.load_transkribus_pickle(pickle_test)

        config = get_config(FLAGS.configid)
        acc_test = run_model(train_graph, config, test_graph)
        print('Accuracy Test', acc_test)


if __name__ == '__main__':
    tf.app.run()