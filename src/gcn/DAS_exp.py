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

import sklearn.metrics
import time

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dpath', 'data', "directory where data is supposed to be found")
tf.app.flags.DEFINE_integer('fold', '1', "FilePath for the pickle")
tf.app.flags.DEFINE_string('out_dir', 'out_res', "outdirectory for saving the results")
tf.app.flags.DEFINE_integer('configid', 0, 'Parameters')
# Details of the training configuration.

import errno
import os


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise



def get_config(config_id=0):
    config = {}

    if config_id == 0:
        config['model'] ='logit'
        config['nb_iter'] = 500
        config['lr'] = 0.001
        config['mu'] = 0.1
        config['num_layers'] = 1
        config['node_indim'] = -1

    elif config_id ==1:
        config['model'] = 'logit'
        config['nb_iter'] = 500
        config['lr'] = 0.001
        config['stack_instead_add'] = False
        config['mu'] = 0.1
        config['num_layers'] = 1
        config['node_indim'] = -1
        config['nconv_edge'] = 1
        config['fast_convolve'] = True
        config['train_Wn0']=False

    else:
        raise ValueError('Invalid Config ID')

    return config


def run_model_train_val_test(gcn_graph,
                             config_params,
                             outpicklefname,
                             ratio_train_val=0.1,
                             gcn_graph_test=None,
                             save_model_path=None
                             ):
    g_1 = tf.Graph()

    with g_1.as_default():

        node_dim = gcn_graph[0].X.shape[1]
        edge_dim = gcn_graph[0].E.shape[1] - 2.0
        nb_class = gcn_graph[0].Y.shape[1]

        if 'model' in config_params and config_params['model'] == 'baseline':
            gcn_model = gcn_models.GraphConvNet(node_dim, nb_class,
                                                num_layers=config_params['num_layers'],
                                                learning_rate=config_params['lr'],
                                                mu=config_params['mu'],
                                                node_indim=config_params['node_indim'],
                                                )
        elif 'model' in config_params and config_params['model'] == 'logit':
            gcn_model = gcn_models.GraphConvNet(node_dim, nb_class,
                                                learning_rate=config_params['lr'],
                                                mu=config_params['mu'],
                                                node_indim=config_params['node_indim'],
                                                )


        else:
            gcn_model = gcn_models.EdgeConvNet(node_dim, edge_dim, nb_class,
                                               num_layers=config_params['num_layers'],
                                               learning_rate=config_params['lr'],
                                               mu=config_params['mu'],
                                               node_indim=config_params['node_indim'],
                                               nconv_edge=config_params['nconv_edge'],
                                               residual_connection=config_params[
                                                   'residual_connection'] if 'residual_connection' in config_params else False
                                               )

            gcn_model.stack_instead_add = config_params['stack_instead_add']

            if 'fast_convolve' in config_params:
                gcn_model.fast_convolve = config_params['fast_convolve']

            if 'logit_convolve' in config_params:
                gcn_model.logit_convolve = config_params['logit_convolve']

            if 'train_Wn0' in config_params:
                gcn_model.train_Wn0 = config_params['train_Wn0']

        gcn_model.create_model()

        # Split Training to get some validation
        # ratio_train_val=0.2
        split_idx = int(ratio_train_val * len(gcn_graph))
        random.shuffle(gcn_graph)
        gcn_graph_train = []
        gcn_graph_val = []

        gcn_graph_val.extend(gcn_graph[:split_idx])
        gcn_graph_train.extend(gcn_graph[split_idx:])

        with tf.Session() as session:
            session.run([gcn_model.init])

            R = gcn_model.train_with_validation_set(session, gcn_graph_train, gcn_graph_val, config_params['nb_iter'],
                                                    eval_iter=10, patience=1000, graph_test=gcn_graph_test,
                                                    save_model_path=save_model_path)

            f = open(outpicklefname, 'wb')
            pickle.dump(R, f)
            f.close()



def main(_):
        config = get_config(FLAGS.configid)
        print(config)

        #Pickle for Logit are sufficient
        pickle_train = os.path.join(FLAGS.dpath,'abp_CV_fold_' + str(FLAGS.fold) + '_tlXlY_trn.pkl')
        pickle_test = os.path.join(FLAGS.dpath,'abp_CV_fold_' + str(FLAGS.fold) + '_tlXlY_tst.pkl')

        # Baseline Cases
        if 'model' in config:
            train_graph = GCNDataset.load_transkribus_pickle(pickle_train)
            test_graph = GCNDataset.load_transkribus_pickle(pickle_test)
            print('Loaded Test Graphs:',len(test_graph))


        else:
            pickle_train_ra = os.path.join(FLAGS.dpath, 'abp_CV_fold_' + str(FLAGS.fold) + '_tlXrlY_trn.pkl')
            pickle_test_ra = os.path.join(FLAGS.dpath, 'abp_CV_fold_' + str(FLAGS.fold) + '_tlXrlY_tst.pkl')
            train_graph = GCNDataset.load_transkribus_reverse_arcs_pickle(pickle_train, pickle_train_ra)
            print('Loaded Trained Graphs:', len(train_graph))
            test_graph = GCNDataset.load_transkribus_reverse_arcs_pickle(pickle_test, pickle_test_ra)
            print('Loaded Test Graphs:', len(test_graph))

            # print('Accuracy Test', acc_test)

        outpicklefname = os.path.join(FLAGS.out_dir,
                                      'table_F' + str(FLAGS.fold) + '_C' + str(FLAGS.configid) + '.pickle')
        run_model_train_val_test(train_graph, config, outpicklefname, gcn_graph_test=test_graph)


if __name__ == '__main__':
    tf.app.run()
