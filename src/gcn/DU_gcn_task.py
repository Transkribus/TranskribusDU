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
tf.app.flags.DEFINE_string('fold', '', "FilePath for the pickle")
tf.app.flags.DEFINE_string('out_dir', '', "outdirectory for saving the results")
tf.app.flags.DEFINE_integer('gridid', -1, 'gridid')

# Details of the training configuration.
tf.app.flags.DEFINE_float('learning_rate', 0.1, """How large a learning rate to use when training, default 0.1 .""")
tf.app.flags.DEFINE_integer('nb_iter', 3000, """How many training steps to run before ending, default 1.""")
tf.app.flags.DEFINE_integer('nb_layer', 1, """How many layers """)
tf.app.flags.DEFINE_bool('stack_PN', False, "whehter to concator add the vector induced from the neighbors")
tf.app.flags.DEFINE_integer('eval_iter', 256, """How often to evaluate the training results.""")
tf.app.flags.DEFINE_string('path_report', 'default', """Path for saving the results """)

pickle_fname = '/opt/project/read/testJL/TABLE/abp_quantile_models/abp_CV_fold_1_tlXlY_trn.pkl'


def get_config(config_id=0):
    config = {}

    if config_id == 0:
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['stack_instead_add'] = True
        config['mu'] = 0.0
        config['num_layers'] = 1
        config['node_indim'] = -1
    else:
        raise NotImplementedError

    return config


def run_model(gcn_graph, config_params, gcn_graph_test):
    g_1 = tf.Graph()
    with g_1.as_default():

        node_dim = gcn_graph[0].X.shape[1]
        edge_dim = gcn_graph[0].E.shape[1] - 2.0
        nb_class = gcn_graph[0].Y.shape[1]

        gcn_model = gcn_models.GCNModelGraphList(node_dim, edge_dim, nb_class,
                                                 num_layers=config_params['num_layers'],
                                                 learning_rate=config_params['lr'],
                                                 mu=config_params['mu'],
                                                 node_indim=config_params['node_indim'])

        gcn_model.stack_instead_add = config_params['stack_instead_add']
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
                        acc = gcn_model.test(session, g.X.shape[0], g.X, g.EA, g.Y, g.NA, verbose=False)
                        mean_acc.append(acc)
                    print('Mean Accuracy', np.mean(mean_acc))
                else:
                    for g in gcn_graph:
                        gcn_model.train(session, g.X.shape[0], g.X, g.EA, g.Y, g.NA, n_iter=1)

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


def main(_):
    pickle_train = '/opt/project/read/testJL/TABLE/abp_quantile_models/abp_CV_fold_' + str(
        FLAGS.fold) + '_tlXlY_trn.pkl'
    pickle_test = '/opt/project/read/testJL/TABLE/abp_quantile_models/abp_CV_fold_' + str(FLAGS.fold) + '_tlXlY_tst.pkl'

    train_graph = GCNDataset.load_transkribus_pickle(pickle_train)
    test_graph = GCNDataset.load_transkribus_pickle(pickle_test)

    config = get_config(0)
    acc_test = run_model(train_graph, config, test_graph)
    print('Accuracy Test', acc_test)


if __name__ == '__main__':
    tf.app.run()
