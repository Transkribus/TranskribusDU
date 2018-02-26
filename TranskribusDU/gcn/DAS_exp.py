# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import pickle
import os.path
import random
import gcn.gcn_models as gcn_models
from gcn.gcn_datasets import GCNDataset


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dpath', 'data', "directory where data is supposed to be found")
tf.app.flags.DEFINE_integer('fold', '1', "FilePath for the pickle")
tf.app.flags.DEFINE_string('out_dir', 'out_das', "outdirectory for saving the results")
tf.app.flags.DEFINE_integer('configid', 0, 'Parameters')
tf.app.flags.DEFINE_bool('das_predict_workflow', False, 'Prediction Experiment for the DAS paper')
tf.app.flags.DEFINE_string('outname', 'default', "default name for saving the results")
# Details of the training configuration.

import errno


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
        config['model'] = 'logit'
        config['nb_iter'] = 1000
        config['lr'] = 0.001
        config['mu'] = 0.1
        config['num_layers'] = 1
        config['node_indim'] = -1

    elif config_id == 1:
        config['name'] = 'Logit-1Conv'
        config['nb_iter'] = 1000
        config['lr'] = 0.001
        config['stack_instead_add'] = False
        config['mu'] = 0.1
        config['num_layers'] = 1
        config['node_indim'] = -1
        config['nconv_edge'] = 1
        config['train_Wn0'] = False

    elif config_id == 5:
        config['name'] = '3Layers-10conv-stack'
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['stack_instead_add'] = True
        config['mu'] = 0.0
        config['num_layers'] = 3
        config['node_indim'] = -1  # INDIM =2 not working here
        config['nconv_edge'] = 10
        config['fast_convolve'] = True
        # Not Sure the test part is correct ...
        config['dropout_rate_edge'] = 0.0
        config['dropout_rate_edge_feat'] = 0.0
        config['dropout_rate_node'] = 0.0
        # config['conv_weighted_avg']=True

    elif config_id == 7:
        config['name'] = '3Layers-10conv-stack'
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['stack_instead_add'] = True
        config['mu'] = 0.0
        config['num_layers'] = 3
        config['node_indim'] = -1  # INDIM =2 not working here
        config['nconv_edge'] = 10
        config['fast_convolve'] = True
        # Not Sure the test part is correct ...
        config['dropout_rate_edge'] = 0.2
        config['dropout_rate_edge_feat'] = 0.2
        config['dropout_rate_node'] = 0.2
        config['activation'] = tf.nn.tanh

    elif config_id == 8:
        config['name'] = '3Layers-10conv-stack'
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['stack_instead_add'] = True
        config['mu'] = 0.0
        config['num_layers'] = 3
        config['node_indim'] = -1  # INDIM =2 not working here
        config['nconv_edge'] = 10
        config['fast_convolve'] = True
        # Not Sure the test part is correct ...
        config['dropout_rate_edge'] = 0.0
        config['dropout_rate_edge_feat'] = 0.2
        config['dropout_rate_node'] = 0.0
        config['activation'] = tf.nn.tanh

    elif config_id == 9:
        config['name'] = '3Layers-10conv-stack'
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['stack_instead_add'] = True
        config['mu'] = 0.0
        config['num_layers'] = 3
        config['node_indim'] = -1  # INDIM =2 not working here
        config['nconv_edge'] = 10
        config['fast_convolve'] = True
        # Not Sure the test part is correct ...
        config['dropout_rate_edge'] = 0.0
        config['dropout_rate_edge_feat'] = 0.0
        config['dropout_rate_node'] = 0.2
        config['activation'] = tf.nn.tanh

    elif config_id == 10:
        config['name'] = '3Layers-10conv-stack'
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['stack_instead_add'] = True
        config['mu'] = 0.0
        config['num_layers'] = 3
        config['node_indim'] = -1  # INDIM =2 not working here
        config['nconv_edge'] = 10
        config['fast_convolve'] = True
        # Not Sure the test part is correct ...
        config['dropout_rate_edge'] = 0.2
        config['dropout_rate_edge_feat'] = 0.0
        config['dropout_rate_node'] = 0.0
        config['activation'] = tf.nn.tanh


    elif config_id == 28:
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['stack_instead_add'] = False
        config['mu'] = 0.0
        config['num_layers'] = 8
        config['node_indim'] = -1  # INDIM =2 not working here
        config['nconv_edge'] = 10
        config['fast_convolve'] = True
        # Not Sure the test part is correct ...
        config['dropout_rate_edge'] = 0.125
        config['dropout_rate_edge_feat'] = 0.0
        config['dropout_rate_node'] = 0.1


    elif config_id == 33:
        config['name'] = '8Layers-1conv'
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['stack_instead_add'] = False
        config['mu'] = 0.001
        config['num_layers'] = 8
        config['node_indim'] = -1  # INDIM =2 not working here
        config['nconv_edge'] = 1
        config['fast_convolve'] = True

    elif config_id == 44:
        config['model'] = 'gcn'
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['mu'] = 0.0
        config['num_layers'] = 7
        config['node_indim'] = -1
        config['dropout_p'] = 0.0
        config['dropout_mode'] = 0


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

        if 'model' in config_params and config_params['model'] == 'gcn':
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

            if 'activation' in config_params:
                gcn_model.activation = config_params['activation']

            if 'fast_convolve' in config_params:
                gcn_model.fast_convolve = config_params['fast_convolve']

            if 'logit_convolve' in config_params:
                gcn_model.logit_convolve = config_params['logit_convolve']

            if 'train_Wn0' in config_params:
                gcn_model.train_Wn0 = config_params['train_Wn0']

            if 'dropout_rate_edge' in config_params:
                gcn_model.dropout_rate_edge = config_params['dropout_rate_edge']
                print('Dropout Edge', gcn_model.dropout_rate_edge)

            if 'dropout_rate_edge_feat' in config_params:
                gcn_model.dropout_rate_edge_feat = config_params['dropout_rate_edge_feat']
                print('Dropout Edge', gcn_model.dropout_rate_edge_feat)

            if 'dropout_rate_node' in config_params:
                gcn_model.dropout_rate_node = config_params['dropout_rate_node']
                print('Dropout Node', gcn_model.dropout_rate_node)

            if 'conv_weighted_avg' in config_params:
                gcn_model.use_conv_weighted_avg = config_params['conv_weighted_avg']

            gcn_model.use_edge_mlp = False

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

    mkdir_p(FLAGS.out_dir)

    # Pickle for Logit are sufficient
    pickle_train = os.path.join(FLAGS.dpath, 'abp_CV_fold_' + str(FLAGS.fold) + '_tlXlY_trn.pkl')
    pickle_test = os.path.join(FLAGS.dpath, 'abp_CV_fold_' + str(FLAGS.fold) + '_tlXlY_tst.pkl')

    # Baseline Models do not need reverse arc features
    if 'model' in config:
        train_graph = GCNDataset.load_transkribus_pickle(pickle_train)
        test_graph = GCNDataset.load_transkribus_pickle(pickle_test)
        print('Loaded Test Graphs:', len(test_graph))

        if FLAGS.outname == 'default':
            outpicklefname = os.path.join(FLAGS.out_dir,
                                          'table_F' + str(FLAGS.fold) + '_C' + str(FLAGS.configid) + '.pickle')
        else:
            outpicklefname = os.path.join(FLAGS.out_dir, FLAGS.outname)


    else:

        if FLAGS.das_predict_workflow is True:
            print('Doing Experiment on Predict Workflow ....')
            pickle_train = '/nfs/project/read/testJL/TABLE/das_abp_models/abp_full_tlXlY_trn.pkl'
            pickle_train_ra = '/nfs/project/read/testJL/TABLE/abp_DAS_CRF_Xr.pkl'
            print(pickle_train_ra, pickle_train)
            # train_graph = GCNDataset.load_transkribus_pickle(pickle_train)
            train_graph = GCNDataset.load_transkribus_reverse_arcs_pickle(pickle_train, pickle_train_ra,
                                                                          format_reverse='lx')

            fX_col9142 = '../../usecases/ABP/resources/DAS_2018/abp_DAS_col9142_CRF_X.pkl'
            fXr_col9142 = '../../usecases/ABP/resources/DAS_2018/abp_DAS_col9142_CRF_Xr.pkl'
            fY_col9142 = '../../usecases/ABP/resources/DAS_2018/DAS_col9142_l_Y_GT.pkl'

            test_graph = GCNDataset.load_transkribus_list_X_Xr_Y(fX_col9142, fXr_col9142, fY_col9142)

            if FLAGS.outname == 'default':
                outpicklefname = os.path.join(FLAGS.out_dir,
                                              'col9142_C' + str(FLAGS.configid) + '.pickle')
            else:
                outpicklefname = os.path.join(FLAGS.out_dir, FLAGS.outname)

        else:
            pickle_train_ra = os.path.join(FLAGS.dpath, 'abp_CV_fold_' + str(FLAGS.fold) + '_tlXrlY_trn.pkl')
            pickle_test_ra = os.path.join(FLAGS.dpath, 'abp_CV_fold_' + str(FLAGS.fold) + '_tlXrlY_tst.pkl')
            train_graph = GCNDataset.load_transkribus_reverse_arcs_pickle(pickle_train, pickle_train_ra,
                                                                          attach_edge_label=True)
            test_graph = GCNDataset.load_transkribus_reverse_arcs_pickle(pickle_test, pickle_test_ra)

            if FLAGS.outname == 'default':
                outpicklefname = os.path.join(FLAGS.out_dir,
                                              'table_F' + str(FLAGS.fold) + '_C' + str(FLAGS.configid) + '.pickle')
            else:
                outpicklefname = os.path.join(FLAGS.out_dir, FLAGS.outname)

        print('Loaded Trained Graphs:', len(train_graph))
        print('Loaded Test Graphs:', len(test_graph))

    run_model_train_val_test(train_graph, config, outpicklefname, gcn_graph_test=test_graph)


if __name__ == '__main__':
    tf.app.run()
