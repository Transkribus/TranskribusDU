# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

import pickle
import os.path
import random
import gcn.gcn_models as gcn_models
from gcn.gcn_datasets import GCNDataset

import time

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train', '', "FilePath Train pickle file")
tf.app.flags.DEFINE_string('test', '', "FilePath for the pickle")
tf.app.flags.DEFINE_integer('fold', '1', "FilePath for the pickle")
tf.app.flags.DEFINE_string('out_dir', 'out_res', "outdirectory for saving the results")
tf.app.flags.DEFINE_integer('configid', 0, 'gridid')
tf.app.flags.DEFINE_bool('snake',False, 'whether to work on the snake dataset')
tf.app.flags.DEFINE_bool('das_train',False, ' Training the Model for the DAS paper')
tf.app.flags.DEFINE_bool('das_predict',False, 'Prediction Experiment for the DAS paper')
tf.app.flags.DEFINE_bool('das_predict_workflow',False, 'Prediction Experiment for the DAS paper')
# Details of the training configuration.
tf.app.flags.DEFINE_float('learning_rate', 0.1, """How large a learning rate to use when training, default 0.1 .""")
tf.app.flags.DEFINE_integer('nb_iter', 3000, """How many training steps to run before ending, default 1.""")
tf.app.flags.DEFINE_integer('nb_layer', 1, """How many layers """)
tf.app.flags.DEFINE_integer('eval_iter', 256, """How often to evaluate the training results.""")
tf.app.flags.DEFINE_string('path_report', 'default', """Path for saving the results """)
tf.app.flags.DEFINE_string('grid_configs', '3_4_5', """Configs to be runned on all the folds """)
tf.app.flags.DEFINE_integer('qsub_taskid', -1, 'qsub_taskid')




#For Snake  python DU_gcn_task.py --snake=True --configid=22

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




def _make_grid_qsub(grid_qsub=0):

    if grid_qsub==0:
        tid=0
        C={}
        for fold_id in [1,2,3,4]:
            #for config in [4,5]:
            #for config in [27,28,29]:
            for config in [31]:
            #for config in [3, 4]:
            #for config in [5]:
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
        config['fast_convolve'] = True
        #config['train_Wn0']=False

    elif config_id==2:
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['stack_instead_add'] = True
        config['mu'] = 0.0
        config['num_layers'] = 1
        config['node_indim'] = -1
        config['nconv_edge'] = 10
        config['fast_convolve'] = True

    elif config_id==3:
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['stack_instead_add'] = True
        config['mu'] = 0.0
        config['num_layers'] = 1
        config['node_indim'] = -1
        config['nconv_edge'] = 50
        config['fast_convolve'] = True

    elif config_id==4:
        #config['nb_iter'] = 2000
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['stack_instead_add'] = True
        config['mu'] = 0.0
        config['num_layers'] = 2
        config['node_indim'] = -1
        config['nconv_edge'] = 7
        config['fast_convolve'] = True

    elif config_id==5:
        #config['nb_iter'] = 2000
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['stack_instead_add'] = True
        config['mu'] = 0.0
        config['num_layers'] = 3
        config['node_indim'] = -1 #INDIM =2 not working here
        config['nconv_edge'] = 10
        config['fast_convolve']=True
        #config['train_Wn0']=False
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
        #config['num_layers'] = 5 #Mean Node  Accuracy 0.9381
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

    elif config_id == 26: #Config for the Snake with the same feature rep as CRF  ie the fixed_node one
        config['nb_iter'] = 500
        config['lr'] = 0.001
        config['stack_instead_add'] = False #Default True
        config['mu'] = 0.0
        config['num_layers'] = 7
        config['node_indim'] = -1  # INDIM =2 not working here
        config['nconv_edge'] =10
        config['dropout_rate'] = 0.0
        config['dropout_mode'] = 0

    elif config_id==27:
        #This is config 5 but with stakcing
        # config['nb_iter'] = 2000
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['stack_instead_add'] = False
        config['mu'] = 0.0
        config['num_layers'] = 3
        config['node_indim'] = -1  # INDIM =2 not working here
        config['nconv_edge'] = 10


    elif config_id == 28:
        # This is config 5 but with stakcing
        # config['nb_iter'] = 2000
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['stack_instead_add'] = False
        config['mu'] = 0.0
        config['num_layers'] = 8
        config['node_indim'] = -1  # INDIM =2 not working here
        config['nconv_edge'] = 10

    elif config_id == 29:
        # This is config 5 but with stakcing
        # config['nb_iter'] = 2000
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['stack_instead_add'] = False
        config['mu'] = 0.0
        config['num_layers'] = 5
        config['node_indim'] = -1  # INDIM =2 not working here
        config['nconv_edge'] = 20

    elif config_id == 30:
        # Same as 28 but with fast convolve
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['stack_instead_add'] = False
        config['mu'] = 0.0
        config['num_layers'] = 8
        config['node_indim'] = -1  # INDIM =2 not working here
        config['nconv_edge'] = 10
        config['fast_convolve']=True

    elif config_id == 31:
        # Same as 28 but with fast convolve
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['stack_instead_add'] = False
        config['mu'] = 0.001
        config['num_layers'] = 8
        config['node_indim'] = -1  # INDIM =2 not working here
        config['nconv_edge'] = 10
        config['fast_convolve']=True

    elif config_id == 32:
        # Same as 31 but with dropout
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['stack_instead_add'] = False
        config['mu'] = 0.000
        config['num_layers'] = 8
        config['node_indim'] = -1  # INDIM =2 not working here
        config['nconv_edge'] = 10
        config['fast_convolve']=True
        config['dropout_rate'] = 0.2
        config['dropout_mode'] = 2
        # config['shared_We'] = True

    elif config_id == 33:
        # Same as 28 but with fast convolve
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['stack_instead_add'] = False
        config['mu'] = 0.001
        config['num_layers'] = 8
        config['node_indim'] = -1  # INDIM =2 not working here
        config['nconv_edge'] = 1
        config['fast_convolve']=True


    elif config_id == 34:
        # Same as 28 but with fast convolve
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['stack_instead_add'] = False
        config['mu'] = 0.001
        config['num_layers'] = 3
        config['node_indim'] = -1  # INDIM =2 not working here
        config['nconv_edge'] = 1
        config['fast_convolve'] = True

    elif config_id==35:
        #This is 5 with small regularization as 31
        #config['nb_iter'] = 2000
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['stack_instead_add'] = True
        config['mu'] = 0.001
        config['num_layers'] = 3
        config['node_indim'] = -1 #INDIM =2 not working here
        config['nconv_edge'] = 10
        config['fast_convolve']=True


    elif config_id == 36:
        # Same as 28 but with fast convolve
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['stack_instead_add'] = False
        config['mu'] = 0.001
        config['num_layers'] = 10
        config['node_indim'] = 10  # INDIM =2 not working here
        config['nconv_edge'] = 10
        config['fast_convolve']=True

    elif config_id == 37:
        # Same as 28 but with fast convolve
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['stack_instead_add'] = False
        config['mu'] = 0.000
        config['num_layers'] = 3
        config['node_indim'] = -1  # INDIM =2 not working here
        config['nconv_edge'] = 1
        config['fast_convolve'] = True

    elif config_id == 38:
        # Same as 28 but with fast convolve
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['stack_instead_add'] = False
        config['mu'] = 0.000
        config['num_layers'] = 5
        config['node_indim'] = -1  # INDIM =2 not working here
        config['nconv_edge'] = 3
        config['fast_convolve'] = True

    elif config_id == 39:
        # Same as 28 but with fast convolve
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['stack_instead_add'] = True
        config['mu'] = 0.000
        config['num_layers'] = 3
        config['node_indim'] = -1  # INDIM =2 not working here
        config['nconv_edge'] = 4
        config['fast_convolve'] = True

    elif config_id == 40:
        # Same as 28 but with fast convolve
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['stack_instead_add'] = True
        config['mu'] = 0.000
        config['num_layers'] = 3
        config['node_indim'] = -1  # INDIM =2 not working here
        config['nconv_edge'] = 10
        config['fast_convolve'] = True
        config['logit_convolve'] = True

    elif config_id==41:
        #Same as 5 but with sum stacking
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['stack_instead_add'] = False
        config['mu'] = 0.0
        config['num_layers'] = 3
        config['node_indim'] = -1 #INDIM =2 not working here
        config['nconv_edge'] = 10
        config['fast_convolve']=True

    elif config_id==42:
        #Baseline GCN model
        config['model']='baseline'
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['mu'] = 0.0
        config['num_layers'] = 3
        config['node_indim'] = -1 #INDIM =2 not working here

    elif config_id==43:
        config['model']='baseline'
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['mu'] = 0.0
        config['num_layers'] = 5
        config['node_indim'] = -1
        config['dropout_p'] = 0.0
        config['dropout_mode'] = 0

    elif config_id==44:
        
        config['model']='baseline'
        
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['mu'] = 0.0
        config['num_layers'] = 7
        config['node_indim'] = -1
        config['dropout_p'] = 0.0
        config['dropout_mode'] = 0

    elif config_id==45: 
        config['model']='baseline'
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['mu'] = 0.0
        config['num_layers'] = 12
        config['node_indim'] = -1
        config['dropout_p'] = 0.0
        config['dropout_mode'] = 0

    elif config_id==46:
        #same as 5 but with less iterations
        #in order to measure predictions time
        config['nb_iter'] = 200
        config['lr'] = 0.001
        config['stack_instead_add'] = True
        config['mu'] = 0.0
        config['num_layers'] = 3
        config['node_indim'] = -1 #INDIM =2 not working here
        config['nconv_edge'] = 10
        config['fast_convolve']=True

    elif config_id==47:
        config['nb_iter'] = 1000
        config['lr'] = 0.001
        config['stack_instead_add'] = True
        config['mu'] = 0.0
        config['num_layers'] = 3
        config['node_indim'] = -1  # INDIM =2 not working here
        config['nconv_edge'] = 10
        config['fast_convolve'] = True
        config['dropout_p'] = 0.3
        config['dropout_mode'] = 2

    elif config_id==48:
        #same as 5 but with less iterations
        #in order to measure predictions time
        config['nb_iter'] = 500
        config['lr'] = 0.00001
        config['stack_instead_add'] = True
        config['mu'] = 0.0
        config['num_layers'] = 3
        config['node_indim'] = -1 #INDIM =2 not working here
        config['nconv_edge'] = 10
        config['fast_convolve']=True
        #config['dropout_rate_edge']=0.2
        #config['dropout_rate_edge_feat'] = 0.0
        #config['dropout_rate_node'] = 0.2


    else:
        raise NotImplementedError

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
        
        
        if 'model' in config_params and config_params['model']=='baseline':
            gcn_model = gcn_models.GraphConvNet(node_dim,nb_class,
                                                 num_layers=config_params['num_layers'],
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
                                                     residual_connection=config_params['residual_connection'] if 'residual_connection' in config_params else False
                                                     )

            gcn_model.stack_instead_add = config_params['stack_instead_add']

            if 'fast_convolve' in config_params:
                gcn_model.fast_convolve = config_params['fast_convolve']

            if 'logit_convolve' in config_params:
                gcn_model.logit_convolve=config_params['logit_convolve']

            if 'train_Wn0' in config_params:
                gcn_model.train_Wn0= config_params['train_Wn0']

            if 'dropout_rate_edge' in config_params:
                gcn_model.dropout_rate_edge=config_params['dropout_rate_edge']
                print('Dropout Edge', gcn_model.dropout_rate_edge)

            if 'dropout_rate_edge_feat' in config_params:
                gcn_model.dropout_rate_edge_feat=config_params['dropout_rate_edge_feat']
                print('Dropout Edge', gcn_model.dropout_rate_edge_feat)

            if 'dropout_rate_node' in config_params:
                gcn_model.dropout_rate_node=config_params['dropout_rate_node']
                print('Dropout Node', gcn_model.dropout_rate_node)



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

            R=gcn_model.train_with_validation_set(session,gcn_graph_train,gcn_graph_val,config_params['nb_iter'],eval_iter=10,patience=1000,graph_test=gcn_graph_test,save_model_path=save_model_path)
            #R=gcn_model.train_All_lG(session,gcn_graph_train,gcn_graph_val,config_params['nb_iter'],eval_iter=10,patience=1000,graph_test=gcn_graph_test,save_model_path=save_model_path)
            f=open(outpicklefname,'wb')
            pickle.dump(R,f)
            f.close()


            #Ypred = gcn_model.predict_lG(session,gcn_graph_test)


        '''
        Y_true_flat=[]
        Ypred_flat=[]

        for graph,ypred in zip(gcn_graph_test,Ypred):
            ytrue = np.argmax(graph.Y,axis=1)
            Y_true_flat.extend(ytrue)
            Ypred_flat.extend(ypred)

        cm=sklearn.metrics.confusion_matrix(Y_true_flat,Ypred_flat)
        print(cm)
        out_conf_mat = outpicklefname+'.conf_mat.pkl'
        g=open(out_conf_mat,'wb')
        pickle.dump([Y_true_flat,Ypred_flat,cm],g)
        g.close()
        out_conf_mat_txt=outpicklefname+'.conf_mat.txt'
        f=open(out_conf_mat_txt,'w')
        f.write('Confusion Matrix \n')
        f.write(str(cm)+'\n')
        f.write(sklearn.metrics.classification_report(Y_true_flat,Ypred_flat))
        f.close()
        '''


def main_fold(foldid,configid,outdir):
    '''
    Simple Fold experiment, loading one fold, train and test
    :param foldid:
    :param configid:
    :param outdir:
    :return:
    '''
    pickle_train = '/nfs/project/read/testJL/TABLE/abp_quantile_models/abp_CV_fold_' + str(
        foldid) + '_tlXlY_trn.pkl'
    pickle_test = '/nfs/project/read/testJL/TABLE/abp_quantile_models/abp_CV_fold_' + str(foldid) + '_tlXlY_tst.pkl'

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

    elif FLAGS.das_train is True:
        #Load all the files of table
        # Train the model
        graph_train=[]

        debug=True
        if debug:

            pickle_train='/nfs/project/read/testJL/TABLE/das_abp_models/abp_full_tlXlY_trn.pkl'
            pickle_train_ra ='/nfs/project/read/testJL/TABLE/abp_DAS_CRF_Xr.pkl'   
            print(pickle_train_ra,pickle_train)
            #train_graph = GCNDataset.load_transkribus_pickle(pickle_train)
            graph_train =GCNDataset.load_transkribus_reverse_arcs_pickle(pickle_train,pickle_train_ra,format_reverse='lx')
        else:
            i=1
            pickle_train = '/nfs/project/read/testJL/TABLE/abp_quantile_models/abp_CV_fold_' + str(i) + '_tlXlY_trn.pkl'
            pickle_test = '/nfs/project/read/testJL/TABLE/abp_quantile_models/abp_CV_fold_' + str(i)  + '_tlXlY_tst.pkl'

            # reversed edged
            pickle_train_ra = '/nfs/project/read/testJL/TABLE/das_abp_models/abp_CV_fold_' + str(i) + '_tlXrlY_trn.pkl'
            pickle_test_ra = '/nfs/project/read/testJL/TABLE/das_abp_models/abp_CV_fold_' + str(i) + '_tlXrlY_tst.pkl'

            train_graph = GCNDataset.load_transkribus_reverse_arcs_pickle(pickle_train, pickle_train_ra)
            test_graph = GCNDataset.load_transkribus_reverse_arcs_pickle(pickle_test, pickle_test_ra)

            graph_train.extend(train_graph)
            graph_train.extend(test_graph)

        print('Graph Train Nb',len(graph_train))
        #Load the other dataset for predictions
        configid = FLAGS.configid
        config = get_config(configid)
        #config['nb_iter'] = 100

        dirp =os.path.join('models_all','C'+str(configid))
        mkdir_p(dirp)
        save_model_dir=os.path.join(dirp,'alldas_exp1_C'+str(configid)+'.ckpt')
        #I should  save the pickle
        outpicklefname=os.path.join(dirp,'alldas_exp1_C'+str(configid)+'.validation_scores.pickle')
        run_model_train_val_test(graph_train, config, outpicklefname, ratio_train_val=0.1,save_model_path=save_model_dir)
        #for test add gcn_graph_test=train_graph


    elif FLAGS.das_predict is True:

        do_test=False #some internal flags to do some testing

        node_dim = 29
        edge_dim = 140
        nb_class = 5

        configid = FLAGS.configid
        config = get_config(configid)


        #Get the best file
        #TODO Get the best file
        #node_dim = gcn_graph[0].X.shape[1]
        #edge_dim = gcn_graph[0].E.shape[1] - 2.0
        #nb_class = gcn_graph[0].Y.shape[1]

        #f = open('archive_models/das_exp1_C31.validation_scores.pickle', 'rb')

        val_pickle = os.path.join('models_all', 'C' + str(configid),
                                  "alldas_exp1_C" + str(configid) + '.validation_scores.pickle')
        print('Reading Training Info from:', val_pickle)
        f = open(val_pickle, 'rb')
        R = pickle.load(f)
        val = R['val_acc']
        print('Validation scores',val)

        epoch_index = np.argmax(val)
        print('Best performance on val set: Epoch',epoch_index)

        gcn_model = gcn_models.EdgeConvNet(node_dim, edge_dim, nb_class,
                                                 num_layers=config['num_layers'],
                                                 learning_rate=config['lr'],
                                                 mu=config['mu'],
                                                 node_indim=config['node_indim'],
                                                 nconv_edge=config['nconv_edge'],
                                                 )

        gcn_model.stack_instead_add = config['stack_instead_add']

        if 'fast_convolve' in config:
            gcn_model.fast_convolve = config['fast_convolve']

        gcn_model.create_model()


        if do_test:
            graph_train = []
            for i in range(1, 5):
                pickle_train = '/nfs/project/read/testJL/TABLE/abp_quantile_models/abp_CV_fold_' + str(i) + '_tlXlY_trn.pkl'
                print('loading ', pickle_train)
                train_graph = GCNDataset.load_transkribus_pickle(pickle_train)
                graph_train.extend(train_graph)

        #TODO load the data for test
        #/nfs/project/read/testJL/TABLE/abp_DAS_col9142_CRF_X.pkl
        if FLAGS.das_predict_workflow :
            pickle_predict='/nfs/project/read/testJL/TABLE/abp_DAS_col9142_workflow_X.pkl'
            pickle_predict_ra = '/nfs/project/read/testJL/TABLE/abp_DAS_col9142_workflow_Xr.pkl'
        else:
           pickle_predict='/nfs/project/read/testJL/TABLE/abp_DAS_col9142_CRF_X.pkl'
           pickle_predict_ra = '/nfs/project/read/testJL/TABLE/abp_DAS_col9142_CRF_Xr.pkl'
        
        print('loading ', pickle_predict,pickle_predict_ra)
        predict_graph = GCNDataset.load_test_pickle(pickle_predict,nb_class,pickle_reverse_arc=pickle_predict_ra)

        with tf.Session() as session:
            # Restore variables from disk.
            session.run(gcn_model.init)

            if do_test:
                gcn_model.restore_model(session, "models/das_exp1_C31.ckpt-99")
                print('Loaded models')

                graphAcc,node_acc=gcn_model.test_lG(session,graph_train)
                print(graphAcc,node_acc)

            model_path =os.path.join('models_all','C'+str(configid),"alldas_exp1_C"+str(configid)+".ckpt-"+str(10*epoch_index))
            print('Model_path',model_path)
            gcn_model.restore_model(session, model_path)
            print('Loaded models')


            start_time = time.time()
            lY_pred = gcn_model.predict_lG(session, predict_graph, verbose=False)
            end_time = time.time()
            print("--- %s seconds ---" % (end_time - start_time))
            print('Number of graphs:',len(lY_pred))

            #Convert to list as Python pickle does not  seem like the array while the list can be pickled
            lY_list=[]
            for x in lY_pred:
                lY_list.append(list(x))

            #print(lY_list)
            if FLAGS.das_predict_workflow:
                outpicklefname = 'allmodel_das_predict_C'+str(configid)+'_workflow.pickle'
            else:
                outpicklefname = 'allmodel_das_predict_C'+str(configid)+'.pickle'
            g=open(outpicklefname,'wb')
            #print(lY_pred)
            pickle.dump(lY_pred, g, protocol=2,fix_imports=True)
            g.close()






    elif FLAGS.qsub_taskid >-1:

        GRID = _make_grid_qsub(0)

        try:
            fold_id,configid =GRID[FLAGS.qsub_taskid]
        except:
            print('Invalid Grid Parameters',FLAGS.qsub_taskid,GRID)
            return -1
        print('Experiement with FOLD',fold_id,' CONFIG',configid)
        pickle_train = '/nfs/project/read/testJL/TABLE/abp_quantile_models/abp_CV_fold_' + str(
            fold_id) + '_tlXlY_trn.pkl'
        pickle_test = '/nfs/project/read/testJL/TABLE/abp_quantile_models/abp_CV_fold_' + str(
            fold_id) + '_tlXlY_tst.pkl'

        train_graph = GCNDataset.load_transkribus_pickle(pickle_train)
        test_graph = GCNDataset.load_transkribus_pickle(pickle_test)

        config = get_config(configid)

        if os.path.exists(FLAGS.out_dir) is False:
            print('Creating Dir',FLAGS.out_dir)
            os.mkdir(FLAGS.out_dir)

        outpicklefname = os.path.join(FLAGS.out_dir, 'table_F' + str(fold_id) + '_C' + str(configid) + '.pickle')
        run_model_train_val_test(train_graph, config, outpicklefname,ratio_train_val=0.1,gcn_graph_test= test_graph)


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


            pickle_train = '/nfs/project/read/testJL/TABLE/abp_quantile_models/abp_CV_fold_' + str(
                FLAGS.fold) + '_tlXlY_trn.pkl'
            pickle_test = '/nfs/project/read/testJL/TABLE/abp_quantile_models/abp_CV_fold_' + str(FLAGS.fold) + '_tlXlY_tst.pkl'

            #reversed edged
            pickle_train_ra = '/nfs/project/read/testJL/TABLE/das_abp_models/abp_CV_fold_' + str(
                FLAGS.fold) + '_tlXrlY_trn.pkl'
            pickle_test_ra = '/nfs/project/read/testJL/TABLE/das_abp_models/abp_CV_fold_' + str(FLAGS.fold) + '_tlXrlY_tst.pkl'

            #train_graph = GCNDataset.load_transkribus_pickle(pickle_train)
            train_graph = GCNDataset.load_transkribus_reverse_arcs_pickle(pickle_train,pickle_train_ra)
            print('Loaded Trained Graphs:',len(train_graph))
            test_graph = GCNDataset.load_transkribus_reverse_arcs_pickle(pickle_test,pickle_test_ra)
            #test_graph = GCNDataset.load_transkribus_pickle(pickle_test, pickle_test_ra)
            print('Loaded Test Graphs:', len(test_graph))

            config = get_config(FLAGS.configid)

            #acc_test = run_model(train_graph, config, test_graph,eval_iter=1)
            #print('Accuracy Test', acc_test)

            outpicklefname=os.path.join(FLAGS.out_dir,'table_F'+str(FLAGS.fold)+'_C'+str(FLAGS.configid)+'.pickle')
            run_model_train_val_test(train_graph,config,outpicklefname,gcn_graph_test= test_graph)



if __name__ == '__main__':
    tf.app.run()
