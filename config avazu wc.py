import os
import argparse
conf_parser = argparse.ArgumentParser(description='args')
conf_parser.add_argument('--model', type=str,default="our",required = False)
conf_parser.add_argument('--dataset', type=str,default="avazu",required = False)
conf_parser.add_argument('--start', type=int,default=0,required = False)
conf_parser.add_argument('--end', type=int,default=1,required = False)
conf_parser.add_argument('--ablation_type', type=int,default=0,required = False)

conf_parser.add_argument('--learning_rate', default=0.001, type=float)
conf_parser.add_argument('--decay_rate', default=0.7, type=float)
conf_parser.add_argument('--regularizer',default='l2',type=str)#l1,l2,None
conf_parser.add_argument('--regularizer_weight_decay',default=0.0001,type=float)
conf_parser.add_argument('--trainable',default=True,type=bool)
conf_parser.add_argument('--drop_keep_prob', default=1.0, type=float)
conf_parser.add_argument('--epochs', default=10, type=int)
conf_parser.add_argument('--batch_size', default=128, type=int)
conf_parser.add_argument('--loss_weight_alpha', type=float,default=0.3,required = False)
conf_parser.add_argument('--USER_EMBEDDING_SIZE', type=int,default=32,required = False)
conf_parser.add_argument('--ITEM_EMBEDDING_SIZE', type=int,default=64,required = False)
conf_parser.add_argument('--OTHER_EMBEDDING_SIZE', type=int,default=32,required = False)
conf_parser.add_argument('--LSTM_1', type=int,default=64,required = False)
conf_args=conf_parser.parse_args()
MAX_LEN=20
BATCH_SIZE=conf_args.batch_size
MAX_EPOCHES=conf_args.epochs
SEED=1234
MODEL_NAME=conf_args.model


DATA_DIR='data/'
USE_DATA=conf_args.dataset
USE_DATA_DIR=DATA_DIR+USE_DATA+'/'

DATA_RESULTS_DIR='data_res/'

MODEL_SAVE_DIR='experiments/'+MODEL_NAME+'/'
if not os.path.exists(MODEL_SAVE_DIR):
    os.mkdir(MODEL_SAVE_DIR)
MODEL_SAVE_DIR=MODEL_SAVE_DIR+USE_DATA+'/'
if not os.path.exists(MODEL_SAVE_DIR):
    os.mkdir(MODEL_SAVE_DIR)
MODEL_LOG_DIR=MODEL_SAVE_DIR+'logs/'
if not os.path.exists(MODEL_LOG_DIR):
    os.mkdir(MODEL_LOG_DIR)
MODEL_LOG_NAME=MODEL_NAME+'_'+USE_DATA+'.log'
