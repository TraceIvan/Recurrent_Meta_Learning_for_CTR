from config import *
from models import ourModel
from models.ourModel import our_train_local_global,our_eval_global,our_eval_global_finetune,our_train_global
from myLog import MyLog
from sklearn.metrics import roc_auc_score
import argparse
import numpy
import random
import pickle
import tensorflow as tf
import os
import numpy as np
from our_dataloader import our_Data_loader
import csv


def init_seed():
    numpy.random.seed(SEED)
    random.seed(SEED)
    tf.set_random_seed(SEED)

def load_model(sess,curlog,args,cur_status,re_use=None):
    model=None
    if MODEL_NAME=="our":
        model=ourModel.ourModel(curlog,cur_status,args)
    else:
        raise Exception('wrong model name.')

    if re_use:
        cur_status_model_save_dir = MODEL_SAVE_DIR + cur_status + '/'
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(cur_status_model_save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            curlog.info("**********************************************************************************")
            curlog.info('Restore model from {} successfully!'.format(ckpt.model_checkpoint_path))
            curlog.info("**********************************************************************************")
        else:
            raise Exception('no existing models to restore.')
    else:
        sess.run(tf.global_variables_initializer())

    return model

def eval_test(cur_status,sess,model,curlog):
    user_dict = pickle.load(open(USE_DATA_DIR + 'user_dict.pkl', 'rb'))
    item_dict = pickle.load(open(USE_DATA_DIR + 'item_dict.pkl', 'rb'))
    test_set = pickle.load(open(USE_DATA_DIR + '%s_slide_test.pkl' % cur_status, 'rb'))
    random.shuffle(test_set)
    if not os.path.exists(DATA_RESULTS_DIR):
        os.mkdir(DATA_RESULTS_DIR)
    cur_batch = 0
    sum_loss = 0.0
    tot_realLabel = []
    tot_preLabel = []
    for _, epoch_data in our_Data_loader(USE_DATA,test_set,user_dict,item_dict, BATCH_SIZE):
        cur_label, cur_predicted, loss=our_eval_global(sess,model,epoch_data)
        realLabel = cur_label.tolist()
        predict_pro = cur_predicted.tolist()
        sum_loss += loss
        tot_realLabel.extend(realLabel)
        tot_preLabel.extend(predict_pro)
        cur_batch += 1
    tot_AUC = roc_auc_score(tot_realLabel, tot_preLabel)
    curlog.info("{} final test: loss:{}; AUC:{}".format(cur_status,sum_loss / cur_batch, tot_AUC))
    return tot_AUC

def eval_test_finetune(cur_status,sess,model,curlog):
    user_dict = pickle.load(open(USE_DATA_DIR + 'user_dict.pkl', 'rb'))
    item_dict = pickle.load(open(USE_DATA_DIR + 'item_dict.pkl', 'rb'))
    test_set = pickle.load(open(USE_DATA_DIR + '%s_slide_test.pkl' % cur_status, 'rb'))
    finetune_a = pickle.load(open(USE_DATA_DIR + '%s_slide_finetune_a.pkl' % cur_status, 'rb'))
    finetune_b = pickle.load(open(USE_DATA_DIR + '%s_slide_finetune_b.pkl' % cur_status, 'rb'))
    finetune_c = pickle.load(open(USE_DATA_DIR + '%s_slide_finetune_c.pkl' % cur_status, 'rb'))

    random.shuffle(test_set)
    if not os.path.exists(DATA_RESULTS_DIR):
        os.mkdir(DATA_RESULTS_DIR)

    for _, epoch_data in our_Data_loader(USE_DATA, finetune_a, user_dict, item_dict, BATCH_SIZE):
        if len(epoch_data[0]) < BATCH_SIZE:
            continue
        cur_label, cur_predicted, loss = our_eval_global_finetune(sess, model, epoch_data)

    cur_batch = 0
    sum_loss = 0.0
    tot_realLabel = []
    tot_preLabel = []
    for _, epoch_data in our_Data_loader(USE_DATA, test_set, user_dict, item_dict, BATCH_SIZE):
        if len(epoch_data[0]) < BATCH_SIZE:
            continue
        cur_label, cur_predicted, loss = our_eval_global(sess, model, epoch_data)
        realLabel = cur_label.tolist()
        predict_pro = cur_predicted.tolist()
        sum_loss += loss
        tot_realLabel.extend(realLabel)
        tot_preLabel.extend(predict_pro)
        cur_batch += 1
    tot_AUC = roc_auc_score(tot_realLabel, tot_preLabel)
    curlog.info("{} after finetune a test: loss:{}; AUC:{}".format(cur_status, sum_loss / cur_batch, tot_AUC))

    for _, epoch_data in our_Data_loader(USE_DATA, finetune_b, user_dict, item_dict, BATCH_SIZE):
        if len(epoch_data[0]) < BATCH_SIZE:
            continue
        cur_label, cur_predicted, loss = our_eval_global_finetune(sess, model, epoch_data)

    cur_batch = 0
    sum_loss = 0.0
    tot_realLabel = []
    tot_preLabel = []
    for _, epoch_data in our_Data_loader(USE_DATA, test_set, user_dict, item_dict, BATCH_SIZE):
        if len(epoch_data[0]) < BATCH_SIZE:
            continue
        cur_label, cur_predicted, loss = our_eval_global(sess, model, epoch_data)
        realLabel = cur_label.tolist()
        predict_pro = cur_predicted.tolist()
        sum_loss += loss
        tot_realLabel.extend(realLabel)
        tot_preLabel.extend(predict_pro)
        cur_batch += 1
    tot_AUC = roc_auc_score(tot_realLabel, tot_preLabel)
    curlog.info("{} after finetune b test: loss:{}; AUC:{}".format(cur_status, sum_loss / cur_batch, tot_AUC))

    for _, epoch_data in our_Data_loader(USE_DATA, finetune_c, user_dict, item_dict, BATCH_SIZE):
        if len(epoch_data[0]) < BATCH_SIZE:
            continue
        cur_label, cur_predicted, loss = our_eval_global_finetune(sess, model, epoch_data)

    cur_batch = 0
    sum_loss = 0.0
    tot_realLabel = []
    tot_preLabel = []
    for _, epoch_data in our_Data_loader(USE_DATA,test_set,user_dict,item_dict, BATCH_SIZE):
        if len(epoch_data[0]) < BATCH_SIZE:
            continue
        cur_label, cur_predicted, loss=our_eval_global(sess,model,epoch_data)
        realLabel = cur_label.tolist()
        predict_pro = cur_predicted.tolist()
        sum_loss += loss
        tot_realLabel.extend(realLabel)
        tot_preLabel.extend(predict_pro)
        cur_batch += 1
    tot_AUC = roc_auc_score(tot_realLabel, tot_preLabel)
    curlog.info("{} after finetune c test: loss:{}; AUC:{}".format(cur_status,sum_loss / cur_batch, tot_AUC))
    return tot_AUC

def train():
    init_seed()
    status = ['wc', 'cw']
    user_dict=pickle.load(open(USE_DATA_DIR+'user_dict.pkl', 'rb'))
    item_dict=pickle.load(open(USE_DATA_DIR+'item_dict.pkl', 'rb'))
    for cur_status in status[conf_args.start:conf_args.end]:
        cur_status_model_save_dir=MODEL_SAVE_DIR+cur_status+'/'
        if not os.path.exists(MODEL_SAVE_DIR):
            os.mkdir(MODEL_SAVE_DIR)
        if not os.path.exists(cur_status_model_save_dir):
            os.mkdir(cur_status_model_save_dir)
        
        train_set = pickle.load(open(USE_DATA_DIR + '%s_slide_train.pkl' % cur_status, 'rb'))
        random.shuffle(train_set)
        train_samples=len(train_set)
        batches = train_samples // BATCH_SIZE
        if train_samples % BATCH_SIZE:
            batches += 1
        if not os.path.exists(MODEL_SAVE_DIR):
            os.mkdir(MODEL_SAVE_DIR)
        if not os.path.exists(MODEL_LOG_DIR):
            os.mkdir(MODEL_LOG_DIR)

        curlog=MyLog(MODEL_LOG_DIR+MODEL_LOG_NAME)

        parser=conf_parser
        try:
            parser.add_argument('--batches', default=batches, type=int)
        except Exception as e:
            parser.set_defaults(batches=batches)
        args = parser.parse_args()

        if args.ablation_type:
            cur_status_model_save_dir=cur_status_model_save_dir+"ablation_"+str(args.ablation_type)+'/'
            if not os.path.exists(cur_status_model_save_dir):
                os.mkdir(cur_status_model_save_dir)

        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        tf.reset_default_graph()
        with tf.Session(config=gpu_config) as sess:
            model=load_model(sess,curlog,args,cur_status)

            best_loss = np.inf
            best_epoch_loss = np.inf
            sum_loss = 0
            train_step = 0
            for cur_epoch in range(MAX_EPOCHES):
                epoch_loss = 0.0
                for _, epoch_data in our_Data_loader(USE_DATA,train_set,user_dict,item_dict,BATCH_SIZE):
                    loss, step, lr=our_train_local_global(sess,model,epoch_data)
                    train_step = step
                    sum_loss += loss
                    epoch_loss += loss
                    if loss < best_loss:
                        best_loss = loss
                        curlog.info('Epoch-{}\tstep-{}\tlr:{:.6f}\tloss: {:.6f}'.format(cur_epoch + 1, train_step, lr,
                                                                                       loss))
                    if train_step % 50 == 0:
                        curlog.info('Epoch-{}/{}\tstep-{}/{}\tlr:{:.6f}\tloss: {:.6f}'.format(cur_epoch + 1, MAX_EPOCHES,
                                                                                             train_step, batches, lr,
                                                                                             sum_loss / 50))
                        sum_loss = 0
                curlog.info('Epoch-{}/{}\tloss: {:.6f}'.format(cur_epoch + 1, MAX_EPOCHES, epoch_loss / batches))
                eval_test(cur_status, sess, model, curlog)
                if epoch_loss / batches < best_epoch_loss:
                    best_epoch_loss = epoch_loss / batches
                    ckpt_path = cur_status_model_save_dir + MODEL_NAME+'-' + str(cur_epoch + 1) + '-.ckpt'
                    model.saver.save(sess, ckpt_path, global_step=train_step)
                    curlog.info("model saved to {}".format(ckpt_path))

            ckpt_path = cur_status_model_save_dir + MODEL_NAME+'-last.ckpt'
            model.saver.save(sess, ckpt_path, global_step=train_step)
            curlog.info("model saved to {}".format(ckpt_path))
            eval_test_finetune(cur_status,sess, model,curlog)

def test():
    init_seed()
    status = ['wc', 'cw']
    for cur_status in status[conf_args.start:conf_args.end]:
        cur_status_model_save_dir = MODEL_SAVE_DIR + cur_status + '/'
        print(cur_status_model_save_dir)
        if not os.path.exists(cur_status_model_save_dir):
            os.mkdir(cur_status_model_save_dir)
        if MODEL_NAME != 'our':
            train_set = pickle.load(open(USE_DATA_DIR + '%s_slide_train.pkl' % cur_status, 'rb'))
        else:
            train_set = pickle.load(open(USE_DATA_DIR + '%s_slide_train.pkl' % cur_status, 'rb'))
        random.shuffle(train_set)
        train_samples=len(train_set)
        batches = train_samples // BATCH_SIZE
        if train_samples % BATCH_SIZE:
            batches += 1
        if not os.path.exists(MODEL_SAVE_DIR):
            os.mkdir(MODEL_SAVE_DIR)
        if not os.path.exists(MODEL_LOG_DIR):
            os.mkdir(MODEL_LOG_DIR)

        curlog=MyLog(MODEL_LOG_DIR+MODEL_LOG_NAME)
        parser=conf_parser
        try:
            parser.add_argument('--batches', default=batches, type=int)
        except Exception as e:
            parser.set_defaults(batches=batches)
        args = parser.parse_args()

        if args.ablation_type:
            cur_status_model_save_dir=cur_status_model_save_dir+"ablation_"+str(args.ablation_type)+'/'
            if not os.path.exists(cur_status_model_save_dir):
                os.mkdir(cur_status_model_save_dir)

        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        tf.reset_default_graph()
        with tf.Session(config=gpu_config) as sess:
            model=load_model(sess,curlog,args,cur_status,True)
            eval_test_finetune(cur_status,sess, model, curlog)


if __name__ == '__main__':
    train()
    #test()