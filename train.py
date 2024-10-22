import random
import os
import argparse
import torch
import time, datetime
import pickle as pkl
import numpy as np

from load_data import DataLoader
from base_model import BaseModel
from utils import *

parser = argparse.ArgumentParser(description="Parser for PRINCE")
# train-related
parser.add_argument('--data_path', type=str, default='data/WN18RR/') # fb15k-237  WN18RR  nell  obgl_biokg  obgl_wikikg2
parser.add_argument('--task_mode', type=str, default='transductive') # inductive
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--max_epoch', type=int, default=50)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--n_batch', type=int, default=16)
parser.add_argument('--n_tbatch', type=int, default=16)
parser.add_argument('--train_ratio', type=float, default=1)
parser.add_argument('--eval_verbose', type=bool, default=False)
parser.add_argument('--training_mode', type=str, default="small") # small large  precom  threehop  leaffill

# model-related
parser.add_argument('--lr', type=float, default=5e-3)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--n_layer', type=int, default=4)
parser.add_argument('--attn_dim', type=int, default=8)
parser.add_argument('--n_layer2', type=int, default=1)
parser.add_argument('--MESS_FUNC', type=str, default='DistMult') # DistMult RotatE TransE
parser.add_argument('--AGG_FUNC', type=str, default='pna') #sum mean
parser.add_argument('--remove_one_loop', type=bool, default=False)
parser.add_argument('--decay_rate', type=float, default=0.8)
parser.add_argument('--lamb', type=float, default=1e-4)
parser.add_argument('--dropout', type=float, default=0.1)
args = parser.parse_args()

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    import torch
    torch.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = True  
    
if __name__ == '__main__':
    opts = args
    setup_seed(args.seed)
    dataset = args.data_path
    dataset = dataset.split('/')
    if len(dataset[-1]) > 0:
        dataset = dataset[-1]
    else:
        dataset = dataset[-2]
    opts.dataset = dataset
    if dataset == "fb15k-237": opts.remove_one_loop = True 
    opts.data_type = torch.float32
    torch.cuda.set_device(opts.gpu)
    print('==> gpu:', opts.gpu)
    if opts.task_mode == "transductive":
        loader_train = DataLoader(opts.data_path, mode = opts.task_mode)
        loader_eval = loader_train
    else: # inductive
        loader_train = DataLoader(opts.data_path, mode = opts.task_mode)
        loader_eval = DataLoader(opts.data_path[:-1] + '_ind/', loader_train.relation2id, mode = opts.task_mode)

    opts.dataset = dataset
    opts.date = str(time.asctime(time.localtime(time.time())))

    # check all output paths
    checkPath('./results/')
    # build model w.r.t. opts
    model = BaseModel(opts, loader_train, loader_eval)
    opts_str = str(opts)
    # train model
    best_mrr = 0
    best_test_mrr = 0
    val_eval_dict, test_eval_dict = {}, {}
    total_time = 0
    for epoch in range(opts.max_epoch):

        epoch_loss = model.train_batch(epoch)
        v_mrr, v_h1, v_h3, v_h10, t_mrr, t_h1, t_h3, t_h10 = model.evaluate(model.loader_eval, verbose=args.eval_verbose, eval_val=True, eval_test=True)
        val_eval_dict[epoch + 1] = (v_mrr, v_h1, v_h3, v_h10)
        test_eval_dict[epoch + 1] = (t_mrr, t_h1, t_h3, t_h10)
        model.scheduler.step(v_mrr)
        epoch_str = '[VALID] MRR:%.4f H@1:%.4f H@3:%.4f H@10:%.4f\t [TEST] MRR:%.4f H@1:%.4f H@3:%.4f H@10:%.4f'%(v_mrr, v_h1, v_h3, v_h10, t_mrr, t_h1, t_h3, t_h10)
        print(f'{epoch + 1}\t[TRAIN] LOSS: {epoch_loss} TIME: {total_time}' + '\t' + epoch_str)
        if v_mrr > best_mrr:
            best_mrr, best_test_mrr = v_mrr, t_mrr
            best_str = epoch_str
    # save to local file
    print("###################Train_finished##########################")
    print(best_str)
    print("#############################################")
