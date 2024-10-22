import torch
import numpy as np
import time, datetime
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import GNNModel as GNNModel
from utils import *
from tqdm import tqdm
from collections import defaultdict

class BaseModel(object):
    def __init__(self, args, loader_train, loader_eval):
        loader = loader_train
        self.model = GNNModel(args, loader)
        self.model.cuda()
        self.args = args
        self.n_batch = args.n_batch
        self.n_tbatch = args.n_tbatch
        self.n_layer = args.n_layer
        self.loader = loader_train
        self.loader_eval = loader_eval
        self.optimizer = Adam(self.model.parameters(), lr=args.lr, weight_decay=args.lamb)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=1, verbose=True, min_lr=1e-5)
        self.modelName = f'{args.n_layer}-layers'
        print(f'==> model name: {self.modelName}')

    def saveModelToFiles(self, best_metric, deleteLastFile=True):
        savePath = f'{self.loader.task_dir}/saveModel/{self.modelName}-{best_metric}.pt'
        print(f'Save checkpoint to : {savePath}')
        torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_mrr':best_metric,
                }, savePath)

        if deleteLastFile and self.lastSaveGNNPath != None:
            print(f'Remove last checkpoint: {self.lastSaveGNNPath}')
            os.remove(self.lastSaveGNNPath)

        self.lastSaveGNNPath = savePath

    def loadModel(self, filePath, layers=-1):
        print(f'Load weight from {filePath}')
        assert os.path.exists(filePath)
        checkpoint = torch.load(filePath, map_location=torch.device(f'cuda:{self.args.gpu}'))

        if layers != -1:
            extra_layers = self.model.gnn_layers[layers:]
            self.model.gnn_layers = self.model.gnn_layers[:layers]
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.gnn_layers += extra_layers
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def train_batch(self, epoch=0):
        epoch_loss = 0
        batch_size = self.n_batch
        self.loader.shuffle_train(ratio=self.args.train_ratio, remove_one_loop=self.args.remove_one_loop) # 0.1
        epoch_rand_idx = np.random.permutation(self.loader.n_train)
        n_epoch_train = len(epoch_rand_idx)
        n_batch = n_epoch_train // batch_size + (n_epoch_train % batch_size > 0)
        self.model.train()
        for i in tqdm(range(n_batch)):
            self.model.set_loader(self.loader, "train")
            start = i*batch_size
            end = min(n_epoch_train, (i+1)*batch_size)
            batch_idx = np.array(epoch_rand_idx[start: end]) #随机顺序
            triple = self.loader.get_batch(batch_idx)
            input_batch = np.stack([triple[:, 0], triple[:, 1], triple[:, 2], batch_idx], axis=1)  # [h,r,fid]
            self.model.zero_grad()
            scores = self.model.forward(input_batch)
            batch_size = len(scores)

            pos_scores = scores[[torch.arange(batch_size).cuda(), torch.LongTensor(triple[:,2]).cuda()]].clone()
            loss = torch.sum(- pos_scores + torch.log(torch.sum(torch.exp(scores), 1)))

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # avoid NaN
            for p in self.model.parameters():
                X = p.data.clone()
                flag = X != X
                X[flag] = np.random.random()
                p.data.copy_(X)
            epoch_loss += loss.item()
            del(loss, scores)
        return epoch_loss

    def evaluate(self, loader, verbose=False, eval_val=True, eval_test=False, recordDistance=False):
        self.model.eval()
        self.model.set_loader(loader, "test")
        batch_size = self.n_tbatch
        # - - - - - - val set - - - - - -
        if not eval_val:
            v_mrr, v_h1, v_h3, v_h10 = -1, -1, -1, -1
        else:
            n_data = loader.n_valid
            n_batch = n_data // batch_size + (n_data % batch_size > 0)
            ranking = []
            # - - - - - - val set - - - - - -
            iterator = tqdm(range(n_batch)) if verbose else range(n_batch)
            for i in iterator:
                start = i*batch_size
                end = min(n_data, (i+1)*batch_size)
                batch_idx = np.arange(start, end)
                subs, rels, objs = loader.get_batch(batch_idx, data='valid')
                input_batch = np.stack([subs, rels, batch_idx], axis=1)  # [h,r,fid]
                scores = self.model(input_batch, mode='valid')
                scores = scores.data.cpu().numpy()

                filters = []
                for i in range(len(subs)):
                    filt = loader.filters[(subs[i], rels[i])]
                    filt_1hot = np.zeros((loader.n_ent,))
                    filt_1hot[np.array(filt)] = 1
                    filters.append(filt_1hot)
                filters = np.array(filters)

                # scores / objs / filters: [batch_size, n_ent]
                ranks = cal_ranks(scores, objs, filters)
                ranking += ranks

            ranking = np.array(ranking)
            v_mrr, v_h1, v_h3, v_h10 = cal_performance(ranking)

        # - - - - - - test set - - - - - -
        if not eval_test:
            t_mrr, t_h1, t_h3, t_h10 = -1, -1, -1, -1
        else:
            n_data = loader.n_test
            n_batch = n_data // batch_size + (n_data % batch_size > 0)
            ranking = []
            self.model.eval()
            iterator = tqdm(range(n_batch)) if verbose else range(n_batch)
            for i in iterator:
                start = i*batch_size
                end = min(n_data, (i+1)*batch_size)
                batch_idx = np.arange(start, end)
                subs, rels, objs = loader.get_batch(batch_idx, data='test')
                input_batch = np.stack([subs, rels, batch_idx], axis=1)  # [h,r,fid]
                scores = self.model(input_batch, mode='test')
                scores = scores.data.cpu().numpy()
                filters = []
                for i in range(len(subs)):
                    filt = loader.filters[(subs[i], rels[i])]
                    filt_1hot = np.zeros((loader.n_ent, ))
                    filt_1hot[np.array(filt)] = 1
                    filters.append(filt_1hot)

                filters = np.array(filters)
                ranks = cal_ranks(scores, objs, filters)
                ranking += ranks

            ranking = np.array(ranking)
            t_mrr, t_h1, t_h3, t_h10 = cal_performance(ranking)
        return v_mrr, v_h1, v_h3, v_h10, t_mrr, t_h1, t_h3, t_h10

