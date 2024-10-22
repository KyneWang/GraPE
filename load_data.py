import os
import torch
import datetime
from tqdm import tqdm
import numpy as np
import pickle as pkl
import scipy
from utils import *
from scipy.sparse import csr_matrix
from collections import defaultdict

class DataLoader:
    def __init__(self, task_dir, relation2id=None, mode="transductive"):
        self.task_dir = task_dir
        self.mode = mode
        self.train_spread_edges = None
        self.fact_spread_edges = None

        with open(os.path.join(task_dir, 'entities.txt')) as f:
            self.entity2id = dict()
            n_ent = 0
            for line in f:
                entity = line.strip().split()[0]
                self.entity2id[entity] = n_ent
                n_ent += 1

        # 默认关系数量一致，后续需要调整
        if relation2id == None:
            with open(os.path.join(task_dir, 'relations.txt')) as f:
                self.relation2id = dict()
                n_rel = 0
                for line in f:
                    relation = line.strip().split()[0]
                    self.relation2id[relation] = n_rel
                    n_rel += 1
        else:
            self.relation2id = relation2id
            n_rel = len(relation2id)

        self.n_ent = n_ent
        self.n_rel = n_rel
        self.filters = defaultdict(lambda:set())
        self.trainfilters = defaultdict(lambda: set())

        self.train_triple = self.read_triples('train.txt', mode="train")
        self.valid_triple = self.read_triples('valid.txt', mode="test")
        self.test_triple = self.read_triples('test.txt', mode="test")

        self.train_data = np.array(self.double_triple(self.train_triple))
        self.valid_data = self.double_triple(self.valid_triple)
        self.test_data = self.double_triple(self.test_triple)

        if self.mode == "transductive":
            self.fact_triple = self.read_triples('facts.txt', mode="train")
            self.fact_data = self.double_triple(self.fact_triple)
            self.all_observed_triple = self.train_triple + self.fact_triple
            self.all_observed_data = self.train_data.tolist() + self.fact_data
        else:
            self.all_observed_triple = self.train_triple
            self.all_observed_data = self.train_data.tolist()

        self.total_train_data = np.array(self.all_observed_data)
        self.n_total_train = len(self.all_observed_data)

        self.load_graph(self.all_observed_data)
        self.load_test_graph(self.all_observed_data)

        self.valid_q, self.valid_a = self.load_query(self.valid_data)
        self.test_q, self.test_a = self.load_query(self.test_data)
        self.trial_q, self.trial_a = self.load_query(self.test_data[:100])

        self.n_train = len(self.train_data)
        self.n_valid = len(self.valid_q)
        self.n_test = len(self.test_q)
        self.n_trial = len(self.trial_q)

        for filt in self.filters:
            self.filters[filt] = list(self.filters[filt])
        print("all_observed_data", len(self.all_observed_data))
        print('n_ent:', self.n_ent, 'n_rel:', self.n_rel)
        print('n_train:', self.n_train, 'n_valid:', self.n_valid, 'n_test:', self.n_test)

    def addToFilter(self, triples):
        for (h,r,t) in triples:
            self.filters[(h,r)].add(t)
            self.filters[(t,r+self.n_rel)].add(h)
        return

    def read_triples(self, filename, mode = "none"):
        triples = []
        with open(os.path.join(self.task_dir, filename)) as f:
            for line in f:
                h, r, t = line.strip().split()
                h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                triples.append([h,r,t])
                if mode == "train":
                    self.trainfilters[(h,r)].add(t)
                    self.trainfilters[(t, r + self.n_rel)].add(h)
                self.filters[(h,r)].add(t)
                self.filters[(t,r+self.n_rel)].add(h)
        return triples

    def double_triple(self, triples):
        new_triples = []
        for triple in triples:
            h, r, t = triple
            new_triples.append([t, r+self.n_rel, h]) 
        return triples + new_triples

    def load_graph(self, triples):
        # (e, r', e)
        # r' = 2 * n_rel, r' is manual generated and not exist in the original KG
        # self.KG: shape=(self.n_fact, 3)
        # M_sub shape=(self.n_fact, self.n_ent), store projection from head entity to triples
        idd = np.concatenate([np.expand_dims(np.arange(self.n_ent),1), 2*self.n_rel*np.ones((self.n_ent, 1)), np.expand_dims(np.arange(self.n_ent),1)], 1)

        self.KG = np.concatenate([np.array(triples), idd], 0)
        self.n_fact = len(self.KG)
        self.M_sub = csr_matrix((np.ones((self.n_fact,)), (np.arange(self.n_fact), self.KG[:, 0])), shape=(self.n_fact, self.n_ent))
        self.M_obj = csr_matrix((np.ones((self.n_fact,)), (np.arange(self.n_fact), self.KG[:, 2])), shape=(self.n_fact, self.n_ent))

    def load_test_graph(self, triples):
        idd = np.concatenate([np.expand_dims(np.arange(self.n_ent),1), 2*self.n_rel*np.ones((self.n_ent, 1)), np.expand_dims(np.arange(self.n_ent),1)], 1)

        self.tKG = np.concatenate([np.array(triples), idd], 0)
        self.tn_fact = len(self.tKG)
        self.tM_sub = csr_matrix((np.ones((self.tn_fact,)), (np.arange(self.tn_fact), self.tKG[:, 0])), shape=(self.tn_fact, self.n_ent))
        self.tM_obj = csr_matrix((np.ones((self.tn_fact,)), (np.arange(self.tn_fact), self.tKG[:, 2])), shape=(self.tn_fact, self.n_ent))

    def load_query(self, triples):
        trip_hr = defaultdict(lambda:list())

        for trip in triples:
            h, r, t = trip
            trip_hr[(h,r)].append(t)
        
        queries = []
        answers = []
        for key in trip_hr:
            queries.append(key)
            answers.append(np.array(trip_hr[key]))
        return queries, answers

    def get_batch(self, batch_idx, steps=2, data='train'):
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

        if data=='train':
            return np.array(self.train_data)[batch_idx]
        if data=='trial':
            query, answer = np.array(self.trial_q), np.array(self.trial_a)
        if data=='valid':
            query, answer = np.array(self.valid_q), np.array(self.valid_a)
        if data=='test':
            query, answer = np.array(self.test_q), np.array(self.test_a)
        subs = query[batch_idx, 0]
        rels = query[batch_idx, 1]
        objs = np.zeros((len(batch_idx), self.n_ent))
        for i in range(len(batch_idx)):
            objs[i][answer[batch_idx[i]]] = 1
        return subs, rels, objs

    def shuffle_train(self, ratio=0.25, remove_one_loop=False):
        all_triple = np.array(self.all_observed_triple)
        n_all = len(all_triple)
        random_mask = (np.random.random(n_all) <= ratio)
        remove_mask = np.ones(n_all)

        train_triple = all_triple[random_mask.astype(np.bool)]

        batch_triples = torch.LongTensor(train_triple)
        q_rel = batch_triples[:,1]
        batch_triples2 = torch.stack([batch_triples[:, 2],
                                      torch.where(q_rel < self.n_rel, q_rel + self.n_rel, q_rel - self.n_rel),
                                      batch_triples[:, 0]], dim=1)
        extend_triples = torch.cat([batch_triples, batch_triples2]).T

        if remove_one_loop:  # 修改一下通过载入fact_edges来计算，不再batch共享feature_mask，从而修改last_filter_edges
            edge_data = torch.LongTensor(all_triple.T[[0, 2], :])
            filter_index, num_match = edge_match(edge_data,
                                                      extend_triples[[0, 2], :])  # for FB
            filter_mask = ~index_to_mask(filter_index, len(all_triple))
            filter_mask = filter_mask | (all_triple[:, 1] == (self.n_rel * 2))  # 保留每个实体的自环关系， 是否有负面影响
            remove_mask = filter_mask.numpy()
        fact_triple = all_triple[((1-random_mask)*remove_mask).astype(np.bool)]
        np.random.shuffle(train_triple)
        np.random.shuffle(fact_triple)
        self.fact_data = self.double_triple(fact_triple.tolist())
        self.train_data = np.array(self.double_triple(train_triple.tolist()))
        self.n_train = len(self.train_data)
        if len(self.fact_data) > 0:
            self.load_graph(self.fact_data)