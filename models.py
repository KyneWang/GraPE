import torch
import torch.nn as nn
import numpy as np
from utils import *
from functools import reduce
from torch_scatter import scatter
from scipy.sparse import csr_matrix

class GNNLayer(torch.nn.Module):
    def __init__(self, params, loader, hidden_dim):
        super(GNNLayer, self).__init__()
        self.params = params
        self.loader = loader
        self.n_rel = loader.n_rel
        self.n_ent = loader.n_ent
        self.in_dim = hidden_dim
        self.out_dim = hidden_dim
        self.act = nn.ReLU()
        self.MESS_FUNC = params.MESS_FUNC.replace("\r", "")
        self.AGG_FUNC = params.AGG_FUNC.replace("\r", "")

        self.relation_embed = nn.Embedding(2 * self.n_rel + 2, self.in_dim)
        self.relation_trans = nn.Linear(self.in_dim * 2, self.in_dim)
        if self.AGG_FUNC == "pna":
            self.W_h = nn.Linear(self.in_dim * 6, self.out_dim, bias=False)
        else:
            self.W_h = nn.Linear(self.in_dim * 1, self.out_dim, bias=False)
        self.layer_norm = nn.LayerNorm(self.in_dim, elementwise_affine=False)

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    # forward for part edges, with flatten data
    def forward(self, query, layer_input, edges, nodes, edge_count):
        sub, rel, obj = edges[:, 4], edges[:, 2], edges[:, 5]
        relem_extend = self.relation_embed.weight.unsqueeze(0).repeat(len(query),1,1)
        query_extend = query.unsqueeze(1).repeat(1,relem_extend.shape[1],1)
        relation = self.relation_trans(torch.cat([query_extend, relem_extend], dim=2))
        input_j = layer_input[sub]
        input_mask = input_j.sum(-1).unsqueeze(-1) > 0
        relation_j = relation[edges[:, 0], edges[:, 2]]

        if self.MESS_FUNC == 'TransE':
            message = (input_j + relation_j) * input_mask
        elif self.MESS_FUNC == 'DistMult':
            message = input_j * relation_j
        elif self.MESS_FUNC == 'RotatE':
            hs_re, hs_im = input_j.chunk(2, dim=-1)
            hr_re, hr_im = relation_j.chunk(2, dim=-1)
            message_re = hs_re * hr_re - hs_im * hr_im
            message_im = hs_re * hr_im + hs_im * hr_re
            message = torch.cat([message_re, message_im], dim=-1)

        # AGG() 扩散后的实体的相关message进行聚合， 得到[k2, d]
        if self.AGG_FUNC == 'pna':
            message_agg = self.pna_process(message, obj, item_size=len(nodes), edge_count=edge_count, scatter_dim=0)
        elif self.AGG_FUNC == 'mean':
            count_sum = edge_count.clamp(min=1)
            sum = scatter(message, index=obj, dim=0, dim_size=len(nodes), reduce="sum")
            message_agg = sum / count_sum
        else:
            message_agg = scatter(message, index=obj, dim=0, dim_size=nodes.shape[0], reduce=self.AGG_FUNC)
        message_agg = self.W_h(message_agg)
        message_agg = self.layer_norm(message_agg)
        hidden_new = self.act(message_agg)  # [n_node, dim]
        return hidden_new

    # forward for all edges, with batch data
    def forward2(self, query, layer_input, edges, edge_count = None):
        sub, rel, obj = edges[:, 0], edges[:, 1], edges[:, 2]
        relem_extend = self.relation_embed.weight.unsqueeze(0).repeat(len(query), 1, 1)
        query_extend = query[:,:self.in_dim].unsqueeze(1).repeat(1, relem_extend.shape[1], 1)
        relation = self.relation_trans(torch.cat([query_extend, relem_extend], dim=2))

        input_j = layer_input.index_select(1, sub)
        relation_j = relation.index_select(1, rel)
        obj_index = obj
        if self.MESS_FUNC == 'TransE':
            message = input_j + relation_j
        elif self.MESS_FUNC == 'DistMult':
            message = input_j * relation_j
        elif self.MESS_FUNC == 'RotatE':
            hs_re, hs_im = input_j.chunk(2, dim=-1)
            hr_re, hr_im = relation_j.chunk(2, dim=-1)
            message_re = hs_re * hr_re - hs_im * hr_im
            message_im = hs_re * hr_im + hs_im * hr_re
            message = torch.cat([message_re, message_im], dim=-1)
        else:
            message = input_j * relation_j

        if self.AGG_FUNC == 'pna':
            message_agg = self.pna_process(message, obj, item_size=self.n_ent,
                                           edge_count=edge_count.unsqueeze(2), scatter_dim=1)
        elif self.AGG_FUNC == 'mean':
            count_sum = edge_count.unsqueeze(2).clamp(min=1)
            sum = scatter(message, index=obj, dim=1, dim_size=self.n_ent, reduce="sum")
            message_agg = sum / count_sum
        else:
            message_agg = scatter(message, index=obj_index, dim=1, dim_size=self.n_ent, reduce=self.AGG_FUNC)

        if len(query)== 1 and len(message_agg.shape) == 2:
            message_agg = message_agg.unsqueeze(0)
        message_agg = self.W_h(message_agg)
        message_agg = self.layer_norm(message_agg)
        hidden_new = self.act(message_agg)  # [n_node, dim]
        return hidden_new

    def pna_process(self, message, obj_index, item_size, edge_count, scatter_dim=0):
        count_sum = edge_count.clamp(min=1)
        count_sum2 = scatter(torch.ones(list(message.shape)[:-1] + [1]).cuda(), index=obj_index, dim=scatter_dim,
                       dim_size=item_size, reduce="sum").clamp(min=1)

        sum = scatter(message, index=obj_index, dim=scatter_dim, dim_size=item_size, reduce="sum")
        mean = sum / count_sum
        sq_sum = scatter(message ** 2, index=obj_index, dim=scatter_dim, dim_size=item_size, reduce="sum")
        sq_mean = sq_sum / count_sum

        std = (sq_mean - mean ** 2).clamp(min=1e-6).sqrt()
        std = std * (mean != 0)
        features = torch.cat([mean.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
        features = features.flatten(-2)

        scale = count_sum2.log()
        scale = scale / (scale.mean().clamp(min=1))

        scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
        message_agg = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        message_agg = message_agg.squeeze(0)
        return message_agg

class GNNModel(torch.nn.Module):
    def __init__(self, params, loader):
        super(GNNModel, self).__init__()
        print(params)
        self.params = params
        self.loader = loader

        self.n_layer = params.n_layer
        self.n_layer2 = params.n_layer2
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        self.n_ent = loader.n_ent
        self.n_rel = loader.n_rel

        self.remove_one_loop = params.remove_one_loop
        self.rela_embed = nn.Embedding(2 * self.n_rel + 2, self.hidden_dim)
        self.rela_embed2 = nn.Embedding(2 * self.n_rel + 2, self.attn_dim)

        self.gnn_layers = []
        self.gnn_layers.append(GNNLayer(self.params, self.loader, hidden_dim=self.hidden_dim))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)

        self.gnn_layers2 = []
        self.gnn_layers2.append(GNNLayer(self.params, self.loader, hidden_dim=self.attn_dim))
        self.gnn_layers2 = nn.ModuleList(self.gnn_layers2)

        self.dropout = nn.Dropout(params.dropout)

        bridge = []
        bridge.append(nn.Linear(self.hidden_dim * 2, self.hidden_dim, bias=False))
        bridge.append(nn.ReLU())
        bridge.append(nn.Linear(self.hidden_dim, self.attn_dim, bias=False))
        self.bridge = nn.Sequential(*bridge)
        final = []
        final.append(nn.Linear(self.attn_dim * 2, self.attn_dim))
        final.append(nn.ReLU())
        final.append(nn.Linear(self.attn_dim, 1))
        self.final = nn.Sequential(*final)

        # calculate parameters
        num_parameters = sum(p.numel() for p in self.parameters())
        # for name, param in self.state_dict().items():
        #     print(name, param.shape)
        print('==> num_parameters: {}'.format(num_parameters))

    def forward(self, triples, mode='train'):
        n = len(triples)
        KG = self.loader.tKG
        self.edge_data = torch.LongTensor(KG).cuda()
        batch_triples = torch.cat([torch.arange(n).unsqueeze(1).cuda(),
                                   torch.LongTensor(triples).cuda()], 1)  # [bid, h, r, t, fid]
        q_sub, q_rel, q_obj = batch_triples[:, 1], batch_triples[:, 2], batch_triples[:, 3]  # [B]
        start_nodes = batch_triples[:, [0, 1]]  # [B, 2] with (batch_idx, node_idx)

        if mode == 'train':
            h_index_ext = torch.cat([q_sub, q_obj], dim=-1)
            t_index_ext = torch.cat([q_obj, q_sub], dim=-1)
            r_index_ext = torch.cat([q_rel, torch.where(q_rel < self.n_rel, q_rel + self.n_rel, q_rel - self.n_rel)], dim=-1)
            extend_triples = torch.stack([h_index_ext, r_index_ext, t_index_ext], dim=0)
            if self.remove_one_loop:
                filter_index = edge_match(self.edge_data.T[[0, 2], :], extend_triples[[0, 2], :])[0]  # for FB
            else:
                filter_index = edge_match(self.edge_data.T, extend_triples)[0]
            filter_mask = ~index_to_mask(filter_index, len(self.edge_data))
            filter_mask = filter_mask | (self.edge_data[:,1]==(self.n_rel*2))
        else:
            filter_mask = torch.ones(len(self.edge_data)).cuda().bool()

        filter_data = self.edge_data[filter_mask]
        edge_count = scatter(torch.ones((len(filter_data), 1)).cuda(), index=filter_data[:, 2], dim=0,
                             dim_size=self.n_ent, reduce="sum")
        batch_size = len(q_rel)
        query = self.rela_embed(q_rel)
        nodes = start_nodes

        layer_input = torch.ones_like(query)
        total_node_1hot = 0
        for i in range(self.n_layer):
            nodes, edges, edge_1hot, total_node_1hot, old_nodes_new_idx = self.get_neighbors(nodes.data.cpu().numpy(),
                                                                     len(start_nodes), total_node_1hot,
                                                                     filter_mask=filter_mask.unsqueeze(1).cpu().numpy(),
                                                                     layer_id=i)  # 增加反向后相当于凭空增加一轮
            node_edge_count = edge_count[nodes[:,1]]
            hidden = self.gnn_layers[0].forward(query, layer_input, edges, nodes, node_edge_count) # 注意此处可以共享参数，那么重复层是否可以减少计算
            hidden = self.dropout(hidden)
            previous_mes = torch.zeros_like(hidden)
            previous_mes[old_nodes_new_idx] += layer_input
            hidden = hidden + previous_mes
            layer_input = hidden

        layer_output = torch.zeros((batch_size, self.n_ent, self.hidden_dim), device=q_sub.device)
        layer_output[nodes[:, 0], nodes[:, 1]] = layer_input
        node_query = query.unsqueeze(1).expand(-1, self.n_ent, -1)
        output = torch.cat([layer_output, node_query], dim=-1)
        layer_input = self.bridge(output)

        for i in range(self.n_layer2):
            hidden = self.gnn_layers2[0].forward2(query, layer_input, filter_data, edge_count.T.repeat(len(query), 1))  # filter_data
            layer_input = hidden + layer_input

        node_query = self.rela_embed2(q_rel).unsqueeze(1).expand(-1, self.n_ent, -1)
        scores_all = self.final(torch.cat([layer_input, node_query], dim=-1)).squeeze(-1)
        return scores_all

    def get_neighbors(self, nodes, batchsize, total_node_1hot=0, filter_mask=None, layer_id=0):
        KG = self.loader.tKG
        M_sub = self.loader.tM_sub

        node_1hot = csr_matrix((np.ones(len(nodes)), (nodes[:, 1], nodes[:, 0])), shape=(self.n_ent, batchsize))
        edge_1hot = node_1hot
        node_triples = M_sub[:, nodes[:, 1]].multiply(filter_mask)
        edges = np.nonzero(node_triples)
        edges_value = nodes[:, 0][edges[1]]
        edges = [edges[0], edges_value]

        sampled_edges = np.concatenate([np.expand_dims(edges[1], 1), KG[edges[0]]], axis=1)
        sampled_edges = torch.LongTensor(sampled_edges).cuda()

        head_nodes, head_index = torch.unique(sampled_edges[:, [0, 1]], dim=0, sorted=True, return_inverse=True)
        tail_nodes, tail_index = torch.unique(sampled_edges[:, [0, 3]], dim=0, sorted=True, return_inverse=True)

        sampled_edges = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)

        mask = sampled_edges[:, 2] == (self.n_rel * 2)
        old_nodes_new_idx = tail_index[mask].sort()[0]
        total_node_1hot += node_1hot
        sampled_edges_masked = sampled_edges
        if layer_id > 0:
            last_mask = total_node_1hot[KG[edges[0], -1], edges[1]] > 1 #1 0
            last_mask2 = total_node_1hot[KG[edges[0], 0], edges[1]] > 1
            last_mask = last_mask + last_mask2
            last_mask = ~torch.from_numpy(last_mask).squeeze(0).bool().cuda()
            sampled_edges_masked = sampled_edges[last_mask]
        return tail_nodes, sampled_edges_masked, edge_1hot, total_node_1hot, old_nodes_new_idx

    def set_loader(self, loader, mode="train"):
        self.loader = loader
        self.n_rel = loader.n_rel  # ind_rel数量可能不完整
        self.n_ent = loader.n_ent
        for gnn_layer in self.gnn_layers:
            gnn_layer.n_rel = loader.n_rel
            gnn_layer.n_ent = loader.n_ent
        for gnn_layer in self.gnn_layers2:
            gnn_layer.n_rel = loader.n_rel
            gnn_layer.n_ent = loader.n_ent


