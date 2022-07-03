import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.ops import edge_softmax



PAD = 0


class GATConv(nn.Module):
    def __init__(self,
                 in_size,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None, ):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self.in_size = in_size
        self.residual = residual
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, in_size)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, in_size)))

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.activation = activation

        nn.init.xavier_normal_(self.attn_l, gain=1.414)
        nn.init.xavier_normal_(self.attn_r, gain=1.414)

    def forward(self, graph, feat):  # graph 是dgl 中的 graph,feat 是[node_num, head, in_size]
        with graph.local_scope():  # 为了不改变原图中的特征
            h_src = h_dst = self.feat_drop(feat).view(-1, self._num_heads, self.in_size)
            # feat_drop

            el = (h_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (h_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': h_src, 'el': el})
            graph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.residual:
                rst = rst + h_dst
            # activation
            if self.activation:
                rst = self.activation(rst)
            return rst


class HetGCNLayer(nn.Module):
    def __init__(self,
                 in_size,
                 aggregator_type='attention',
                 num_heads=8,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None):
        super(HetGCNLayer, self).__init__()

        self.num_heads = num_heads
        self.in_size = in_size
        if aggregator_type == 'pool':
            self.fc_pool = nn.Linear(in_size * num_heads, in_size * num_heads)
            nn.init.xavier_normal_(self.fc_pool.weight, gain=1.414)
        self.aggre_type = aggregator_type

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, self.num_heads, in_size)))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.residual = residual
        self.activation = activation

        nn.init.xavier_normal_(self.attn_l, gain=1.414)

    def forward(self, g, feat):
        with g.local_scope():
            if self.aggre_type == 'attention':
                if isinstance(feat, tuple):
                    h_src = self.feat_drop(feat[0]).view(-1, self.num_heads, self.in_size)
                    h_dst = self.feat_drop(feat[1]).view(-1, self.num_heads, self.in_size)
                el = (h_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
                g.srcdata.update({'ft': h_src, 'el': el})
                g.apply_edges(fn.copy_u('el', 'e'))
                e = self.leaky_relu(g.edata.pop('e'))
                g.edata['a'] = self.attn_drop(edge_softmax(g, e))
                g.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
                rst = g.dstdata['ft'].flatten(1)
                if self.residual:
                    rst = rst + h_dst
                if self.activation:
                    rst = self.activation(rst)

            elif self.aggre_type == 'mean':
                h_src = self.feat_drop(feat[0]).view(-1, self.in_size * self.num_heads)
                g.srcdata['ft'] = h_src
                g.update_all(fn.copy_u('ft', 'm'), fn.mean('m', 'ft'))
                rst = g.dstdata['ft']

            elif self.aggre_type == 'pool':
                h_src = self.feat_drop(feat[0]).view(-1, self.in_size * self.num_heads)
                g.srcdata['ft'] = F.relu(self.fc_pool(h_src))
                g.update_all(fn.copy_u('ft', 'm'), fn.mean('m', 'ft'))
                rst = g.dstdata['ft']
            return rst


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        return (beta * z).sum(1) , beta # (N, D * K)


class HeDANLayer(nn.Module):
    def __init__(self, meta_paths, in_size, aggre_type, layer_num_heads, dropout):
        super(HeDANLayer, self).__init__()
        self.nunm_heads = layer_num_heads
        self.semantic_attention_m = SemanticAttention(in_size=in_size * layer_num_heads)
        self.semantic_attention_u = SemanticAttention(in_size=in_size * layer_num_heads)
        # self.semantic_attention_d = SemanticAttention(in_size=in_size * layer_num_heads)

        self.hedan_layers = nn.ModuleList()

        self.hedan_layers.append(GATConv(in_size, layer_num_heads,
                                        dropout, dropout, activation=F.elu, residual=False))  # 'user' 'follow' 'user'
        self.hedan_layers.append(GATConv(in_size, layer_num_heads,
                                        dropout, dropout, activation=F.elu, residual=False))  # 'user' 'retweet' 'user'
        self.hedan_layers.append(GATConv(in_size, layer_num_heads,
                                        dropout, dropout, activation=F.elu, residual=False))  # 'user' 'follow' 'user'

        self.hedan_layers.append(HetGCNLayer(in_size, aggre_type, self.nunm_heads,
                                            dropout, dropout, activation=F.elu,
                                            residual=False))  # 'user' 'interest' 'message'

        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = {'message': [], 'user': []}
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()

            for meta_path in self.meta_paths:
                if meta_path in {('follow',), }:
                    print('******************follow**********************')
                    self._cached_coalesced_graph[meta_path] = dgl.edge_type_subgraph(g, [('user', 'follow', 'user')])
                elif meta_path in {('retweet',)}:
                    print('******************retweet**********************')
                    self._cached_coalesced_graph[meta_path] = dgl.edge_type_subgraph(g, [('user', 'retweet', 'user')])
                elif meta_path in {('interest',)}:
                    print('******************interest**********************')
                    self._cached_coalesced_graph[meta_path] = dgl.edge_type_subgraph(g,
                                                                                     [('user', 'interest', 'message')])
                elif meta_path in {('interested',)}:
                    print('******************interest**********************')
                    self._cached_coalesced_graph[meta_path] = dgl.edge_type_subgraph(g,
                                                                                     [(
                                                                                      'message', 'interested', 'user')])

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            if new_g.is_homogeneous:
                ntype = new_g.ntypes[0]
                semantic_embeddings[ntype].append(self.hedan_layers[i](new_g, h[ntype]).flatten(1))
            else:
                if meta_path in {('interest',), }:
                    h_ = (h['user'], h['message'])
                    semantic_embeddings['message'].append(self.hedan_layers[i](new_g, h_))
                elif meta_path in {('interested',)}:
                    h_ = (h['message'], h['user'])
                    semantic_embeddings['user'].append(self.hedan_layers[i](new_g, h_))

        embedings = {}
        for ntype in semantic_embeddings.keys():
            if ntype == 'message':
                semantic_embeddings[ntype] = torch.stack(semantic_embeddings[ntype], dim=1)
                embedings[ntype] = self.semantic_attention_m(semantic_embeddings[ntype])
            elif ntype == 'user' and semantic_embeddings[ntype]:
                semantic_embeddings[ntype] = torch.stack(semantic_embeddings[ntype], dim=1)
                embedings[ntype] = self.semantic_attention_u(semantic_embeddings[ntype])

        return embedings


class HeDAN(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, out_size, aggre_type, num_heads, dropout):
        super(HeDAN, self).__init__()

        self.fc_m = nn.Linear(in_size['message'], hidden_size * num_heads, bias=True)
        self.fc_u = nn.Linear(in_size['user'], hidden_size * num_heads, bias=True)

        self.fc_time = nn.Linear(in_size['user'], hidden_size * num_heads, bias=True)

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.layers = HeDANLayer(meta_paths, hidden_size, aggre_type, num_heads, dropout)

        self.semantic_attention_batch = SemanticAttention(in_size=hidden_size * num_heads)

        self.predict_m = nn.Linear(hidden_size * num_heads, out_size)
        self.predict_u = nn.Linear(hidden_size * num_heads, out_size)
        self.activation = F.elu
        self.dropout = nn.Dropout(0.6)
        nn.init.xavier_normal_(self.fc_m.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc_u.weight, gain=1.414)
        # nn.init.xavier_normal_(self.pos_embedding.weight, gain=1.414)

    def forward(self, tgt, tgt_timestamp, tgt_index, g, inputs):
        mask = (tgt == 0)
        h_trans = {}
        h_trans['message'] = self.fc_m(inputs['message']).view(-1, self.num_heads, self.hidden_size)
        h_trans['user'] = self.fc_u(inputs['user']).view(-1, self.num_heads, self.hidden_size)
        h_trans = self.layers(g, h_trans)

        batch_h_m = F.embedding(tgt_index, h_trans['message'])
        batch_h_u = F.embedding(tgt, h_trans['user'])
        batch_h = torch.cat((batch_h_m.unsqueeze(1), batch_h_u), dim=1)
        batch_h = self.semantic_attention_batch(batch_h)
        batch_pre = self.activation(self.predict_m(batch_h))


        return h_trans, batch_pre.squeeze(1)