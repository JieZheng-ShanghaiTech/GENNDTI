import numpy as np
import pandas as pd

from layers import *


class GENNDTI(nn.Module):
    """
    GENNDTI main model
    """

    def __init__(self, args, n_features, device):
        super(GENNDTI, self).__init__()

        self.n_features = n_features
        self.dim = args.dim
        self.hidden_layer = args.hidden_layer
        self.device = device
        self.batch_size = args.batch_size
        self.num_user_features = args.num_user_features
        self.inner_choice = args.inner_model
        self.cross_choice = args.cross_model

        if self.inner_choice == 0:
            self.inner_gnn = inner_GNN(self.dim, self.hidden_layer)
        elif self.inner_choice == 1:
            print("inner choice is gcn")
            self.inner_gnn = standard_GCN1(self.dim, self.hidden_layer, self.dim)
        elif self.inner_choice == 2:
            self.inner_gnn = GAT(self.dim, self.hidden_layer, self.dim)
        elif self.inner_choice == 3:
            pass

        if self.cross_choice == 0:
            self.outer_gnn = cross_GNN(self.dim, self.hidden_layer)
        elif self.cross_choice == 1:
            print("cross choice is gcn")
            self.outer_gnn = standard_GCN(self.dim, self.hidden_layer, self.dim)
        elif self.cross_choice == 2:
            self.outer_gnn = GAT(self.dim, self.hidden_layer, self.dim)
        elif self.cross_choice == 3:
            pass

        # self.feature_embedding.weight.data.normal_(0.0,0.01)

        ###################
        drug_embed = pd.read_pickle(f'../data/{args.dataset}/drug_bind.pkl')
        target_embed = pd.read_pickle(f'../data/{args.dataset}/target_bind.pkl')
        # fp = drug_embed['fp']
        fp = np.random.normal(size=(len(drug_embed), 64))
        sec = np.random.normal(size=(len(target_embed), 64))
        if args.dataset=='davis':
            sec = target_embed['emb']
            pre_train = np.concatenate([fp, np.array(sec.to_list())])
        elif args.dataset=='KIBA':
            # fp = drug_embed
            # sec = target_embed
            pre_train = np.concatenate([np.array(fp), np.array(sec)])
        # pre_train = np.array(pd.concat([fp, sec]).to_list())

        pad_num = self.n_features + 1 - len(pre_train)
        print(self.n_features)
        if pad_num != 0:
            pad = np.random.normal(size=(pad_num, 64))
            pre_train = np.concatenate([pre_train, pad])

        # pre_train = np.random.normal(size=(self.n_features+1, 64))
        pre_train = pre_train.astype(np.float32)
        ###########################################
        self.feature_embedding = nn.Embedding.from_pretrained((torch.from_numpy(pre_train)))
        #
        self.feature_embedding = nn.Embedding(self.n_features + 1, self.dim)
        self.node_weight = nn.Embedding(self.n_features + 1, 1)
        self.node_weight.weight.data.normal_(0.0, 0.01)
        self.update_f = nn.GRU(input_size=self.dim, hidden_size=self.dim, dropout=0.5)
        self.g = nn.Linear(self.dim, 1, bias=False)

    def forward(self, data, is_training=True):
        # does not conduct link prediction, use all interactions

        node_id = data.x.to(self.device)
        batch = data.batch

        # handle pointwise features
        node_w = torch.squeeze(self.node_weight(node_id))
        sum_weight = global_add_pool(node_w, batch)

        node_emb = self.feature_embedding(node_id)
        inner_edge_index = data.edge_index
        outer_edge_index = torch.transpose(data.edge_attr, 0, 1)
        outer_edge_index = self.outer_offset(batch, self.num_user_features, outer_edge_index)

        # outer_node_message = self.outer_gnn(node_emb, outer_edge_index)

        inner_node_message = self.inner_gnn(node_emb, inner_edge_index)
        outer_node_message = self.outer_gnn(node_emb, outer_edge_index)
        # aggregate all message
        if len(outer_node_message.size()) < len(node_emb.size()):
            outer_node_message = outer_node_message.unsqueeze(1)
            inner_node_message = inner_node_message.unsqueeze(1)
        updated_node_input = torch.cat((node_emb, inner_node_message, outer_node_message), 1)
        updated_node_input = torch.transpose(updated_node_input, 0, 1)
        # print("**********node,inner,outer",node_emb.shape,inner_node_message.shape,outer_node_message.shape)
        # print('updated_node_input',updated_node_input.shape)
        
        gru_h0 = torch.normal(0, 0.01, (1, node_emb.size(0), self.dim)).to(self.device)
        gru_output, hn = self.update_f(updated_node_input, gru_h0)
        updated_node = gru_output[-1]  # [batch_size*n_node, dim]
        # print('updated_node',updated_node.shape)
        
        new_batch = self.split_batch(batch, self.num_user_features)
        updated_graph = torch.squeeze(global_mean_pool(updated_node, new_batch))
        item_graphs, user_graphs = torch.split(updated_graph, int(updated_graph.size(0) / 2))
        # print('updated_graph',updated_graph.shape)
        # print('item_graphs, user_graphs',item_graphs.shape, user_graphs.shape)
        
        y = torch.unsqueeze(torch.sum(user_graphs * item_graphs, 1) + sum_weight, 1)
        # print('y.unsqueeze',y.shape)
        y = torch.sigmoid(y)
        # print('y',y.shape)
        # if not torch.squeeze(y)==Tensor([]):
        #     print(f'{user_graphs}, {item_graphs}, {sum_weight}')
        return y

    def split_batch(self, batch, user_node_num):
        """
        split batch id into user nodes and item nodes
        """
        ones = torch.ones_like(batch)
        nodes_per_graph = global_add_pool(ones, batch)
        cum_num = torch.cat((torch.LongTensor([0]).to(self.device), torch.cumsum(nodes_per_graph, dim=0)[:-1]))
        cum_num_list = [cum_num + i for i in range(user_node_num)]
        multi_hot = torch.cat(cum_num_list)
        test = torch.sum(F.one_hot(multi_hot, ones.size(0)), dim=0) * (torch.max(batch) + 1)

        return batch + test

    def outer_offset(self, batch, user_node_num, outer_edge_index):
        ones = torch.ones_like(batch)
        nodes_per_graph = global_add_pool(ones, batch)
        inter_per_graph = (nodes_per_graph - user_node_num) * user_node_num * 2
        cum_num = torch.cat((torch.LongTensor([0]).to(self.device), torch.cumsum(nodes_per_graph, dim=0)[:-1]))
        offset_list = torch.repeat_interleave(cum_num, inter_per_graph, dim=0).repeat(2, 1)
        # print(offset_list.shape,outer_edge_index.shape)
        # outer_edge_index_offset = outer_edge_index + offset_list
        outer_edge_index_offset = torch.cat((outer_edge_index, offset_list), dim=-1)
        return outer_edge_index_offset






