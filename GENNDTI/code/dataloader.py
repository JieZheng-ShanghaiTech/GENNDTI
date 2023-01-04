from telnetlib import SE
import torch
from torch_geometric.data import InMemoryDataset, Data, Dataset
from torch_geometric.utils import to_undirected
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random
import numpy as np
import pickle
import pandas as pd
import os.path as osp
import itertools
import os
from icecream import ic


class Dataset(InMemoryDataset):
    def __init__(self, root, dataset, rating_file, sep, args, transform=None, pre_transform=None):

        self.path = root
        self.dataset = dataset
        self.rating_file = rating_file
        self.split_name = f'split_s{args.split + 1}'
        self.sep = sep
        self.args = args
        super(Dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.stat_info = torch.load(self.processed_paths[1])
        self.data_num = self.stat_info['data_num']
        self.feature_num = self.stat_info['feature_num']



    @property
    def raw_file_names(self):
        return ['{}{}/drug_dict.pkl'.format(self.path, self.dataset),
                '{}{}/target_dict.pkl'.format(self.path, self.dataset),
                '{}{}/feature_dict.pkl'.format(self.path, self.dataset),
                '{}{}/{}'.format(self.path, self.dataset, self.rating_file)]

    @property
    def processed_file_names(self):
        return ['{}/{}/{}.dataset'.format(self.dataset, self.split_name,self.dataset),
                '{}/{}/{}.statinfo'.format(self.dataset, self.split_name,self.dataset)]


    def download(self):
        # Download to `self.raw_dir`.
        pass


    def data_2_graphs(self, ratings_df, dataset='train'):
        graphs = []
        processed_graphs = 0
        num_graphs = ratings_df.shape[0]
        one_per = int(num_graphs/1000)
        if one_per==0:
            one_per = 1
        percent = 0.0
        print(ratings_df)
        for i in range(len(ratings_df)):
            if processed_graphs % one_per == 0:
                print(f"Processing [{dataset}]: {percent/10.0}%, {processed_graphs}/{num_graphs}", end="\r")
                percent += 1
            processed_graphs += 1 
            line = ratings_df.iloc[i]
            user_index = self.user_key_type(line[0])
            item_index = self.item_key_type(line[1])
            rating = int(line[2])

            if item_index not in self.target_dict or user_index not in self.drug_dict:
                error_num += 1
                continue

            user_id = self.drug_dict[user_index]['name']
            item_id = self.target_dict[item_index]['title']

            user_attr_list = self.drug_dict[user_index]['attribute']
            item_attr_list = self.target_dict[item_index]['attribute']

            user_list = [user_id] + user_attr_list
            item_list = [item_id] + item_attr_list

            graph = self.construct_graphs(user_list, item_list, rating)

            graphs.append(graph)

        return graphs



    def read_data(self):
        self.drug_dict = pickle.load(open(self.userfile, 'rb'))
        self.target_dict = pickle.load(open(self.itemfile, 'rb'))
        self.user_key_type = type(list(self.drug_dict.keys())[0])
        self.item_key_type = type(list(self.target_dict.keys())[0])
        try:
            feature_dict = pickle.load(open(self.featurefile, 'rb'))
        except:pass
        # print(feature_dict)

        data = []
        error_num = 0
        train_df = pd.read_csv(f"{self.path}{self.dataset}/{self.split_name}/train_data.csv", header=None)
        valid_df = pd.read_csv(f"{self.path}{self.dataset}/{self.split_name}/valid_data.csv", header=None)
        test_df = pd.read_csv(f"{self.path}{self.dataset}/{self.split_name}/test_data.csv", header=None)

        print('(Only run at the first time training the dataset)')
        train_graphs = self.data_2_graphs(train_df, dataset='train')
        valid_graphs = self.data_2_graphs(valid_df, dataset='valid')
        test_graphs = self.data_2_graphs(test_df, dataset='test')

        graphs = train_graphs + valid_graphs + test_graphs 

        stat_info = {}
        stat_info['data_num'] = len(graphs)
        try:
            stat_info['feature_num'] = len(feature_dict)
        except:stat_info['feature_num']=len(self.drug_dict)+len(self.target_dict)

        stat_info['train_test_split_index'] = [len(train_graphs), len(train_graphs) + len(valid_graphs)]

        print('error number of data:', error_num,"\n")
        return graphs, stat_info


    def construct_graphs(self, user_list, item_list, rating):

        u_n = len(user_list)   # user node number
        i_n = len(item_list)   # item node number

        # construct full inner edge
        inner_edge_index = [[],[]]
        for i in range(u_n):
            for j in range(i, u_n):
                inner_edge_index[0].append(i)
                inner_edge_index[1].append(j)

        for i in range(u_n, u_n + i_n ):
            for j in range(i, u_n + i_n):
                inner_edge_index[0].append(i)
                inner_edge_index[1].append(j)

        # construct outer edge
        outer_edge_index = [[],[]]
        for i in range(u_n):
            for j in range(i_n):
                outer_edge_index[0].append(i)
                outer_edge_index[1].append(u_n + j)

        #construct graph
        inner_edge_index = torch.LongTensor(inner_edge_index)
        inner_edge_index = to_undirected(inner_edge_index)
        outer_edge_index = torch.LongTensor(outer_edge_index)
        outer_edge_index = to_undirected(outer_edge_index)
        graph = self.construct_graph(user_list + item_list, inner_edge_index, outer_edge_index, rating)

        return graph


    def construct_graph(self, node_list, edge_index_inner, edge_index_outer, rating):
        x = torch.LongTensor(node_list).unsqueeze(1)
        rating = torch.FloatTensor([rating])
        return Data(x=x, edge_index=edge_index_inner, edge_attr=torch.transpose(edge_index_outer, 0, 1), y=rating)

    def process(self):
        self.userfile  = self.raw_file_names[0]
        self.itemfile  = self.raw_file_names[1]
        self.featurefile = self.raw_file_names[2]
        self.ratingfile  = self.raw_file_names[3]
        graphs, stat_info = self.read_data()
        #check whether foler path exist
        if not os.path.exists(f"{self.path}processed/{self.dataset}"):
            os.mkdir(f"{self.path}processed/{self.dataset}")
        if not os.path.exists(f"{self.path}processed/{self.dataset}/{self.split_name}"):
            os.mkdir(f"{self.path}processed/{self.dataset}/{self.split_name}")


        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])

        torch.save(stat_info, self.processed_paths[1])

    def feature_N(self):
        return self.feature_num

    def data_N(self):
        return self.data_num


