import json
from collections import defaultdict

import numpy
import scipy.io as sio
import scipy.sparse
import scipy.sparse as sp
import numpy as np
import torch
from sklearn import preprocessing

from utils import normalize, sparse_matrix2torch_sparse_tensor

'''
                           _ooOoo_
                          o8888888o
                          88" . "88
                          (| -_- |)
                          O\  =  /O
                       ____/`---'\____
                     .'  \\|     |//  `.
                    /  \\|||  :  |||//  \
                   /  _||||| -:- |||||-  \
                   |   | \\\  -  /// |   |
                   | \_|  ''\---/''  |   |
                   \  .-\__  `-`  ___/-. /
                 ___`. .'  /--.--\  `. . __
              ."" '<  `.___\_<|>_/___.'  >'"".
             | | :  `- \`.;`\ _ /`;.`/ - ` : | |
             \  \ `-.   \_ __\ /__ _/   .-` /  /
        ======`-.____`-.___\_____/___.-`____.-'======
                           `=---='
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                 佛祖保佑       永无BUG
'''


def data_preprocess(dataset: str):
    """pre-process the data before training

        Parameter:
            dataset: the name of dataset, including 'Amazon_eletronics', 'dblp', 'cora-full' and 'ogbn-arxiv'
        Return:

    """
    train_class_list: list = list()
    test_class_list: list = list()
    valid_class_list: list = list()
    train_class_list, valid_class_list, test_class_list = json.load(
        open('./dataset/{}_class_split.json'.format(dataset)))
    if dataset == "Amazon_eletronics" or dataset == 'dblp':
        # all the edges in graph is denoted by (node1[i], node2[i])
        node1: list = list()
        node2: list = list()
        for line in open("./dataset/{}_network".format(dataset)):
            n1, n2 = line.strip().split("\t")
            node1.append(int(n1))
            node2.append(int(n2))
        node_number: int = max(max(node1), max(node2)) + 1

        # data_train and data_test are dicts, they have useful keys 'Index', 'Label' and 'Attributes'
        # Index: [[1,2,3...]]
        # Label: [[1],[1],[2],[2],...]
        # Attributes: matrix
        data_train: dict = sio.loadmat("./dataset/{}_train.mat".format(dataset))
        data_test: dict = sio.loadmat("./dataset/{}_test.mat".format(dataset))
        # label of nodes
        labels = np.zeros((node_number, 1))
        labels[data_train['Index']] = data_train["Label"]
        labels[data_test['Index']] = data_test["Label"]
        # feature matrix
        features_matrix = np.zeros((node_number, data_train["Attributes"].shape[1]))
        features_matrix[data_train['Index']] = data_train["Attributes"].toarray()
        features_matrix[data_test['Index']] = data_test["Attributes"].toarray()
        # adjacency matrix
        adjacency_matrix = sp.coo_matrix((np.ones(len(node1)), (node1, node2)), shape=(node_number, node_number))

        # all the classes in a list
        all_class_list: list = []
        for cls in labels:
            if cls[0] not in all_class_list:
                all_class_list.append(cls[0])

        # class_id -> [node_id, node_id, ...]
        class_dict: dict = {}
        for cls in class_dict:
            class_dict[cls] = []
        for node_id, class_id in labels:
            class_dict[class_id[0]].append(node_id)

        label_binarizer = preprocessing.LabelBinarizer()
        labels = label_binarizer.fit_transform(labels)
        features_matrix = torch.FloatTensor(features_matrix)
        labels = torch.LongTensor(np.where(labels)[1])
        adjacency_matrix = sparse_matrix2torch_sparse_tensor(
            normalize(adjacency_matrix + sp.eye(adjacency_matrix.shape[0])))
    '''
    elif dataset == 'cora-full':
        pass
    elif dataset == 'ogbn-arxiv':
        pass
    '''
    # store node id
    train_node_index: list = list()
    valid_node_index: list = list()
    test_node_index: list = list()
    for idx, class_list in zip([train_node_index, valid_node_index, test_node_index],
                               [train_class_list, valid_class_list, test_class_list]):
        for class_id in class_list:
            idx.append(class_dict[class_id])

    class_train_dict = defaultdict(list)
    for one in train_class_list:
        for i, label in enumerate(labels.numpy().tolist()):
            if label == one:
                class_train_dict[one].append(i)
    class_valid_dict = defaultdict(list)
    for one in valid_class_list:
        for i, label in enumerate(labels.numpy().tolist()):
            if label == one:
                class_valid_dict[one].append(i)

    class_test_dict = defaultdict(list)
    for one in test_class_list:
        for i, label in enumerate(labels.numpy().tolist()):
            if label == one:
                class_test_dict[one].append(i)

    graph = Graph(adjacency_matrix, features_matrix, labels,
                  train_node_index, valid_node_index, test_node_index,
                  node1, node2,
                  class_train_dict, class_valid_dict, class_test_dict)

    return graph


class Graph:
    """ class Graph is used to store the information of a graph,
        including adjacency_matrix, features_matrix, and etc.

    """

    def __init__(self, adjacency_matrix, features_matrix, labels,
                 train_node_index, valid_node_index, test_node_index,
                 node1, node2,
                 class_train_dict, class_valid_dict, class_test_dict):
        self.adjacency_matrix = adjacency_matrix
        self.features_matrix = features_matrix
        self.labels = labels

        self.train_node_index: list = train_node_index
        self.valid_node_index: list = valid_node_index
        self.test_node_index: list = test_node_index

        self.node1: list = node1
        self.node2: list = node2

        self.class_train_dict: defaultdict = class_train_dict
        self.class_valid_dict: defaultdict = class_valid_dict
        self.class_test_dict: defaultdict = class_test_dict
