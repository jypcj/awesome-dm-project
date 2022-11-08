import json
import scipy.io as sio
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
    train_class_list, valid_class_list, test_class_list = json.load(open('./dataset/{}_class_split.json'.format(dataset)))
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
        class_list: list = []
        for cls in labels:
            if cls[0] not in class_list:
                class_list.append(cls[0])
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
        adjacency_matrix = sparse_matrix2torch_sparse_tensor(normalize(adjacency_matrix + sp.eye(adjacency_matrix.shape[0])))
    elif dataset == 'cora-full':
        pass
    elif dataset == 'ogbn-arxiv':
        pass


if __name__ == '__main__':
    data_preprocess("Amazon_eletronics")