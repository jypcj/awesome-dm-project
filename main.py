import argparse
import random
from collections import defaultdict

import numpy as np
import torch
from data import data_preprocess

from model import GCN4ClassLevel, GCN4NodeLevel

# the name of dataset
datasets = ['cora-full','Amazon_eletronics','dblp','ogbn-arxiv']
CORA_FULL = 'cora-full'
AMAZON_ELECTRONICS = 'Amazon_eletronics'
DBLP = 'dblp'
OGBN_ARXIV = 'ogbn-arxiv'

args: argparse.Namespace = argparse.Namespace()

# K and N in experiment
Ks: list = [3, 5]
Ns: list = [5, 10]

# repeat times for each (n, k)
repeat_times = 5

def main():
    """main function of the whole project
    """

    # parameter
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=True, help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Number of epochs to train.')
    parser.add_argument('--test_epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Initial learning rate.')

    parser.add_argument('--weight_decay', type=float, default=5e-4,  # 5e-4
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden1', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--hidden2', type=int, default=16,
                        help='Number of hidden units.')
    # dropout rate
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate (1 - keep probability).')

    args = parser.parse_args(args=[])

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)

    # cross-entropy classification loss
    cross_entropy_loss = torch.nn.CrossEntropyLoss()

    # query size
    query_size = 10

    # store the final result
    final_results: dict = defaultdict(dict)
    train_and_test()


def train_and_test():
    # conduct the experiment in each dataset
    for dataset in datasets:
        graph = data_preprocess(dataset)

        adjacency_matrix = graph.adjacency_matrix.to_dense()
        if dataset != OGBN_ARXIV:
            adjacency_matrix = adjacency_matrix.cuda()
        else:
            args.use_cuda = False

        for n in Ns:
            for k in Ks:
                for repeat in range(repeat_times):
                    print("begin", dataset, "n= ", n, "k= ", k)

                    class_level_model: GCN4ClassLevel = GCN4ClassLevel(nfeat=args.hidden1,
                                                                       nhid=args.hidden2,
                                                                       dropout=args.dropout)
                    node_level_model: GCN4NodeLevel = GCN4NodeLevel(nfeat=graph.features_matrix.shape[1],
                                                                    nhid=args.hidden1,
                                                                    nclass=graph.labels.max().item() + 1,
                                                                    dropout=args.dropout)






# entry of the program
if __name__ == '__main__':
    main()

