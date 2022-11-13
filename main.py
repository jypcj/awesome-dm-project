import argparse
import json
import random
from collections import defaultdict

import numpy as np
import torch
from torch import Tensor, optim
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from data import data_preprocess, Graph

from model import GNN4ClassLevel, GNN4NodeLevel, Linear
from utils import l2_normalize, accuracy, InforNCE_Loss

# the name of dataset
datasets = ['cora-full', 'Amazon_eletronics', 'dblp']
CORA_FULL = 'cora-full'
AMAZON_ELECTRONICS = 'Amazon_eletronics'
DBLP = 'dblp'


# the name of mode
TRAIN = 'train'
VALID = 'valid'
TEST = 'test'

args: argparse.Namespace = argparse.Namespace()

# K and N in experiment
# k nodes for each of n classes
Ks: list = [3, 5]
Ns: list = [5, 10]

query_size = 10

# repeat times for each (n, k)
repeat_times = 5

final_results = defaultdict(dict)

# loss function
loss_function = torch.nn.CrossEntropyLoss




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

    # store the final result
    # final_results: dict = defaultdict(dict)
    train_and_test()


def train_and_test():
    # conduct the experiment in each dataset
    for dataset in datasets:
        graph = data_preprocess(dataset)

        adjacency_matrix = graph.adjacency_matrix.to_dense()
        adjacency_matrix = adjacency_matrix.cuda()

        for n in Ns:
            for k in Ks:
                for repeat in range(repeat_times):
                    print("begin", dataset, "n= ", n, "k= ", k)

                    # two models in class-level and node level adaption
                    node_level_model: GNN4NodeLevel = GNN4NodeLevel(nfeat=graph.features_matrix.shape[1],
                                                                    nhid=args.hidden1,
                                                                    dropout=args.dropout)
                    class_level_model: GNN4ClassLevel = GNN4ClassLevel(nfeat=args.hidden1,
                                                                       nhid=args.hidden2,
                                                                       dropout=args.dropout)
                    support_labels: Tensor = torch.zeros(n * k, dtype=torch.long)
                    query_labels: Tensor = torch.zeros(n * k, dtype=torch.long)

                    for i in range(n):
                        support_labels[i * k:(i + 1) * k] = i
                    for i in range(n):
                        query_labels[i * query_size:(i + 1) * query_size] = i

                    classifier: Linear = Linear(args.hidden1, graph.labels.max().item() + 1)
                    optimizer: optim.Adam = optim.Adam(
                        [{'params': class_level_model.parameters()}, {'params': classifier.parameters()},
                         {'params': node_level_model.parameters()}],
                        lr=args.lr, weight_decay=args.weight_decay)

                    if args.use_cuda:
                        node_level_model = node_level_model.cuda()
                        class_level_model = class_level_model.cuda()

                        graph.features_matrix = graph.features_matrix.cuda()
                        graph.adjacency_matrix = graph.adjacency_matrix.cuda()
                        graph.labels = graph.labels.cuda()

                        classifier = classifier.cuda()

                        support_labels = support_labels.cuda()
                        query_labels = query_labels.cuda()

                    cnt: int = 0
                    valid_accuracy_best: float = 0.0
                    test_accuracy_best: list = []
                    for epoch in range(args.epochs):
                        train_accuracy: float = calculate_accuracy(TRAIN)

                        # epoch for test and valid
                        if epoch % 50 == 0 and epoch != 0:
                            tmp_accuracies: list = []
                            for test_epoch in range(50):
                                tmp_accuracies.append(calculate_accuracy(TEST))

                            valid_accuracies: list = []
                            for valid_epoch in range(50):
                                valid_accuracies.append(calculate_accuracy(VALID))

                            valid_accuracy = np.array(valid_accuracies).mean(axis=0)

                            print("Epoch: {:04d} Meta-valid_Accuracy: {:.4f}".format(epoch + 1, valid_accuracy))

                            if valid_accuracy > valid_accuracy_best:
                                valid_accuracy_best = valid_accuracy
                                test_accuracy_best = tmp_accuracies
                                cnt = 0
                            else:
                                cnt += 1
                                if cnt >= 10:
                                    break

                    print('Test Acc', np.array(test_accuracy_best).mean(axis=0))
                    final_results[dataset]['{}-way {}-shot {}-repeat'.format(n, k, repeat)] = [np.array(test_accuracy_best).mean(axis=0)]
                    json.dump(final_results[dataset], open('./TENT-result_{}.json'.format(dataset), 'w'))

                final_accuracies: list = []
                for i in range(repeat_times):
                    final_accuracies.append(final_results[dataset]['{}-way {}-shot {}-repeat'.format(n, k, i)][0])

                final_results[dataset]['{}-way {}-shot'.format(n, k)] = [np.mean(final_accuracies)]
                final_results[dataset]['{}-way {}-shot_print'.format(n, k)] = 'acc: {:.4f}'.format(np.mean(final_accuracies))

                json.dump(final_results[dataset], open('./TENT-result_{}.json'.format(dataset), 'w'))

                del node_level_model
                del class_level_model


def calculate_accuracy(class_level_model: GNN4ClassLevel, node_level_model: GNN4NodeLevel,
                       optimizer: optim.Adam,
                       classifier: Linear,
                       graph: Graph,
                       dataset: str,
                       epoch: int,
                       n: int, k: int,
                       mode: str) -> float:
    if mode == TRAIN:
        class_level_model.train()
        optimizer.zero_grad()
    else:
        class_level_model.eval()

    # first-step representations of nodes
    node_features = node_level_model(graph.features_matrix, graph.adjacency_matrix)

    class_dict: defaultdict = defaultdict()
    if mode == TRAIN:
        class_dict = graph.class_train_dict
    elif mode == TEST:
        class_dict = graph.class_test_dict
    elif mode == VALID:
        class_dict = graph.class_valid_dict

    chosen_class = np.random.choice(list(class_dict.keys()), n, replace=False)
    chosen_class = chosen_class.tolist()

    pos_node_index: list = []
    query_node_index: list = []
    pos_graph_adj_and_feat: list = []

    # construct class-ego subgraph for each class
    for cls in chosen_class:
        # sampled node id in this class
        sampled_node_index = np.random.choice(class_dict[cls], k + query_size, replace=False).tolist()
        pos_node_index.append(sampled_node_index[:k])
        query_node_index.append(sampled_node_index[k:])

        class_pos_index: list = sampled_node_index[:k]

        pos_graph_neighbors = torch.nonzero(graph.adjacency_matrix[class_pos_index, :].sum(0)).squeeze()
        pos_graph_adj = graph.adjacency_matrix[pos_graph_neighbors, :][:, pos_graph_neighbors]

        pos_class_graph_adj = torch.eye(pos_graph_neighbors.shape[0] + 1, dtype=torch.float)

        pos_class_graph_adj[1:, 1:] = pos_graph_adj

        pos_graph_feat = torch.cat(
            [node_features[class_pos_index].mean(0, keepdim=True), node_features[pos_graph_neighbors]], 0)

        pos_class_graph_adj = pos_class_graph_adj.cuda()

        pos_graph_adj_and_feat.append((pos_class_graph_adj, pos_graph_feat))

    target_graph_adj_and_feat = []
    for node in query_node_index:
        if torch.nonzero(graph.adjacency_matrix[node, :]).shape[0] == 1:
            pos_graph_adj = graph.adjacency_matrix[node, node].reshape([1, 1])
            pos_graph_feat = node_features[node].unsqueeze(0)
        else:
            pos_graph_neighbors = torch.nonzero(graph.adjacency_matrix[node, :]).squeeze()
            pos_graph_neighbors = torch.nonzero(graph.adjacency_matrix[pos_graph_neighbors, :].sum(0)).squeeze()
            pos_graph_adj = graph.adjacency_matrix[pos_graph_neighbors, :][:, pos_graph_neighbors]
            pos_graph_feat = node_features[pos_graph_neighbors]

        target_graph_adj_and_feat.append((pos_graph_adj, pos_graph_feat))

    class_generate_emb = torch.stack([sub[1][0] for sub in pos_graph_adj_and_feat], 0).mean(0)

    parameters = class_level_model.generater(class_generate_emb)

    gc1_parameters = parameters[:(args.hidden1 + 1) * args.hidden2 * 2]
    gc2_parameters = parameters[(args.hidden1 + 1) * args.hidden2 * 2:]

    gc1_w = gc1_parameters[:args.hidden1 * args.hidden2 * 2].reshape([2, args.hidden1, args.hidden2])
    gc1_b = gc1_parameters[args.hidden1 * args.hidden2 * 2:].reshape([2, args.hidden2])

    gc2_w = gc2_parameters[:args.hidden2 * args.hidden2 * 2].reshape([2, args.hidden2, args.hidden2])
    gc2_b = gc2_parameters[args.hidden2 * args.hidden2 * 2:].reshape([2, args.hidden2])

    class_level_model.eval()
    ori_emb = []
    for i, one in enumerate(target_graph_adj_and_feat):
        sub_adj, sub_feat = one[0], one[1]
        ori_emb.append(class_level_model(sub_feat, sub_adj, gc1_w, gc1_b, gc2_w, gc2_b).mean(0))  # .mean(0))

    target_embs = torch.stack(ori_emb, 0)

    class_ego_embs = []
    for sub_adj, sub_feat in pos_graph_adj_and_feat:
        class_ego_embs.append(class_level_model(sub_feat, sub_adj, gc1_w, gc1_b, gc2_w, gc2_b)[0])
    class_ego_embs = torch.stack(class_ego_embs, 0)

    target_embs = target_embs.reshape([n, query_size, -1]).transpose(0, 1)

    support_features = node_features[pos_node_index].reshape([n, k, -1])
    class_features = support_features.mean(1)
    taus = []
    for j in range(n):
        taus.append(torch.linalg.norm(support_features[j] - class_features[j], -1).sum(0))
    taus = torch.stack(taus, 0)

    similarities = []
    for j in range(query_size):
        class_contras_loss, similarity = InforNCE_Loss(target_embs[j], class_ego_embs / taus.unsqueeze(-1), dataset=dataset,tau=0.5)
        similarities.append(similarity)

    loss_supervised = loss_function(classifier(node_features[graph.train_node_index]), graph.labels[graph.train_node_index])

    loss = loss_supervised

    labels_train = graph.labels[query_node_index]
    for j, class_idx in enumerate(chosen_class[:n]):
        labels_train[labels_train == class_idx] = j

    loss += loss_function(torch.stack(similarities, 0).transpose(0, 1).reshape([n * query_size, -1]), labels_train)

    acc_train = accuracy(torch.stack(similarities, 0).transpose(0, 1).reshape([n * query_size, -1]), labels_train)

    if mode == 'valid' or mode == 'test' or (mode == 'train' and epoch % 250 == 249):
        support_features = l2_normalize(node_features[pos_node_index].detach().cpu()).numpy()
        query_features = l2_normalize(node_features[query_node_index].detach().cpu()).numpy()

        support_labels = torch.zeros(n * k, dtype=torch.long)
        for i in range(n):
            support_labels[i * k:(i + 1) * k] = i

        query_labels = torch.zeros(n * query_size, dtype=torch.long)
        for i in range(n):
            query_labels[i * query_size:(i + 1) * query_size] = i

        clf = LogisticRegression(penalty='l2',
                                 random_state=0,
                                 C=1.0,
                                 solver='lbfgs',
                                 max_iter=1000,
                                 multi_class='multinomial')
        clf.fit(support_features, support_labels.numpy())
        query_ys_pred = clf.predict(query_features)

        acc_train = metrics.accuracy_score(query_labels, query_ys_pred)

    if mode == 'train':
        loss.backward()
        optimizer.step()

    if epoch % 250 == 249 and mode == 'train':
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss.item()),
              'acc_train: {:.4f}'.format(acc_train.item()))
    return acc_train.item()

   # return 0.0


# entry of the program
if __name__ == '__main__':
    main()
