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
from ema import EMA

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


# K and N in experiment
# k nodes for each of n classes
Ks: list = [3, 5]
Ns: list = [5, 10]

query_size = 10

# repeat times for each (n, k)
repeat_times = 5

final_results = defaultdict(dict)

# loss function
loss_function = torch.nn.CrossEntropyLoss()

# parameters
class args:
    use_cuda = True
    seed = 1234
    epochs = 2000
    test_epochs = 100
    lr = 0.05
    weight_decay = 5e-4
    hidden1 = 16
    hidden2 = 16
    dropout = 0.2


def main():
    """main function of the whole project
    """

    # parameter

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)

    train_and_test()


def train_and_test():
    # conduct the experiment in each dataset
    for dataset in datasets:
        graph = data_preprocess(dataset)

        adj_dense = graph.adjacency_matrix.to_dense()
        adj_dense = adj_dense.cuda()

        for n in Ns:
            for k in Ks:
                for repeat in range(repeat_times):
                    print("begin ", dataset, "n= ", n, "k= ", k)

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
                        [{'params': class_level_model.parameters()},
                         {'params': classifier.parameters()},
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

                        # ema
                        ema1 = EMA(node_level_model, 0.999)
                        ema1.register()
                        ema2 = EMA(class_level_model, 0.999)
                        ema2.register()

                    def calculate_accuracy(epoch: int,
                                           n: int, k: int,
                                           mode: str) -> float:
                        if mode == 'train':
                            class_level_model.train()
                            optimizer.zero_grad()
                        else:
                            class_level_model.eval()

                            # first-step node representation?
                        emb_features = node_level_model(graph.features_matrix, graph.adjacency_matrix)

                        target_idx = []
                        target_graph_adj_and_feat = []
                        support_graph_adj_and_feat = []

                        pos_node_idx = []

                        if mode == 'train':
                            class_dict = graph.class_train_dict
                        elif mode == 'test':
                            class_dict = graph.class_test_dict
                        elif mode == 'valid':
                            class_dict = graph.class_valid_dict

                        K = k
                        N = n
                        Q = query_size

                        classes = np.random.choice(list(class_dict.keys()), N, replace=False).tolist()

                        pos_graph_adj_and_feat = []
                        # construct class-ego subgraphs?
                        for i in classes:
                            # sample from one specific class
                            sampled_idx = np.random.choice(class_dict[i], K + Q, replace=False).tolist()
                            pos_node_idx.extend(sampled_idx[:K])
                            target_idx.extend(sampled_idx[K:])

                            class_pos_idx = sampled_idx[:K]

                            # why k = 1?
                            if K == 1 and torch.nonzero(adj_dense[class_pos_idx, :]).shape[0] == 1:
                                pos_class_graph_adj = adj_dense[class_pos_idx, class_pos_idx].reshape([1, 1])
                                pos_graph_feat = emb_features[class_pos_idx]
                            else:
                                pos_graph_neighbors = torch.nonzero(adj_dense[class_pos_idx, :].sum(0)).squeeze()

                                pos_graph_adj = adj_dense[pos_graph_neighbors, :][:, pos_graph_neighbors]

                                pos_class_graph_adj = torch.eye(pos_graph_neighbors.shape[0] + 1, dtype=torch.float)

                                pos_class_graph_adj[1:, 1:] = pos_graph_adj

                                pos_graph_feat = torch.cat([emb_features[class_pos_idx].mean(0, keepdim=True),
                                                            emb_features[pos_graph_neighbors]], 0)

                            if dataset != 'ogbn-arxiv':
                                pos_class_graph_adj = pos_class_graph_adj.cuda()

                            pos_graph_adj_and_feat.append((pos_class_graph_adj, pos_graph_feat))

                        target_graph_adj_and_feat = []
                        for node in target_idx:
                            if torch.nonzero(adj_dense[node, :]).shape[0] == 1:
                                pos_graph_adj = adj_dense[node, node].reshape([1, 1])
                                pos_graph_feat = emb_features[node].unsqueeze(0)
                            else:
                                pos_graph_neighbors = torch.nonzero(adj_dense[node, :]).squeeze()
                                pos_graph_neighbors = torch.nonzero(adj_dense[pos_graph_neighbors, :].sum(0)).squeeze()
                                pos_graph_adj = adj_dense[pos_graph_neighbors, :][:, pos_graph_neighbors]
                                pos_graph_feat = emb_features[pos_graph_neighbors]

                            target_graph_adj_and_feat.append((pos_graph_adj, pos_graph_feat))

                        class_generate_emb = torch.stack([sub[1][0] for sub in pos_graph_adj_and_feat], 0).mean(0)

                        parameters = class_level_model.generater(class_generate_emb)

                        gc1_parameters = parameters[:(args.hidden1 + 1) * args.hidden2 * 2]
                        gc2_parameters = parameters[(args.hidden1 + 1) * args.hidden2 * 2:]

                        gc1_w = gc1_parameters[:args.hidden1 * args.hidden2 * 2].reshape(
                            [2, args.hidden1, args.hidden2])
                        gc1_b = gc1_parameters[args.hidden1 * args.hidden2 * 2:].reshape([2, args.hidden2])

                        gc2_w = gc2_parameters[:args.hidden2 * args.hidden2 * 2].reshape(
                            [2, args.hidden2, args.hidden2])
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

                        target_embs = target_embs.reshape([N, Q, -1]).transpose(0, 1)

                        support_features = emb_features[pos_node_idx].reshape([N, K, -1])
                        class_features = support_features.mean(1)
                        taus = []
                        for j in range(N):
                            taus.append(torch.linalg.norm(support_features[j] - class_features[j], -1).sum(0))
                        taus = torch.stack(taus, 0)

                        similarities = []
                        for j in range(Q):
                            class_contras_loss, similarity = InforNCE_Loss(target_embs[j],
                                                                           class_ego_embs / taus.unsqueeze(-1), tau=0.5,dataset=dataset)
                            similarities.append(similarity)

                        loss_supervised = loss_function(classifier(emb_features[graph.train_node_index]), graph.labels[graph.train_node_index])

                        loss = loss_supervised

                        labels_train = graph.labels[target_idx]
                        for j, class_idx in enumerate(classes[:N]):
                            labels_train[labels_train == class_idx] = j

                        loss += loss_function(torch.stack(similarities, 0).transpose(0, 1).reshape([N * Q, -1]), labels_train)

                        acc_train = accuracy(torch.stack(similarities, 0).transpose(0, 1).reshape([N * Q, -1]),
                                             labels_train)

                        if mode == 'valid' or mode == 'test' or (mode == 'train' and epoch % 250 == 249):
                            # ema
                            ema1.apply_shadow()
                            ema2.apply_shadow()
                            support_features = l2_normalize(emb_features[pos_node_idx].detach().cpu()).numpy()
                            query_features = l2_normalize(emb_features[target_idx].detach().cpu()).numpy()

                            support_labels = torch.zeros(N * K, dtype=torch.long)
                            for i in range(N):
                                support_labels[i * K:(i + 1) * K] = i

                            query_labels = torch.zeros(N * Q, dtype=torch.long)
                            for i in range(N):
                                query_labels[i * Q:(i + 1) * Q] = i

                            clf = LogisticRegression(penalty='l2',
                                                     random_state=0,
                                                     C=1.0,
                                                     solver='lbfgs',
                                                     max_iter=1000,
                                                     multi_class='multinomial')
                            clf.fit(support_features, support_labels.numpy())
                            query_ys_pred = clf.predict(query_features)

                            acc_train = metrics.accuracy_score(query_labels, query_ys_pred)
                            # ema
                            ema1.restore()
                            ema2.restore()

                        if mode == 'train':
                            loss.backward()
                            optimizer.step()

                            # ema
                            ema1.update()
                            ema2.update()

                        if epoch % 250 == 249 and mode == 'train':
                            print('Epoch: {:04d}'.format(epoch + 1),
                                  'loss_train: {:.4f}'.format(loss.item()),
                                  'acc_train: {:.4f}'.format(acc_train.item()))
                        return acc_train.item()

                    # begin to train and test
                    cnt: int = 0
                    valid_accuracy_best: float = 0.0
                    test_accuracy_best: list = []
                    for epoch in range(args.epochs):
                        train_accuracy: float = calculate_accuracy(
                                                                   epoch=epoch,
                                                                   n=n, k=k,
                                                                   mode=TRAIN)

                        # epoch for test and valid
                        if epoch % 50 == 0 and epoch != 0:
                            tmp_accuracies: list = []
                            for test_epoch in range(50):
                                tmp_accuracy = calculate_accuracy(
                                                                  epoch=test_epoch,
                                                                  n=n, k=k,
                                                                  mode=TEST)
                                tmp_accuracies.append(tmp_accuracy)

                            valid_accuracies: list = []
                            for valid_epoch in range(50):
                                tmp_accuracy = calculate_accuracy(
                                                                  epoch=valid_epoch,
                                                                  n=n, k=k,
                                                                  mode=VALID)
                                valid_accuracies.append(tmp_accuracy)

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
                    final_results[dataset]['{}-way {}-shot {}-repeat'.format(n, k, repeat)] = [
                        np.array(test_accuracy_best).mean(axis=0)]
                    json.dump(final_results[dataset], open('./TENT-result_{}.json'.format(dataset), 'w'))

                final_accuracies: list = []
                for i in range(repeat_times):
                    final_accuracies.append(final_results[dataset]['{}-way {}-shot {}-repeat'.format(n, k, i)][0])

                final_results[dataset]['{}-way {}-shot'.format(n, k)] = [np.mean(final_accuracies)]
                final_results[dataset]['{}-way {}-shot_print'.format(n, k)] = 'acc: {:.4f}'.format(
                    np.mean(final_accuracies))

                json.dump(final_results[dataset], open('./TENT-result_{}.json'.format(dataset), 'w'))

                del node_level_model
                del class_level_model

        del graph
        del adj_dense





# return 0.0


# entry of the program
if __name__ == '__main__':
    main()
