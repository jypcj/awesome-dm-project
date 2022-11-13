import torch.nn
import math
from torch.nn.parameter import Parameter
import numpy as np


# node-level adaption model
class GraphNeuralNode(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphNeuralNode, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight.size(1))

        self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


# used in node-level adaption
class GNN4NodeLevel(torch.nn.Module):
    # 构建模型
    def __init__(self, nfeat, nhid, dropout):
        super(GNN4NodeLevel, self).__init__()

        self.gc1 = GraphNeuralNode(nfeat, nhid)
        self.gc2 = GraphNeuralNode(nhid, nhid)
        self.dropout = dropout

    # 前向传播
    def forward(self, x, adj):
        return self.gc1(x, adj)


# node-level adaption model
class GraphNeuralClass(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphNeuralClass, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))

        self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, w, b):
        if w is None and b is None:
            alpha_w = alpha_b = beta_w = beta_b = 0
        else:
            alpha_w = w[0]
            beta_w = w[1]
            alpha_b = b[0]
            beta_b = b[1]

        support = torch.mm(input, self.weight * (1 + alpha_w) + beta_w)  # formula (6)
        output = torch.mm(adj, support)

        if self.bias is not None:
            return output + self.bias * (1 + alpha_b) + beta_b
        else:
            return output


# used in node-level adaption
class GNN4ClassLevel(torch.nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GNN4ClassLevel, self).__init__()

        self.gc1 = GraphNeuralClass(nfeat, nhid)
        self.gc2 = GraphNeuralClass(nhid, nhid)
        self.generater = torch.nn.Linear(nfeat, (nfeat + 1) * nhid * 2 + (nhid + 1) * nhid * 2)

        self.dropout = dropout

    def permute(self, input_adj, input_feat, drop_rate=0.1):
        # return input_adj

        adj_random = torch.rand(input_adj.shape).cuda() + torch.eye(input_adj.shape[0]).cuda()

        feat_random = np.random.choice(input_feat.shape[0], int(input_feat.shape[0] * drop_rate),
                                       replace=False).tolist()

        masks = torch.zeros(input_feat.shape).cuda()
        masks[feat_random] = 1

        random_tensor = torch.rand(input_feat.shape).cuda()

        return input_adj * (adj_random > drop_rate), input_feat * (1 - masks) + random_tensor * masks

    def forward(self, x, adj, w1=None, b1=None, w2=None, b2=None):
        x = torch.nn.functional.relu(self.gc1(x, adj, w1, b1))
        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj, w2, b2)
        return x


# used in classifier
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))

        self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, w=None, b=None):
        if w is not None:
            return torch.mm(input, w) + b
        else:
            return torch.mm(input, self.weight) + self.bias
