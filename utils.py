import numpy as np
import scipy.sparse as sp
import torch


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_matrix2torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor.

        Parameter:
            sparse_matrix: the scipy matrix to be conversed

        Return:
            torch sparse tensor
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def l2_normalize(x):
    norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
    out = x.div(norm)
    return out


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def InforNCE_Loss(anchor, sample, tau, dataset, all_negative=False, temperature_matrix=None):
    def _similarity(h1: torch.Tensor, h2: torch.Tensor):
        h1 = torch.nn.functional.normalize(h1)
        h2 = torch.nn.functional.normalize(h2)
        return h1 @ h2.t()

    assert anchor.shape[0] == sample.shape[0]

    pos_mask = torch.eye(anchor.shape[0], dtype=torch.float)

    pos_mask = pos_mask.cuda()

    neg_mask = 1. - pos_mask

    sim = _similarity(anchor, sample / temperature_matrix if temperature_matrix != None else sample) / tau
    exp_sim = torch.exp(sim) * (pos_mask + neg_mask)

    if not all_negative:
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
    else:
        log_prob = - torch.log(exp_sim.sum(dim=1, keepdim=True))

    loss = log_prob * pos_mask
    loss = loss.sum(dim=1) / pos_mask.sum(dim=1)

    return -loss.mean(), sim