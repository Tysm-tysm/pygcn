import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        '''
        卷积层1：输入的特征为nfeat，维度是2708，输出的特征为nhid，维度是16；
        卷积层2：输入的特征为nhid，维度是16，输出的特征为nclass，维度是7（即类别的结果）
        '''
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        """
        relu–>fropout–>gc2–>softmax
        """
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
