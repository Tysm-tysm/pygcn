import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    # set() 函数创建一个无序不重复元素集
    classes = set(labels)
    '''enumerate()函数生成序列，带有索引i和值c。
        这一句将string类型的label变为int类型的label，建立映射关系
        np.identity(len(classes)) 为创建一个classes的单位矩阵
        创建一个字典，索引为 label， 值为独热码向量（就是之前生成的矩阵中的某一行）'''
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    # 为所有的标签生成相应的独热码
    # map() 会根据提供的函数对指定序列做映射。
    # 这一句将string类型的label替换为int类型的label
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))  # 从文本文件中读取数据并将其转换为NumPy数组，以二维数组存储[论文编号，内容，label]
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)  # 以稀疏矩阵（采用CSR格式压缩）将数据中的特征存储
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)

    # 上边的edges_unordered中存储的是端点id，要将每一项的id换成编号。
    # 在idx_map中以idx作为键查找得到对应节点的编号，reshape成与edges_unordered形状一样的数组。
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    # 首先要明确coo_matrix()的作用，该方法是构建一个矩阵，根据给出的下标、数据和形状，构建一个矩阵，其中下标位置的值对应数据中的值。使用方法见[5]。
    # 所以这一段的作用就是，网络有多少条边，邻接矩阵就有多少个1，所以先创建一个长度为edge_num的全1数组（data部分），每个1的填充位置就是一条边中两个端点的编号，
    # 即edges[:, 0], edges[:, 1]，矩阵的形状为(node_size, node_size)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    # 对于无向图，邻接矩阵是对称的。上一步得到的adj是按有向图构建的，转换成无向图的邻接矩阵需要扩充成对称矩阵。
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # 进行归一化，对应于论文中的A^=(D~)^0.5 A~ (D~)^0.5,但是本代码实现的是A^=(D~)^-1 A~
    #  A~=I+A
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix
        首先对每一行求和得到rowsum；
        求倒数得到r_inv；
        如果某一行全为0，则r_inv算出来会等于无穷大，将这些行的r_inv置为0；
        构建对角元素为r_inv的对角矩阵；
        用对角矩阵与原始矩阵的点积起到标准化的作用，原始矩阵中每一行元素都会与对应的r_inv相乘。"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    # 使用type_as(tesnor)将张量转换为给定类型的张量。
    preds = output.max(1)[1].type_as(labels)
    # 记录等于preds的label eq:equal
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    # 先将稀疏矩阵转换成coo类型表示，即（行索引，列索引，值），然后再将其数据类型转成float32
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    # 行索引和列索引垂直方向堆叠（行顺序，结果是两行n列），然后转成int64,然后将数组转换成张量
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
