import paddle

# def paddleeye(x, n):
#     tmp =x[0][paddle.eye(n).astype(paddle.bool)]
#     return tmp.unsqueeze_(0)

# 为了提速，没有使用该函数，而是用numpy语句实现，在反向传播方面可能有隐患。
def paddleindexjia (x, y, xindex):
    '''
    切片+索引，使用循环来解决切片问题，然后使用中间变量，来实现按照索引赋值
    支持类似的语句pos[:, group] -= offset.unsqueeze(1)
    '''
    xlen = len(x)
    assert len(x.shape) == 3 , "维度不一致,必须为3D数据"
#     if len(y.shape) == 3 and y.shape[0] ==1 :
#         y = paddle.squeeze(y)
    assert len(y.shape) ==2 , "维度不一致，必须为2D数据"
    for i in range(xlen):
        tmp = x[i]
        tmp[xindex] += y
        x[i] = tmp
    return x


class Wrapper:
    def __init__(self, natoms, bonds):
        self.groups, self.nongrouped = calculate_molecule_groups(natoms, bonds)
        # self.groups [22] self.nongrouped 688个[3]
#         print(f"==self.groups, self.nongrouped {self.groups, self.nongrouped}") 

    def wrap(self, pos, box, wrapidx=None):
        nmol = len(self.groups)
#         print(f"== box.sahpe {box.shape}")
#         box = box[:, paddle.eye(3).astype(paddle.bool)]  # Use only the diagonal
#         box = box[:][paddle.eye(3).astype(paddle.bool)]
#         box = box[paddle.eye(3).astype(paddle.bool)]
#         box = box.reshape([3, 3]) # 先试试这样的shape可以不？ 速度15
#         box = box* (paddle.eye(3).astype(paddle.bool))
#         print(f"== after eye box.sahpe {box.shape}")
#         box = box.reshape([-1, 3, 3])
#         box[0] = box[0] * (paddle.eye(3).astype(paddle.bool)) # 速度15 torch速度9 
#         box = paddleeye(box, 3)
        box = box*paddle.eye(3) # 可以很好的处理box[2, 3, 3]类型数据
        box = box.sum(1)
        if paddle.all(box == 0):
            return

        if wrapidx is not None:
            # Get COM of wrapping center group
#             com = paddle.sum(pos[:, wrapidx], axis=1) / len(wrapidx)
            com = paddle.sum(paddle.gather(pos, wrapidx, axis=1), axis=1) / len(wrapidx)
            # Subtract COM from all atoms so that the center mol is at [box/2, box/2, box/2]
            pos = (pos - com) + (box / 2)

        if nmol != 0:
            # Work out the COMs and offsets of every group and move group to [0, box] range
            for i, group in enumerate(self.groups):
#                 print(f"==i, group {i, group}")
#                 tmp_com = paddle.sum(pos[:, group], axis=1) / len(group)
                tmp_com = paddle.sum(paddle.gather(pos, group, axis=1), axis=1) / len(group)
                offset = paddle.floor(tmp_com / box) * box
#                 print(f"pos group offset {pos.shape, offset.shape}")
#                 pos[:, group] -= offset.unsqueeze(1)
#                 pos = paddleindexjia(pos, -offset, group)
                pos = pos.numpy()
                offset = offset.unsqueeze(1).numpy()
                pos[:, group] -= offset # 尝试使用numpy来处理 前后相关语句共4句
                pos = paddle.to_tensor(pos)

        # Move non-grouped atoms
        if len(self.nongrouped):
            offset = paddle.floor(pos[:, self.nongrouped] / box) * box
#             pos[:, self.nongrouped] -= offset.unsqueeze(1)
#             pos = paddleindexjia(pos, -offset, self.nongrouped)
            pos = pos.numpy()
            offset = offset.unsqueeze(1).numpy()
            pos[:, self.nongrouped] -= offset # 尝试使用numpy来处理 前后相关语句共4句
            pos = paddle.to_tensor(pos)


def calculate_molecule_groups(natoms, bonds):
    import networkx as nx
    import numpy as np

    # Calculate molecule groups and non-bonded / non-grouped atoms
    if bonds is not None and len(bonds):
        bondGraph = nx.Graph()
        bondGraph.add_nodes_from(range(natoms))
        bondGraph.add_edges_from(bonds.astype(np.int64))
        molgroups = list(nx.connected_components(bondGraph))

        nongrouped = paddle.to_tensor(
            [list(group)[0] for group in molgroups if len(group) == 1]
        ) 
        molgroups = [
            paddle.to_tensor(list(group)) 
            for group in molgroups
            if len(group) > 1
        ]
    else:
        molgroups = []
        nongrouped = paddle.arange(0, natoms) 
    return molgroups, nongrouped
