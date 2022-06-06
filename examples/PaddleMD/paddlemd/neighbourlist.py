import paddle

# 写飞桨版本的笛卡尔直积函数cartesian_prod
from itertools import product
def paddlecartesian_prod(*x):
    z = list(product(*x))
    z = paddle.to_tensor(z)
    return z.squeeze(axis=-1)

def discretize_box(box, subcell_size):
    xbins = paddle.arange(0, box[0, 0] + subcell_size, subcell_size)
    ybins = paddle.arange(0, box[1, 1] + subcell_size, subcell_size)
    zbins = paddle.arange(0, box[2, 2] + subcell_size, subcell_size)
    nxbins = len(xbins) - 1
    nybins = len(ybins) - 1
    nzbins = len(zbins) - 1

    r = paddle.to_tensor([-1, 0, 1])
    neighbour_mask = paddlecartesian_prod(r, r, r)

    cellidx = paddlecartesian_prod(
        paddle.arange(nxbins), paddle.arange(nybins), paddle.arange(nzbins)
    )
    cellneighbours = cellidx.unsqueeze(2) + neighbour_mask.T.unsqueeze(0).repeat(
        cellidx.shape[0], 1, 1
    )

    # Can probably be done easier as we only need to handle -1 and max cases, not general -2, max+1 etc
    nbins = paddle.to_tensor([nxbins, nybins, nzbins])[None, :, None].repeat(
        cellidx.shape[0], 1, 27
    )
    negvals = cellneighbours < 0
    cellneighbours[negvals] += nbins[negvals]
    largevals = cellneighbours > (nbins - 1)
    cellneighbours[largevals] -= nbins[largevals]

    return xbins, ybins, zbins, cellneighbours


# def neighbour_list(pos, box, subcell_size):
#     nsystems = coordinates.shape[0]

#     for s in range(nsystems):
#         spos = pos[s]
#         sbox = box[s]

#         xbins, ybins, zbins = discretize_box(sbox, subcell_size)

#         xidx = paddle.bucketize(spos[:, 0], xbins, out_int32=True)
#         yidx = paddle.bucketize(spos[:, 1], ybins, out_int32=True)
#         zidx = paddle.bucketize(spos[:, 2], zbins, out_int32=True)

#         binidx = paddle.stack((xidx, yidx, zidx)).T
