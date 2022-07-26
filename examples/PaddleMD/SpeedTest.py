#!/usr/bin/env python
# coding: utf-8

# # 飞桨提速
# 针对飞桨速度慢的问题，对比常见的函数，发现瓶颈代码，修改以提升运行速度
# 
# 所有的函数运行测试。
# 发现只要带上for循环的，速度就要慢很多，大约30倍左右。
# 2022.5.12 今天把速度提升了10倍，离torch还差20倍左右。但是2号文件提速成功，1号文件提速没有成功，还没找到代码差异在哪里。 
# 
# 

# In[1]:


import time
class Timer:  #@save
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()


class Benchmark:
    """用于测量运行时间"""
    def __init__(self, description='Done'):
        self.description = description

    def __enter__(self):
        self.timer = Timer()
        return self

    def __exit__(self, *args):
        print(f'{self.description}: {self.timer.stop():.4f} sec')

# In[2]:


import paddle
import torch
import time
import time
class Timer:  #@save
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()


class Benchmark:
    """用于测量运行时间"""
    def __init__(self, description='Done'):
        self.description = description

    def __enter__(self):
        self.timer = Timer()
        return self

    def __exit__(self, *args):
        print(f'{self.description}: {self.timer.stop():.4f} sec')

def paddlerandn_like(x) : # 添加飞桨的randn_like函数
    '''输出x维度的随机tensor'''
    return paddle.randn(x.shape)

from math import pi
# 发现飞桨支持atan2函数，且自己写的只适合1D数据
# def paddleatan2(input, other): # 飞桨的atan2函数
#     atan = paddle.atan(input/other)
#     atan[1] = atan[1] + pi
#     atan[2] = atan[2] + pi
#     return atan

def paddlescatter(x, dim, index, src): # scatter支持1D版本
    
    updates = src
    if len(index.shape) == 1 :
#         for i in index:
#             x[i] += updates[i]
        
        for i in range(index.shape[0]):
            x[index[i]] += updates[i]
        
        return x
                                
    i, j = index.shape
    grid_x , grid_y = paddle.meshgrid(paddle.arange(i), paddle.arange(j))
    if dim == 0 :
        index = paddle.stack([index.flatten(), grid_y.flatten()], axis=1)
    elif dim == 1:
        index = paddle.stack([grid_x.flatten(), index.flatten()], axis=1)
        
    # PaddlePaddle updates 的 shape 大小必须与 index 对应
    updates_index = paddle.stack([grid_x.flatten(), grid_y.flatten()], axis=1)
    updates = paddle.gather_nd(updates, index=updates_index)
    return paddle.scatter_nd_add(x, index, updates)

def paddleindex_add(x, dim, index, source): # 飞桨的index_add
    '''
x = paddle.ones([5, 3])
t = paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=paddle.float32)
index = paddle.to_tensor([0, 4, 2])
# print(x)
with Benchmark("paddleindex_add"):
    x = paddleindex_add(x, 0, index, t)
print(x)
    '''
    for i in range(len(index)):
        x[index[i]] += source[i]
    return x

def paddleeye(x, n): # 针对[1, 3, 3]输入的特供eye函数
    tmp =x[0][paddle.eye(n).astype(paddle.bool)]
    return tmp.unsqueeze_(0)

def paddleindexjia (x, y, xindex): # 索引/切片/赋值特供版本
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

# 写飞桨版本的笛卡尔直积函数cartesian_prod
from itertools import product
def paddlecartesian_prod(x,y): # 飞桨版本的笛卡尔直积函数
    z = list(product(x,y))
    z = paddle.to_tensor(z)
    return z.squeeze(axis=-1)

def paddlecartesian_prod(arrays, out=None): # 飞桨版本的笛卡尔直积函数cartesian_prod
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    # print(arrays)
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    #m = n / arrays[0].size
    m = int(n / arrays[0].size) 
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
    for j in range(1, arrays[0].size):
    #for j in xrange(1, arrays[0].size):
        out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out

# In[3]:


import torch
import paddle
import numpy as np 

npa = np.ndarray([3,3])
ta = torch.Tensor(npa)
pa = paddle.to_tensor(npa)
with Benchmark("torch.randn_like"):
    ta = torch.randn_like(ta)
#     torch.cuda.synchronize
with Benchmark("paddlerandn_like"):
    pa = paddlerandn_like(pa)
with Benchmark("torch.randn_like"):
    ta = torch.randn_like(ta)
print(ta,ta.shape, pa)

# In[4]:


import math
pi = math.pi
real = torch.tensor([ 0.12,  -1.1, -0.1, 1])
imag = torch.tensor([ 0.22,  1.2, -1.2, -1.2])
atan = torch.atan(imag/real)
print(atan)
atan[1] = atan[1] + pi
atan[2] = atan[2] - pi
print(atan)
with Benchmark("torch.atan2 time"):
    atan2 = torch.atan2(imag,real)
print(atan2)

# In[5]:


import paddle
import math

pi = math.pi
real = paddle.to_tensor([ 0.12,  -1.1, -0.1, 1])
imag = paddle.to_tensor([ 0.22,  1.2, -1.2, -1.2])
atan = paddle.atan(imag/real)
print(atan)
atan[1] = atan[1] + pi
atan[2] = atan[2] - pi
print(atan)
# atan2 = torch.atan2(imag,real)
# print(atan2)

import paddle
import math
def paddleatan2(input, other):
    atan = paddle.atan(input/other)
    atan[1] = atan[1] + pi
    atan[2] = atan[2] + pi
    return atan
with Benchmark("paddleatan2 time"):
    print(paddleatan2(imag, real ))

# In[6]:


with Benchmark("paddle.atan2 time"):
    paddle.atan2(imag,real)

# In[7]:


x = paddle.randn([2, 3, 3])
y = paddle.randn([2, 3, 4])
with Benchmark("paddle.atan2 time"):
    paddle.atan2(x, y)

# In[8]:


# 测试paddlescatter
# pot = paddlescatter(x=pot, dim=0, index=idx, src=k0 * (1 + paddle.cos(angleDiff))) # x, dim, index, src
x = paddle.zeros([3, 5], dtype="int64")
updates = paddle.arange(1, 11).reshape([2,5])
# 输出
# Tensor(shape=[2, 5], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
#        [[1 , 2 , 3 , 4 , 5 ],
#         [6 , 7 , 8 , 9 , 10]])
index = paddle.to_tensor([[0, 1, 2], [0, 1, 4]])

tmp = paddlescatter(x=x, dim=1, index=index, src=updates)
print(tmp)

# In[9]:


import paddle
src = paddle.ones([5])
src = paddle.to_tensor([1, 2, 3, 4, 5])
index = paddle.to_tensor([0, 2, 0, 1, 4])
index = paddle.to_tensor([0, 0, 0, 0, 1])
print(index.shape)
tmp = paddle.zeros([10], dtype=src.dtype)
with Benchmark("paddlescatter"):
    tmp = paddlescatter(tmp, 0, index, src)
# tmp = paddle.scatter(tmp, index, src)
print(tmp)

# In[10]:


import torch
src = torch.ones([5])
src = torch.tensor([1, 2, 3, 4, 5])
# index = torch.tensor([0, 1, 2, 0, 0])
index = torch.tensor([0, 2, 0, 1, 4])
index = torch.tensor([0, 0, 0, 0, 1])
print(index.shape)
with Benchmark("torch.scatter_add_"):
    tmp = torch.zeros(10, dtype=src.dtype).scatter_add_(0, index, src)
print(tmp)
with Benchmark("torch.scatter_add"):
    tmp = torch.scatter_add(torch.zeros(10, dtype=src.dtype), 0, index, src)
print(tmp)

# In[11]:


x = torch.ones(5, 3)
t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
index = torch.tensor([0, 4, 2])
print(len(index))
print(x)
with Benchmark("torch.index_add_"):
    x.index_add_(0, index, t)
print(x)
# x.index_add_(0, index, t, alpha=-1)

x = paddle.ones([5, 3])
t = paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=paddle.float32)
index = paddle.to_tensor([0, 4, 2])
# print(len(index))
# print(x)
with Benchmark("paddleindex_add"):
    x = paddleindex_add(x, 0, index, t)
print(x)

# In[12]:


import torch
x = torch.ones(3, 5)
t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
index = torch.tensor([0, 4, 2])
print(len(index))
print(x)
with Benchmark("torch.index_add_"):
    x.index_add_(1, index, t)
print(x)
# x.index_add_(0, index, t, alpha=-1)print(x)

# In[32]:


x = torch.ones(688, 3)
# t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
t = torch.randn(687, 3)
# index = torch.tensor([0, 4, 2])
index = torch.randn(687).type(torch.int64)+5
print(index[:10])
index = torch.Tensor(np.arange(687)).type(torch.int64)

x.index_add_(0, index, t)

x = torch.ones(688,3)
t = torch.randn
# x.index_add_?
# x.index_add_(0, index, t, alpha=-1)

# In[35]:


import paddle

# x = paddle.to_tensor([[10, 30, 20], [60, 40, 50]])
# index = paddle.to_tensor([[0]])
# value = 99
# value = paddle.to_tensor([100,100,200])
# axis = 0
# result = paddle.put_along_axis(x, index, value, axis, reduce="add")
# print(result)

x = paddle.ones([688, 3])
# value = paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=paddle.float32)
value = paddle.randn([688, 3])
# index = paddle.randn([687]).astype(paddle.int64) +5
index = paddle.to_tensor(np.arange(687), paddle.int64)
# index = paddle.to_tensor([0, 4, 2])
axis = 0
# print(len(index))
# print(x)
with Benchmark("paddleindex_add"):
    x = paddle.put_along_axis(x, index, value, axis, reduce="add")
print(x.shape)
# [[99, 99, 99],
# [60, 40, 50]]

# In[12]:


# torch eye 看看
import torch 
box = torch.ones([1, 3, 3])
# box = box +box
with Benchmark("box[:, torch.eye(3).bool()]"):
    box = box[:, torch.eye(3).bool()]
# box = box[:][torch.eye(3).bool()]
print(box, box.shape)
import paddle
x = paddle.ones([1, 3, 3])
with Benchmark("paddleeye(x,3)"):
    tmp = paddleeye(x,3)
print(tmp)

# In[10]:


import torch
x = torch.ones([1,4,3])
xindex = [0,1,2]
y = torch.ones([1,3])
uy = y.unsqueeze(1)
print(uy.shape)
# uy = paddle.squeeze(y)
# tmp = paddleindexjia (x, uy, xindex)
# tmp = paddleindexjia(x, paddle.squeeze(y), xindex)
# x[:, xindex] -= y.unsqueeze(1)
with Benchmark("torch x[:, xindex] -= uy"):
    x[:, xindex] -= uy
print(x.shape, x)

x = paddle.ones([1,4,3])
xindex = [0,1,2]
y = paddle.ones([1,3])
uy = paddle.squeeze(y)
print(f"paddle uy shape:{uy.shape}")
# tmp = paddleindexjia (x, uy, xindex)
with Benchmark("paddleindexjia"):
    tmp = paddleindexjia(x, -y, xindex)
print(tmp)
y.numpy()
import numpy as np
nx = np.ndarray([1, 4, 3])
xindex = [0, 1, 2]
y = torch.ones([1,3])
uy = y.unsqueeze(1)
uy = uy.numpy()
x = paddle.ones([1,4,3])
nx = x.numpy()
with Benchmark("numpy x[:, xindex] -= uy"):
    nx[:, xindex] -= uy
print(nx)

# # 笛卡尔积

# In[14]:


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    # print(arrays)
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    #m = n / arrays[0].size
    m = int(n / arrays[0].size) 
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
    for j in range(1, arrays[0].size):
    #for j in xrange(1, arrays[0].size):
        out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out

# In[15]:


cartesian(([1, 2, 3], [4, 5], [6, 7]))
r = [-1, 0, 1]
r = paddle.to_tensor(r)
with Benchmark("paddleindexjia"):
    cartesian([r, r, r])

# In[16]:


from itertools import product
def paddlecartesian_prod(*x): # 飞桨版本的笛卡尔直积函数
    z = list(product(x,y))
    z = paddle.to_tensor(z)
    return z.squeeze(axis=-1)
r = paddle.to_tensor([-1, 0, 1])
# neighbour_mask = paddlecartesian_prod(r, r, r)

r = torch.Tensor([-1, 0, 1])
neighbour_mask = torch.cartesian_prod(r, r, r)
print(neighbour_mask)

# In[3]:


import torch
x = torch.ones(5, 3)
t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
index = torch.tensor([0, 4, 2])
x.index_add_(0, index, t)
# x.index_add_(0, index, t, alpha=-1)

# In[4]:


def paddleindex_add(x, dim, index, source): # 飞桨的index_add
    # return x # 测试速度
    for i in range(len(index)):
        x[index[i]] += source[i]
    return x

# In[5]:


import paddle

x = paddle.to_tensor([[10, 30, 20], [60, 40, 50]])
index = paddle.to_tensor([[0]])
value = 99
value = paddle.to_tensor([100,100,200])
axis = 0
result = paddle.put_along_axis(x, index, value, axis, reduce="add")
print(result)
# [[99, 99, 99],
# [60, 40, 50]]


# In[1]:


# 看issue里面的报错，经验证没有发现问题
import paddle

# 当label，mask仅有一个元素时，Paddle2.2会报错，现在的Paddle2.3解决了这个问题，不会报错了
label = paddle.to_tensor([0])
mask = paddle.to_tensor([True], dtype='bool')
print(label[mask])


# 但是有多个元素时，Paddle2.3怎么又不支持tensor索引了？索引的结果是错的！！
label = paddle.to_tensor([0, 8, 9])
mask = paddle.to_tensor([False, True, True], dtype='bool')
print(label[mask])  # Paddle2.3运行后输出[0, 0]，在Paddle2.2下运行是[8, 9]正常的
print(label[[False, True, True]])  # Paddle2.3下这样运行是正常的输出[8, 9]，总之变成Tensor就不对了

# In[6]:


# 看issue里面的花样索引 https://github.com/PaddlePaddle/Paddle/issues/42554
# 解决方法就是把多索引改成单索引的组合
import paddle
import numpy as np

bbox_annotation = [[131.27171326, 413.02932739, 163.76470947, 482.08468628, -24.44395447, 11.],
                   [166.36415100, 406.51464844, 187.15966797, 465.14657593, -25.40771866, 11.],
                   [191.05882263, 388.27362061, 217.05322266, 465.14657593, -24.84239006, 11.],
                   [230.05041504, 392.18240356, 256.04483032, 457.32897949, -23.96249008, 11.],
                   [257.34454346, 368.72964478, 282.03921509, 442.99673462, -26.11391258, 11.],]
bbox_annotation = paddle.to_tensor(bbox_annotation)
# bbox_annotation = np.array(bbox_annotation)  # 必须转化为numpy格式才行，但是这样，在计算loss的时候会丢失梯度
# bbox_annotation[[0, 1, 2, 3], :]
print (bbox_annotation.shape)
tmp = bbox_annotation[[0,1,2,3]]
tmp
### 其他补充信息 Additional Supplementary Information



# ## paddlegather

# In[37]:


# paddlegather https://gitee.com/paddlepaddle/Paddle/issues/I4QBTN
# issue里面提供的例子，跑不通
def paddle_gather(x, index):
    print(x.shape)
    m, n, k = paddle.split(paddle.shape(x), 3)
    print(m, n, k)
    idx_mk = paddle.arange(end=m * k)
    idx_m = idx_mk / k
    idx_k = idx_mk % k
    idx_n = paddle.gather(index, idx_m)
    idx = idx_m * n * k + idx_n * k + idx_k
    x_flatten = paddle.flatten(x)
    y_flatten = paddle.gather(x_flatten, idx)
    ret = paddle.reshape(y_flatten, [m, 1, k])
    return ret


t = paddle.to_tensor([[1, 2], [3, 4], [0,0]])
t = paddle.randn([2,3,4])
# print(t.shape)
paddle_gather(t, paddle.to_tensor([[0, 0], [1, 0]]))

# In[38]:


# 飞桨组合实现
def paddle_gather(x, dim, index):
    index_shape = index.shape
    index_flatten = index.flatten()
    if dim < 0:
        dim = len(x.shape) + dim
    nd_index = []
    for k in range(len(x.shape)):
        if k == dim:
            nd_index.append(index_flatten)
        else:
            reshape_shape = [1] * len(x.shape)
            reshape_shape[k] = x.shape[k]
            x_arange = paddle.arange(x.shape[k], dtype=index.dtype)
            x_arange = x_arange.reshape(reshape_shape)
            dim_index = paddle.expand(x_arange, index_shape).flatten()
            nd_index.append(dim_index)
    ind2 = paddle.transpose(paddle.stack(nd_index), [1, 0]).astype("int64")
    paddle_out = paddle.gather_nd(x, ind2).reshape(index_shape)
    return paddle_out

t = paddle.to_tensor([[1, 2], [3, 4]])
with Benchmark("paddle_gather"):
    paddle_gather(t, 1, paddle.to_tensor([[0, 0], [1, 0]]))
# 输出
# Tensor(shape=[2, 2], dtype=int32, place=CPUPlace, stop_gradient=True,
#        [[1, 1],
#         [4, 3]])

# In[ ]:




# In[39]:


# PyTorch示例：
import torch
t = torch.tensor([[1, 2], [3, 4]])
with Benchmark("torch.gather"):
    torch.gather(t, 1, torch.tensor([[0, 0], [1, 0]]))
# 输出
# tensor([[ 1,  1],
#         [ 4,  3]])

# ## index_select
# 感觉这个弄好，应该可以大有可为。
# 但是这个index只能1D吧？ 后来明白，index就是1D的。

# In[49]:


import paddle

x = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0],
                      [5.0, 6.0, 7.0, 8.0],
                      [9.0, 10.0, 11.0, 12.0]])
index = paddle.to_tensor([0, 1, 1, 2], dtype='int32')
index = paddle.to_tensor([0, 1, 0], dtype='int32')
out_z1 = paddle.index_select(x=x, index=index)
#[[1. 2. 3. 4.]
# [5. 6. 7. 8.]
# [5. 6. 7. 8.]]
with Benchmark("index_select"):
    out_z2 = paddle.index_select(x=x, index=index, axis=1)
#[[ 1.  2.  2.]
# [ 5.  6.  6.]
# [ 9. 10. 10.]]
out_z1

# ## mask_select
# 

# In[51]:


import paddle

x = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0],
                      [5.0, 6.0, 7.0, 8.0],
                      [9.0, 10.0, 11.0, 12.0]])
mask = paddle.to_tensor([[True, False, False, False],
                         [True, True, False, False],
                         [True, False, False, False]])
out = paddle.masked_select(x, mask)
out
#[1.0 5.0 6.0 9.0]

# In[ ]:


## where已经对齐

# In[64]:



# x is ndarray, shape=(m, n)
# torch
import numpy as np 
x = np.ndarray((2, 3))
b = torch.tensor(x)
bnorm = torch.norm(b, p=2, dim=-1, keepdim=True)
# b[bnorm.flatten(), :] 


# paddle报错
a = paddle.to_tensor(x)
anorm = paddle.norm(a, p=2, axis=-1, keepdim=True)
print(anorm)
# a[anorm.flatten(), :] 



# In[62]:


# x is ndarray, shape=(m, n)
# torch
b = torch.tensor(x)
bnorm = torch.norm(b, p=2, dim=-1, keepdim=True)
bn = torch.where(bnorm>=1, b/bnorm, b)

# paddle报错，要求 condition 与 x、y 的 shape 一样。
a = paddle.to_tensor(x)
anorm = paddle.norm(a, p=2, axis=-1, keepdim=True)
an = paddle.where(anorm>=1, a/anorm, a)
print(bn, an)

# In[65]:


b = torch.tensor([[1,2],[3,4]])
print(b)
print(b[torch.tensor([[True,False],[False,True]])])

a = paddle.to_tensor([[1,2],[3,4]])
print(a)
print(a[paddle.to_tensor([[True,False],[False,True]])])

# ## 多维索引
# 看到了曙光，可以这样组合来进行多维索引啊！ 
# 只要将第一维度的索引使用全量索引即可。

# In[14]:


import numpy as np
import paddle

t = paddle.to_tensor(np.arange(0, 12).reshape(3, 4))
i = paddle.to_tensor([0, 0, 1, 2])
i = [0, 0, 1, 2]
with Benchmark("list(range"):
    i = list(range(t.shape[0]))
with Benchmark("paddle.arange"):
    paddle.arange(t.shape[0])
print(i)
j = paddle.to_tensor([0, 0, 3])

print('t[i]:')
print(t[i])
print('t[i, j]:')
print(t[i, j])
with Benchmark("index "):
    t[i, j] = 42 
print(t)

# In[68]:


t[i, j]  = 7

# In[1]:


1

# In[ ]:



