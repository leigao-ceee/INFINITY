# lettuce_paddle


## 目录


- [1. 简介]()
- [2. 效果展示]()
- [3. 准备环境]()
- [4. 开始使用]()
- [5. 参考链接与文献]()


## 1. 简介

本项目为使用paddle复现lettuce项目,lettuce是一个基于LBM的计算流体动力学框架,具有gpu加速计算、二维和三维快速成型等优点。

**论文:** [Lettuce: PyTorch-based Lattice Boltzmann Framework](https://arxiv.org/pdf/2106.12929.pdf)

**参考repo:** [https://github.com/lettucecfd/lettuce](https://github.com/lettucecfd/lettuce)


## 2. 效果展示

### 三维Taylor-Green旋涡Q准则等值面，雷诺数和网格分辨率分别为1600和256

<div>
    <img src="./figs/p5.png" width=250">
    <img src="./figs/p7.png" width=250"> 
    <img src="./figs/p10.png" width=250">
</div>

### 能量耗散率

<div>
    <img src="./figs/dp.png" width=400">
</div>                                          


## 3. 准备环境

* 下载代码

```bash
git clone https://github.com/simonsLiang/lettuce_paddle
```

* 安装paddlepaddle

```bash
# 需要安装2.2及以上版本的Paddle
# 安装GPU版本的Paddle
pip install paddlepaddle-gpu==2.2.0

更多安装方法可以参考：[Paddle安装指南](https://www.paddlepaddle.org.cn/)。

* 安装requirements
```bash
pip install -r requirements.txt
```

## 4. 开始使用

下面代码将在GPU上运行一个三维Taylor-Green旋涡

```
import lettuce as lt
import paddle
import numpy as np
import matplotlib.pyplot as plt
device = 'gpu'  
lattice = lt.Lattice(lt.D3Q27, device=device, dtype=paddle.float32) 
resolution = 80
flow = lt.TaylorGreenVortex3D(resolution, 1600, 0.05, lattice)
collision = lt.BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
streaming = lt.StandardStreaming(lattice)
simulation = lt.Simulation(flow, lattice, collision, streaming)
print("MLUPS: ", simulation.step(100))
```

运行
```
python run.py --resolution 256 --reynolds 1600
```
将会每隔1000步生成网格数为255x256x256,雷诺数位1600的vtr文件，可在第三方软件比如paraview进行仿真，同时得到动能变化数值，保存于TGV3DoutRes256E.npy中，如下：
<div>
    <img src="./figs/output1.png" width=300">
</div>     

运行
```
python plotE.py --filename TGV3DoutRes256E.npy --savename ./dissipation.png
```
将会在根据TGV3DoutRes256E.npy绘制能量耗散曲线，保存于./dissipation.png中，如下：

<div>
    <img src="./figs/output2.png" width=300">
</div>


## 5. 参考链接与文献
[Lettuce: PyTorch-based Lattice Boltzmann Framework](https://arxiv.org/pdf/2106.12929.pdf)

```
@inproceedings{bedrunka2021lettuce,
  title={Lettuce: PyTorch-Based Lattice Boltzmann Framework},
  author={Bedrunka, Mario Christopher and Wilde, Dominik and Kliemank, Martin and Reith, Dirk and Foysi, Holger and Kr{\"a}mer, Andreas},
  booktitle={High Performance Computing: ISC High Performance Digital 2021 International Workshops, Frankfurt am Main, Germany, June 24--July 2, 2021, Revised Selected Papers},
  pages={40},
  organization={Springer Nature}
}
```
