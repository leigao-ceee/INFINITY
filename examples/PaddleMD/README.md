# PaddleMD
Paddle for MD 飞桨分子动力学模拟科学计算
## 关于
PaddleMD打算提供一个简单易用的API，用于使用飞桨（PaddlePaddle）进行分子动力学模拟。这使得研究人员能够更快速地进行力场开发研究，并利用飞桨的易用和强大功能，将神经网络潜力无缝集成到动力学模拟中。

PaddleMD参考学习自[TorchMD](https://github.com/torchmd/torchmd)



## 许可

项目使用Apache-2.0 license许可

项目中使用了Moleculekit等其它软件，请遵守相关许可制度。
we use several file format readers that are taken from Moleculekit which has a free open source non-for-profit, research license. This is mainly in torchmd/run.py. Moleculekit is installed automatically being in the requirement file. Check out Moleculekit here: https://github.com/Acellera/moleculekit

## 环境安装初始化
需要的软件软件包可以通过`pip install -r requirements.txt` 和 `pip install -r requirements_tests.txt` 进行安装。

具体步骤：

### 1 安装飞桨环境
`pip install paddlepaddle-gpu`
具体命令见paddlepaddle.org.cn[网站安装指引](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/guides/09_hardware_support/rocm_docs/paddle_install_cn.html)

### 2 安装相关软件包
2.1 通过`pip install -r requirements.txt` 和 `pip install -r requirements_tests.txt` 进行安装。
```
pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple
pip install -r requirements_tests.txt -i https://mirror.baidu.com/pypi/simple
pip install  ase  -i https://mirror.baidu.com/pypi/simple
```
2.2 也可以直接安装：
`pip install moleculekit pyyaml tqdm pandas networkx scipy parmed ase`

2.3 若有软件安装不成功，比如moleculekit没有安装成功， 可以尝试使用降低软件包版本解决：`pip install "moleculekit<=0.9.14"` 或者使用conda安装甚至使用源码安装。

### 3 使用conda安装前面没有安装成功的软件包
若使用pip无法正常安装成功，可以使用conda安装
```
conda install moleculekit -c acellera
conda install openmm -c conda-forge
```

### 4 若没有条件进行conda安装，可以采取源码安装
比如AIStudio平台，无法直接只用conda安装软件，可以采用源码安装方式：
到github.com查找相应的软件包源码，git clone下载后， 编译安装。具体步骤略。



## 例子
在当前目录有 `tutorial.ipynb` ，可以用notebook动态调试的方式使用PaddleMD。

# 项目使用和进度
## 项目布局
当前项目列表
```
├── 2PaddleMD.ipynb
├── 3集成测试.ipynb
├── bak2PaddleMD1.ipynb
├── bin
├── examples
├── input.yaml
├── logs
├── mytrajectory.npy
├── mywaterrun
├── paddlemd
├── paddlemd.egg-info
├── PaddleMD.ipynb
├── profiler_log
├── README.md
├── requirements_tests.txt
├── requirements.txt
├── run.py
├── setup.py
├── SpeedTest.ipynb
├── test-data
├── tests
└── tutorial.ipynb
```
### tutorial.ipynb 为例子  
### PaddleMD.ipynb为源码统一编辑文件，可以在notebook下编辑所有核心代码，编辑后只要运行一下就写入相应文件。
### 2PaddleMD.ipynb为动态编辑和运行文件，可以边改代码，边看效果。
注意调试完成后，需要在PaddleMD.ipynb里写入修改，以便代码一致。
### 3测试.ipynb 为集成测试文件
首先使用`python setup.py develop`安装paddlemd开发模式。

然后就可以测试了。
比如可以使用`python tests/test_paddlemd.py`进行集成测试，使用`./bin/paddlemd --conf tests/water/water_conf.yaml`测试水分子，使用`./bin/paddlemd --conf tests/prod_alanine_dipeptide_amber/conf.yaml`测试prod alanine dipeptide前丙氨酸二
目前水分子和前丙氨酸二测试可以运行不报错。
集成测试，可以测试一部分，可以看到势能和力场等数值跟openmm的较接近。但是参数更新优化那块，还有问题。

## 当前已经实现的功能
1 本大框架和核心代码完成
势能和力场基本能对齐。

2 例子tutorial.ipynb可以执行

3 水分子和prod alanine dipeptide前丙氨酸二的测试可以通过

## 当前还存在的问题
1 集成测试无法完全通过。

2 没有找到论文中Trypsin的分子结构文件。所以没有测试Trypsin

3 参数自动更新优化那块还有问题。

## 进一步说明
本项目为参加[中国软件开源创新大赛：开源任务挑战赛(顶会论文复现赛)](https://aistudio.baidu.com/aistudio/competition/detail/249/0/introduction)的项目。
飞桨第六期论文复现赛128 https://aistudio.baidu.com/aistudio/competition/detail/205/0/task-definition
issue报名地址：https://github.com/PaddlePaddle/Paddle/issues/41482
torch代码学习：https://github.com/torchmd/torchmd

## 帮助和注释
未来会在github提供

## 引用Citation
Please cite:
```
@misc{doerr2020torchmd,
      title={TorchMD: A deep learning framework for molecular simulations}, 
      author={Stefan Doerr and Maciej Majewsk and Adrià Pérez and Andreas Krämer and Cecilia Clementi and Frank Noe and Toni Giorgino and Gianni De Fabritiis},
      year={2020},
      eprint={2012.12106},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph}
}
```
To reproduce the paper go to the tutorial notebook https://github.com/torchmd/torchmd-cg/blob/master/tutorial/Chignolin_Coarse-Grained_Tutorial.ipynb
