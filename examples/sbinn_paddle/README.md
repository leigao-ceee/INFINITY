# sbinn_paddle

## 目录


- [1. 简介]()
- [2. 复现精度]()
- [3. 准备环境]()
- [4. 开始使用]()
- [5. LICENSE]()
- [6. 参考链接与文献]()


## 1. 简介

本项目为使用paddle复现sbinn项目。

**论文:** [Systems Biology: Identifiability analysis and parameter identification via systems-biology informed neural networks](https://arxiv.org/pdf/2202.01723.pdf)

**参考repo:** [https://github.com/lu-group/sbinn](https://github.com/lu-group/sbinn)
                                        
快速使用可参考[AIStudio](https://aistudio.baidu.com/aistudio/projectdetail/3966656?contributionType=1&shared=1)        

代码主要基于[sbinn](https://github.com/lu-group/sbinn)和[deepxde](https://github.com/lululxvi/deepxde)修改，在涉及商业应用时需谨慎使用！

Code was heavily based on https://github.com/lu-group/sbinn and https://github.com/lululxvi/deepxde . Users should be careful about adopting these functions in any commercial matters.

感谢百度飞桨提供的算力支持

## 2. 复现精度

论文fig13复现结果：
（蓝色实线为目标值，红色虚线为预测值）

### 1. 学习率为1e-6
epoch=400000:
<div>
    <img src="./figs/Ip40e.png" width=300">
    <img src="./figs/Ii40e.png" width=300">
</div>     
<div>
    <img src="./figs/G40e.png" width=300">
    <img src="./figs/IG40e.png" width=300">
</div> 

<div>
</div> 
epoch=600000:
<div>
    <img src="./figs/Ip60e.png" width=300">
    <img src="./figs/Ii60e.png" width=300">
</div>     
<div>
    <img src="./figs/G60e.png" width=300">
    <img src="./figs/IG60e.png" width=300">
</div> 

### 2. 学习率为5e-8
epoch=400000:
<div>
    <img src="./figs/Ip40e2.png" width=300">
    <img src="./figs/Ii40e2.png" width=300">
</div>     
<div>
    <img src="./figs/G40e2.png" width=300">
    <img src="./figs/IG40e2.png" width=300">
</div> 

<div>
</div> 
epoch=600000:
<div>
    <img src="./figs/Ip60e2.png" width=300">
    <img src="./figs/Ii60e2.png" width=300">
</div>     
<div>
    <img src="./figs/G60e2.png" width=300">
    <img src="./figs/IG60e2.png" width=300">
</div> 

训练日志,以及生成的variable.csv,vallist.csv可在[BaiduYun](https://pan.baidu.com/s/10XljlwKKUdJXrQ7VmL7gTA?pwd=vfnr)下载

## 3. 准备环境

* 下载代码

```bash
git clone https://github.com/simonsLiang/sbinn_paddle
```

* 安装paddlepaddle

```bash
# 需要安装2.1.2的Paddle版本
# 安装GPU版本的Paddle
pip install paddlepaddle-gpu==2.1.2
```
                                            
更多安装方法可以参考：[Paddle安装指南](https://www.paddlepaddle.org.cn/)


## 4. 开始使用

此部分可参考[AIStudio](https://aistudio.baidu.com/aistudio/projectdetail/3966656?contributionType=1&shared=1)       
运行以下命令生成数据                                            
```bash
python  data_generation.py                                           
```   
                                            
然后运行以下命令训练模型和参数                                        
```bash
python  sbinn_paddle.py                                           
```                                          

训练完成后得到variable.csv文件,再运行以下命令得到最终的生成参数，保存于vallist.csv文件中                                                  
```bash
python  variable_to_parameter_transform.py                                        
```  
                                            
接着利用生成的参数得到预测数据                                                  
```bash
python  data_prediction.py                                           
```     
            
最后对预测数据与原数据进行绘图对比                                               
```bash
python  plot.py                                           
```                                                                                     

## 5. LICENSE

[LICENSE](./License)


## 6. 参考链接与文献
[Systems Biology: Identifiability analysis and parameter identification via systems-biology informed neural networks](https://arxiv.org/pdf/2202.01723.pdf)
