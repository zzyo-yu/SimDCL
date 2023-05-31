# SimDCL
This is our PyTorch implementation code for our paper:
> [1] Xu Yuhao, Wang Zhenhai, Wang ZhiRu, Guo YunLong, Fan Rong, Tian Hongyu and Wang Xing . “SimDCL: dropout-based simple graph contrastive learning for recommendation.” Complex & Intelligent Systems (2023)

# Requirements

> recbole==1.1.1
> 
> pyg>=2.0.4
> 
> pytorch>=1.7.0
> 
> python>=3.7.0

# Quick-Start

With the source code, you can use the provided script for initial usage of our library:

> python run_recbole_gnn.py

If you want to change the models or datasets, just run the script by setting additional command parameters:

> python run_recbole_gnn.py -m [model] -d [dataset]
