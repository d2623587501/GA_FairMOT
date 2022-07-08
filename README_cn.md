# GA_FairMOT(低照度场景)
## 1 研究基础
###  1.1 论文研究来源
> FairMOT是由华中科技大学和微软亚洲研究院联合提出的，论文地址[FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking](https://arxiv.org/abs/2004.01888)，本项目是在其基础上的进一步改进，为研究提供理论基础。
### 1.2 代码基础来源
> FairMOT原模型代码来源于[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)，为基于飞桨PaddlePaddle的端到端视觉套件，覆盖**目标检测、实例分割、跟踪、关键点检测**等方向，为模型模型研究提供代码基础。
### 1.3 数据来源
> MOT17公开数据源于[motchallenge](https://motchallenge.net/data/MOT17/)，目前多目标跟踪领域的重要基准是MOTChallenge,作为上传并公布多目标跟踪方法研究成果的公共平台，其拥有最大的公开行人跟踪数据集。
### 1.4 AI社区分享
>  [AI Studio](https://aistudio.baidu.com/)是基于百度深度学习开源平台飞桨的人工智能学习与实训社区，为开发者提供了功能强大的线上训练环境、免费GPU算力及存储资源。[泸沽](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/539945)这是我的社区账号，欢迎各位大佬前来探讨学习。
## 2 项目文件介绍
 1. requirements.txt：代码运行安装配置文件
 2. configs：模型运行的参数文件
 3. dataset：数据集存放文件
 4. ppdet：模型存放文件，包含各个模型模块
 5. tools.：训练、预测文件
 6. output：权重文件，权重下载[Google云盘](https://drive.google.com/file/d/1JHWOnjZ3Yrq-a7qn4Uc6WDuj-27WlJbG/view?usp=sharing)
## 3 项目运行步骤
 1. 安装依赖
 ```
pip install -r requirements.txt
```
 2. 模型训练
  ```
   CUDA_VISIBLE_DEVICES=0 
   !python -u tools/train.py \ 
   -c configs/fairmot/fairmot_dla34_30e_576x320.yml \
```
 3. 模型评估
 ```
CUDA_VISIBLE_DEVICES=0 
!python tools/eval_mot.py \ 
-c configs/fairmot/fairmot_dla34_30e_1088x608.yml \ 
-o weights=output/T_724fairmot_dla34_30e_1088x608/model_final.pdparams
```