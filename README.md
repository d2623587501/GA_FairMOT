 - [中文](https://github.com/d2623587501/GA_FairMOT/blob/main/README_cn.md)
 - [English](https://github.com/d2623587501/GA_FairMOT/blob/main/README.md)

# GA_FairMOT(Low-illumination scenes)
## 1 Research Base
###  1.1 Thesis Research Sources
> FairMOT is a joint initiative of Huazhong University of Science and Technology and Microsoft Research Asia，Thesis Address[FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking](https://arxiv.org/abs/2004.01888)，This project is a further improvement on its basis to provide a theoretical basis for the research.
### 1.2 Code Base Source
> The original FairMOT model code is derived from[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)，End-to-end vision suite based on Flying PaddlePaddle, covering **target detection, instance segmentation, tracking, key point detection** and other directions, providing a code base for model model research。
### 1.3 Data Source
> MOT17 public data is derived from[motchallenge](https://motchallenge.net/data/MOT17/)，An important benchmark in the field of multi-objective tracking is MOTChallenge, which is a public platform for uploading and publishing research results of multi-objective tracking methods and has the largest publicly available pedestrian tracking dataset.
### 1.4 AI Community Sharing
>  [AI Studio](https://aistudio.baidu.com/)It is an AI learning and practical training community based on Baidu's deep learning open source platform Flying Paddle, which provides developers with a powerful online training environment, free GPU computing power and storage resources.[LuGu](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/539945)This is my community account, welcome to discuss and learn from the big guys。
## 2 Project Document Introduction
 1. requirements.txt：Code to run the installation configuration file
 2. configs：Configuration files for model runs
 3. dataset：Dataset storage file
 4. ppdet：Model storage file containing the individual model blocks
 5. tools.：Training, Prediction files
 6. output：Weighted files, weighted downloads at[Google Cloud Drive](https://drive.google.com/file/d/1JHWOnjZ3Yrq-a7qn4Uc6WDuj-27WlJbG/view?usp=sharing)
## 3 Project Operation Steps
 1. Installing dependencies
 ```
pip install -r requirements.txt
```
 2. Model Training
  ```
   CUDA_VISIBLE_DEVICES=0 
   !python -u tools/train.py \ 
   -c configs/fairmot/fairmot_dla34_30e_576x320.yml \
```
 3. Model Evaluation
 ```
CUDA_VISIBLE_DEVICES=0 
!python tools/eval_mot.py \ 
-c configs/fairmot/fairmot_dla34_30e_1088x608.yml \ 
-o weights=output/T_724fairmot_dla34_30e_1088x608/model_final.pdparams
```