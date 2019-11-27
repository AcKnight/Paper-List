### Toward 3D Human Pose Estimation in the Wild:a Weakly-supervised Approach

- [ ] Use: https://github.com/xingyizhou/Pytorch-pose-hg-3d
- [x] Read: https://arxiv.org/pdf/1704.02447.pdf

```

```

### A Simple Yet Effective Baseline for 3D Human Pose Estimation

- [x] Use: https://github.com/una-dinosauria/3d-pose-baseline
- [x] Read: https://arxiv.org/pdf/1705.03098.pdf

```

```

### Sparse Representation for 3D Shape Estimation: A Convex Relaxation Approach

- [ ] Use: 
- [ ] Read: https://arxiv.org/pdf/1509.04309.pdf

### 3D Human Pose Estimation in the Wild by Adversarial Learning

- [x] Read: https://arxiv.org/pdf/1803.09722.pdf
- [ ] Use: 

```

```

### Single-Shot Multi-Person 3D Pose Estimation From Monocular RGB

- [ ] Read: https://arxiv.org/pdf/1712.03453.pdf
- [ ] Use: 

### Ordinal Depth Supervision for 3D Human Pose Estimation

- [ ] Read: https://arxiv.org/pdf/1805.04095.pdf
- [ ] only vaild code: https://github.com/geopavlakos/ordinal-pose3d/

### DRPose3D: Depth Ranking in 3D Human Pose Estimation

- [x] Read: https://arxiv.org/pdf/1805.08973.pdf
- [ ] Use: 

```
1.成对排序卷积神经网络(PRCNN)：从单个RGB图像中提取成对人体关节深度排序信息，将深度排序问题转换成分类问题
2.提出由粗到精的3D姿态提取器，取名DPNet，由DepthNet和PoseNet组成，输入是2D关键点位置信息和深度排序矩阵，先用DPNet粗略的估计深度值，然后由PoseNet从粗到细的方式精确回归3D姿态
3.在主体周围随机采样相机位置，在其中添加噪声，使增强的数据服从训练集的分布

```

![1560250340995](/home/nlpr/.config/Typora/typora-user-images/1560250340995.png)

### Self-Supervised Learning of 3D Human Pose using Multi-view Geometry

- [ ] paper:https://arxiv.org/pdf/1903.02330.pdf
- [ ] github:https://github.com/mkocabas/EpipolarPose

#### 贡献

- 单张图片估计3D姿态，训练时不需要3D监督或者摄像机外参，利用极线几何和2DGT创建3D监督
- 提出Pose Structure Score(PSS)，新的3D姿态估计评测标准

#### pipeline

![1561636682669](/home/xiangran/.config/Typora/typora-user-images/1561636682669.png)

- 训练过程中上方的分支输入两个视角的图片，输出3D姿态估计结果；下方分支接收相同的输入，输出监督信息同样是3D姿态估计结果
- 部署阶段接收单张图片，输出相应3D姿态估计结果
- 上分支参数不锁定，下分支在训练时是锁定的

#### Training

- 假设第一个摄像头坐标系是全局坐标系
- 在图像平面，对不同摄像机相同的关节点的图像坐标，对任意 $j$ 使用RANSAC算法使得满足 $U_{i, j} F U_{i+1, j}=0 $ ，然后计算essential matrix E by $E=K^{T} F K$， 通过将E SVD分解得到4个对于 $R$ 的可能解，用 cheirality     check 来验证
- 利用多项式三角剖分，从两张不同视角拍摄的图片上获取相应的3D姿态

#### Drawback

- 目前无法在实验室以外的图像上推广



### Semantic Graph Convolutional Networks for 3D Human Pose Regression (CVPR 2019)

- [ ] paper:https://arxiv.org/pdf/1904.03345.pdf
- [ ] git:https://github.com/garyzhao/SemGCN



### 3D Human Pose Machines with Self-supervised Learning
- [ ] paper:https://arxiv.org/pdf/1901.03798.pdf
- [ ] git: https://github.com/chanyn/3Dpose_ssl

#### pipeline

1. 提取2D姿态特征 $f_t^{2d}$ 后接两个子模块
2. 2Dto3D pose Transformer：根据当前帧的时序来估计中间3D结果
3. 3Dto2D pose projector module: 引入自监督校正机制双向refine中间3D估计结果
4. ![1561907837699](/home/nlpr/.config/Typora/typora-user-images/1561907837699.png)

### Absolute Human Pose Estimation with Depth Prediction Network

- [ ] Paper:https://arxiv.org/pdf/1904.05947.pdf
- [ ] GIT:https://github.com/vegesm/depthpose

#### pipline

![1561976454993](/home/nlpr/.config/Typora/typora-user-images/1561976454993.png)

1. 2D Pose Estimator: OpenPose
2. Depth Estimator和3D PoseNet一起训练，选取Megadepth为baseline（）
3. depth prediction和normalized像素坐标送入3D PoseNet(Martinez)

##### Code:

###### baseline.py

- 输入14*3，14个关键点，(x, y, score)，输出17个关键点(x, y, z)，hip坐标系

- 误差计算：$\hat{t}=\underset{t \in \mathbb{R}^{3}}{\operatorname{argmin}}\left\|P^{2 D}-\Pi\left(P^{3 D}+t\right)\right\|_{2}^{2}$  通过下式解决

  $\hat{t}=\alpha\left(\begin{array}{c}{\overline{P}^{2 D}} \\ {f}\end{array}\right)-\left(\begin{array}{c}{\overline{P} 3 D^{\prime}} \\ {0}\end{array}\right)$
  $\alpha=\frac{\sum_{i}\left\|P_{i}^{3 D^{\prime}}-\overline{P}^{3 D^{\prime}}\right\|_{2}^{2}}{\sum_{i}\left\langle P_{i}^{2 D}-\overline{P}^{2 D}, P_{i}^{3 D^{\prime}}-\overline{P} 3 D^{\prime}\right\rangle}$

- 将$\hat{t}$ 直接与网络输出相加和GT做误差

###### full.py

- [MegaDepth][https://github.com/lixx2938/MegaDepth]
- 

#### 分析

- 现实中使用的时候需要分开跑openpose和MegaDepth，主要算力就是这两者(不清楚速度)，3DposeNet是marinez的简单变种
- 数据集有图像有3D标签，基于[Single-Shot Multi-Person 3D Pose Estimation From Monocular RGB][http://gvv.mpi-inf.mpg.de/projects/SingleShotMultiPerson/] ，有数据集使用的matlab代码
- 暂不清楚用自己摄像头测试MegaDepth的表现如何
- 因为有把2D姿态做normalization的操作，其实就是根据不同摄像机的内参把3D姿态降维到2D(z=1)平面，所以有希望泛化到普通摄像头而不用做图像转换
- 拍摄数据的摄像头高度不一，角度也不一定相同

Note;

- 用自己图片生成2D送入对比结果(depth estimation)
- 几何搜索或者slovepnp

#### ==数据读取处理==

1. 读入pose2d、pose3d、goodpose
2. 按照openpose hip关键点评分是否大于0.5，留取好的pose2d/3d
3. 读取当前seq的mupots图片size，并按照不同的seq设置depth_w和depth_h
4. 处理pose2d:
   1. 按照摄像机内参处理pose2d->self.normed_poses:主要是==内参+2dpose归一化==
      - openpose25个关键点转为14个
      - 根据摄像机内参减c除以f，复制hip关键点的坐标
      - 每个人的每个关键点都减去hip关键点
      - 根据openpose的关键点的评分记录bad_frame,具体做法是评分小于0.1的关键点记为bad_frame,根据bad_frame的记录，将相应的关键点的图像坐标设置为(-1700/fx,-1700/fy)
      - reshape（帧，人数，关键点数，3）->（帧，人数，关键点数*3），==最后附加上hip关键点坐标==
   2. 图片在送入网络时，会进行缩放，所以相应的2dpose也要进行缩放，具体做法就是每个2dpose坐标*target_size/img_size
5. 处理3dpose:
   - 首先复制每个帧每个人的hip关键点3d坐标，==对每个hip坐标的z轴信息做log处理，注意此处没有对x，y轴的坐标做log==
   - remove_root函数，除了将每个关键点都减去hip关键点之外，把hip关键点的信息删除
   - 最后把复制并log处理的hip关键点信息贴在最后
   - pose3d减均值除方差

#### ==送入网络的处理==

1. mupots图片读取，像素归一化，通道变换
2. pose2d，normed_pose，和good_pose送入网络，之后得到结果pred

#### 得到输出的处理

1. 对pred进行乘方差加均值处理，单独记录pred中的hip关键点数据
2. ==对hip的z轴的数据做自然指数处理，恢复其真实的z轴深度==，每个3d关键点加上hip关键点的数据
3. 把hip关键点放到下标为14的位置(恢复hip关键点原先真实的位置，方便与真实值对比)

#### Evaluation_results



### Monocular 3D Human Pose Estimation In The Wild Using Improved CNN Supervision

- [ ] paper:https://arxiv.org/pdf/1611.09813.pdf
- [ ] page:http://gvv.mpi-inf.mpg.de/3dhp-dataset/

#### pipeline

![1562051682251](/home/nlpr/.config/Typora/typora-user-images/1562051682251.png)

1. 2D PoseNet 从BB获取2D姿态点
2. BB和2D姿态点K一起送入3D姿态估计(root-centered)
3. 利用3D、2D姿态估计结果和摄像机标定参数计算全局3D姿态坐标和透视校正

#### Detail

##### 1. Bounding Box and 2D Pose Computation

##### 2. 3D Pose Regression

##### 3. Multi-level Corrective Skip Connections

##### 4. Global Pose Computation

1. 通过BB获取2D姿态估计信息的方式使得3D pose regression丢失global pose information

2. 通过 $P_{fused}$ 、摄像机内参和2D姿态估计结果重建global 3D pose $P^{[G]}$ 

3. BB裁剪过程中有旋转，计算相应的R，对相应的3D姿态估计结果进行旋转

4. $\begin{aligned} z &=f \frac{\sum_{i}\left\|P_{[x y]}^{i}-\overline{P}_{[x y]}\right\|^{2}}{\sum_{i}\left(K^{i}-\overline{K}\right)^{\top}\left(P_{[x y]}^{i}-\overline{P}_{[x y]}\right)} \\ & \approx f \frac{\sqrt{\sum_{i}\left\|P_{[x y]}^{i}-\overline{P}_{[x y]}\right\|^{2}}}{\sqrt{\sum_{i}\left\|K^{i}-\overline{K}\right\|^{2}}} \end{aligned}$  利用2D和3D估计结果来估计keypoint的深度$T=\frac{\sqrt{\sum_{i}\left\|P_{[x y]}^{i}-\overline{P}_{[x y]}\right\|^{2}}}{\sqrt{\sum_{i}\left\|K^{i}-\overline{K}\right\|^{2}}}\left(\begin{array}{c}{\overline{K}_{[x]}} \\ {\overline{K}_{[y]}} \\ {f}\end{array}\right)-\left(\begin{array}{c}{\overline{P}_{[x]}} \\ {\overline{P}_{[y]}} \\ {0}\end{array}\right)$

   
   
   

### Generalizing Monocular 3D Human Pose Estimation in the Wild

- [x] paper:https://arxiv.org/pdf/1904.05512.pdf
- [x] GIt:https://github.com/llcshappy/Monocular-3D-Human-Pose

#### main contribution:

- 3D label生成网络和几何搜索方法来进一步细化3D关键点
- 基于以上创建40w张户外图片及其标注
- 基于以上数据集训练的3D网络达到了SOTA

#### pipeline:



![1562219617957](/home/nlpr/.config/Typora/typora-user-images/1562219617957.png)

1. **Stereoscopic view synthesis subnetwork**

   - [Single view stereo matching][https://arxiv.org/pdf/1803.02612.pdf]
   - Input：左向视角2D姿态像素坐标 Output：右向视角2D姿态像素坐标
   - $s\left[\begin{array}{c}{u_{R}} \\ {v_{R}} \\ {1}\end{array}\right]=\left[\begin{array}{ccc}{\alpha_{x}} & {0} & {u_{0}} \\ {0} & {\alpha_{y}} & {v_{0}} \\ {0} & {0} & {1}\end{array}\right]\left[\begin{array}{c}{x_{c}+\Delta x} \\ {y_{c}} \\ {z_{c}}\end{array}\right]=M_{c} P_{c}$ 把3D姿态x轴平移大约500mm
   - 使用Human3.6M和Unity toolbox共4.8million 2D/3D对训练

2. **3D pose reconstruction subnetwork**

   - $Q_{r}=f_{r}\left(\left(u_{L}, v_{L}\right),\left(u_{R}, v_{R}\right)\right)$ where $Q_{r}=\left(x_{r}, y_{r}, z_{r}\right) \in \mathbb{R}^{3 \times N}$ 
   - $Q_r$ 是较为粗糙的结果

3. **Geometric search scheme**

   - 基于重投影误差

   - $Q_{g}=f_{g e o}\left(P_{g t}, Q_{r}\right)$ where $P_{g t}=\left(u_{g t}, v_{g t}\right) \in \mathbb{R}^{2 \times N}$ ，$Q_g$ 是**root joint(pelvis)coordinate** ，不是**camera coordinate**

   - 启发式搜索解决**absolution depth**: ![1562223665335](/home/nlpr/.config/Typora/typora-user-images/1562223665335.png)

   - $\begin{aligned} \widetilde{x}_{r} &=\left(u_{g t}-c_{x}\right)\left(z_{r}+\Delta z\right) / f_{x} \\ \widetilde{y}_{r} &=\left(v_{g t}-c_{y}\right)\left(z_{r}+\Delta z\right) / f_{y} \end{aligned}$ , $Z_r$ 是hip坐标系的，通过一点点增加$\Delta_z$ , 使以下Loss最小

     $L_{g e o}=\underset{\Delta z}{\arg \min }\left\|\left(\left(\widetilde{x}_{r}-x_{r}\right)^{2}+\left(\widetilde{y}_{r}-y_{r}\right)^{2}\right)\right\|_{2}^{2}$  

   - 以上获取**camera frame**下的3D坐标

#### 分析:

-  

#### Training

- 



### Camera Distance-aware Top-down Approach for 3D Multi-person Pose Estimation from a Single RGB Image

- [ ] code:https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE
- [ ] paper:https://arxiv.org/pdf/1907.11346.pdf

#### main contribution:

1. 单目多人姿态估计通用框架，特性：learning-based/camera distance-aware/top-down approach
2. 提出RootNet，能够输出摄像机坐标系下多人关键点的坐标
3. 在多个数据集上达到SOTA

### overview

- **goal:** 得到摄像机坐标系下的多人关键点的绝对坐标
- 框架包含：DetectNet、RootNet、PoseNet，作者称为top-down approach
  - DetectNet，检测每个人的bbx
  - RootNet，输入DetectNet的bbx的图像，输出$(x_R,y _R, Z_R)$ ，Z~R~ 是绝对深度
  - PoseNet，输入DetectNet的bbx的图像，输出$(x_j,y_j,Z_j^{rel})$ 第三个量是root-relative depth value, $Z_j^{abs} = Z_R + Z_j^{rel}$ 

### pipeline

![1565150681663](/home/nlpr/.config/Typora/typora-user-images/1565150681663.png)

- **DetectNet** 使用Mask R-CNN，输入图片256×256，得到剪裁框 
- **RootNet**，输入ResNet提取的**fm**，利用cam_param得到的**K**，输出是2D Root 图像坐标和深度信息
  - ![1565167271300](/home/nlpr/.config/Typora/typora-user-images/1565167271300.png)
  - $k=\sqrt{\alpha_{x} \alpha_{y} \frac{A_{r e a l}}{A_{i m g}}}$ ，A~real~ 设置为2000mm*2000mm，A~img~会固定宽高比为1:1，$\alpha_{x},\alpha_{y}$ = f~x~ / c~x~ ，**尽管** K可以一定程度上表示人离摄像头的距离，但因为A~real~ 固定为2000×2000，K的结果会出现很大偏差，比如两个人的A~img~不同，但距离相同，又或者A~img~ 相同但是距离不同，**因此**设计RootNet利用图像特征来纠正A~img~最终改变K的值，比如一个人是站着还是蹲着，是大人还是小孩，通过输出矫正参数$\gamma$ 来调节，$A_{img}^{\gamma}$ = $\gamma$ * $A_{img}$ ，最终Z~R~ = $k/\sqrt{\gamma}$。
  -  可以根据不同的摄像头内参进行归一化，输出合理的矫正参数
  - 2D part用反卷积，Depth part用全局平均池化，最后都用1*1卷积压缩通道数
  - 损失函数：$L_{r o o t}=\left\|x_{R}^{*}-x_{R}\right\|_{1}+\left\|y_{R}^{*}-y_{R}\right\|_{1}+\left\|Z_{R}^{*}-Z_{R}\right\|_{1}$
- **PoseNet**，参考`Integral human pose regression. In ECCV, 2018.`，SOTA开源，
  - 第一步，利用ResNet从裁剪的图片中提取有用的全局信息
  - 第二步，连续三个反卷积层上采样，接ReLu，1*1卷积，soft-argmax提取2D图像坐标和相对深度值
  - 损失函数：$L_{p o s e}=\frac{1}{J} \sum_{j=1}^{J}\left(\left\|x_{j}^{*}-x_{j}\right\|_{1}+\left\|y_{j}^{*}-y_{j}\right\|_{1}+\left\|Z_{j}^{r e l *}-Z_{j}^{r e l}\right\|_{1}\right)$

 ### 实现细节

- Mask R-CNN没有fine-tuning
- 初始化学习率1×10^-3^
- 参数初始化0.001
- 256*256 input image
- 数据增强：正负30°翻转，水平翻转，颜色抖动，合成遮挡，其中水平翻转遵循Sun *et al.*  
- 训练RootNet和PoseNet 20个epochs，4块1080Ti，分别耗费两天时间

### POSENet 测试KITTI数据预处理

1. ##### 读取图片

2. crop patch from img


note:

1. 研读微软亚研院的pose工作
2. kitti GT提取，做loss曲线
3. 























































