### 背景和目的

#### 什么是AI辅助医疗诊断?
  - 患者拍摄X射线照片后, AI算法通过对图片分析辅助医生做医疗诊断, 包括
    - 自动对不同的器官做区域分割, 精确计算比例或角度
    - 根据不同疾病特征, 发现潜在的病灶可能, 提示医生
    - 医生审核诊断结果后, 自动生成结构化诊断报告

  <table>
    <tr>
      <th>脊柱侧弯</th>
      <th>心影增大</th>
      <th>肺结节可能</th>
    </tr>
    <tr>
      <td><img src="/assets/images/AI_assisted_diagnose/图片 1.png" alt="图片 1" style="zoom:100%;" /></td>
      <td><img src="/assets/images/AI_assisted_diagnose/心影增大诊断.jpeg" alt="心影增大诊断" style="zoom:15%;" /></td>
      <td><img src="/assets/images/AI_assisted_diagnose/简单器官分割+肺结节检测.jpg" alt="简单器官分割+肺结节检测" style="zoom:18%;" /></td>
    </tr>
  </table>

#### 为什么需要AI辅助诊断?
  - 提高效率
    - 根据经验, 医生完成一次人工诊断需要3-5分钟, 而AI算法在10秒内即可给出初步诊断结果, 医生审核结果并给出诊断报告的时间不超过1分钟, 效率极大提高
  - 风险预警
    - 医院繁忙时段, 医生数量可能不足, 不能立刻查看结果, 造成X射线图片堆积. 而有些疾病(如严重肺气肿)时效性要求很高, 发现病情后需要立刻给患者安排手术治疗, 否则可能危及生命
    - AI辅助诊断可以实时给出所有患者的初诊结果, 根据病情优先级排序, 较严重的病情优先交给医生审核, 很严重的病情将提醒医生立即处理
  - 补充偏远地区的医疗资源
  - 教学使用

------

### 处理流程和结果

#### 软件展示
  <img src="/assets/images/AI_assisted_diagnose/软件展示.png" alt="软件展示" style="zoom:100%;" />{:.align-center}

#### 测试结果
  <img src="/assets/images/AI_assisted_diagnose/image-20220804160558024.png" alt="image-20220804160558024" style="zoom:80%;" />{:.align-center}

#### 处理流程是怎样的?
  - 先做数据过滤, 确保图像清晰、有效
  - 然后做器官区域分割, 包括脊椎、左右肺部、肋骨等
  - 对于简单的疾病, 直接输入原图或者分割后的结果做分类即可
  - 对于复杂的疾病, 需要配合定制的目标检测或者分类模型来判断
  - 最后汇总, 生成病例报告
  <img src="/assets/images/AI_assisted_diagnose/算法流程图.png" alt="算法流程图" style="zoom:40%;" />{:.align-center}

------

### 遇到的挑战 - 数据准备

#### 训练数据是怎么准备的?
  - <span id="公开数据集的问题">调研了公开数据集: [ChestX-ray14(2017)](https://arxiv.org/pdf/1705.02315.pdf)、[CheXpert(2019)](https://arxiv.org/pdf/1901.07031.pdf)、[MIMIC-CXR(2019)](https://arxiv.org/pdf/1901.07042.pdf)等</span>
    - 公开数据集的一些问题: 疾病种类不全、标注方法不一致、标注质量不高、国内外医学界对病灶定义差异等
    - 对公开数据集的准确性还有很多的质疑, 如[质疑一](https://zhuanlan.zhihu.com/p/37384516)、[质疑二](https://lukeoakdenrayner.wordpress.com/2017/12/18/the-chestxray14-dataset-problems/)、[质疑三](https://twitter.com/erictopol/status/930980060835614720)等

  <table style="width: 100%;">
    <tr>
      <th>公开数据集的标注</th>
      <th>期望的标注</th>
    </tr>
    <tr>
      <td><img src="/assets/images/AI_assisted_diagnose/公开数据集举例.png" alt="公开数据集举例" style="zoom:48%;" /></td>
      <td><img src="/assets/images/AI_assisted_diagnose/期望的标注-9511800.png" alt="期望的标注" style="zoom:48%;" /></td>
    </tr>
  </table>

  - <span id="自制数据集">为了符合国内标准, 统一病种和标注方法 , 决定自制数据集</span>
    - 与上海交通大学附属瑞金医院(三甲)合作, 数据库共有18万张原始X射线照片及对应病例报告, 院内脱敏及清洗后获得有效性数据共15.3万条
    - 由瑞金医院放射科主任医师主导标注, 并对标注结果负责, 我们负责标注系统建立和数据准备, 得到精标数据3.4万张
    - 建立瑞金医院DR数据集(不开源)
  - <span id="数据标注">数据标注</span>
    - 开发标注系统
    - 需要对器官区域做分割, 同时对病灶类别和位置打标签
    - 采用矩形框和多边形标注两种形式
    <img src="/assets/images/AI_assisted_diagnose/标注系统.png" alt="标注系统" style="zoom:40%;" />{:.align-center}
  - <span id="成本控制">成本控制</span>
    - 专业数据集制作的成本很高, 尤其是医生标注费用, 平均40RMB/张
    - 建立初级医生标注, 高级医生审核的二级机制, 可以降低成本, 同时确保结果的准确性
    - 先统一标注, 再按需求标注: 有的模型需要少量数据即可达到要求, 效果不好的模型有针对性地补充数据
    <img src="/assets/images/AI_assisted_diagnose/image-20220803120731568.png" alt="image-20220803120731568" style="zoom:40%;" />{:.align-center}

#### 数据是怎么做预处理的?
  - <span id="区域裁剪">区域裁剪</span>
    - 不同病种的出现区域不一样, 如肺结节只会出现在双肺位置, 将肺部区域裁剪下来单独检测, 优点如下
    - 可以在图像压缩后保留更多有效信息, 提高召回率
    - 可以去除无关信息, 降低假阳率
    <img src="/assets/images/AI_assisted_diagnose/图片裁剪的优势.png" alt="图片裁剪的优势" style="zoom:40%;" />{:.align-center}
  - <span id="数据增强">数据增强</span>
    - 随机缩放、随机擦除、水平翻转等常用数据增强方法
    - CLAHE特征加强
      - CLAHE (Contrast Limited Adaptive Histogram Equalization), 对比度有限的自适应直方图均衡算法, 通过基于分块的思想来提出直方图均衡算法, 广泛用于数据加强
      - 我们观察到CLAHE的处理可以明显影响X射线胸片上骨头的对比度, 比如椎体、肋骨等边缘特征的对比度更强
    <img src="/assets/images/AI_assisted_diagnose/图像增强-9513884.png" alt="图像增强" style="zoom:40%;margin-top:2em;" />{:.align-center}

------

### 遇到的挑战 - 算法开发

#### 脊柱侧弯是如何判断的?
  - <span id="cobb角及其定义">cobb角及其定义</span>
    - 首先找到造成最大夹角的两个椎体, 然后延长上椎体的上缘线和下椎体的下缘线, 最后两条延长线的垂直线夹角即为cobb角(等于两条延长线的夹角)
    - cobb角度是脊柱侧弯的重要判断依据,通常cobb角大于10度会被定义为脊柱侧弯

  <table>
    <tr>
      <th>cobb角的定义</th>
      <th>原图</th>
      <th>CLAHE数据增强</th>
    </tr>
    <tr>
      <td><img src="/assets/images/AI_assisted_diagnose/cobb角png.png" alt="cobb角png" style="zoom:30%;" /></td>
      <td><img src="/assets/images/AI_assisted_diagnose/原图.png" alt="原图" style="zoom:30%;" /></td>
      <td><img src="/assets/images/AI_assisted_diagnose/CLAHE数据增强.png" alt="CLAHE数据增强" style="zoom:30%;" /></td>
    </tr>
  </table>

  - <span id="处理流程">处理流程</span>
    - 原图 --> CLAHE数据增强 --> MaskRCNN实例分割 --> 外缘点转换 --> 后处理
  
  <table>
    <tr>
      <th>MaskRCNN结果</th>
      <th>外缘点转换</th>
      <th>后处理</th>
    </tr>
    <tr>
      <td><img src="/assets/images/AI_assisted_diagnose/MaskRCNN结果.png" alt="MaskRCNN结果" style="zoom:48%;" /></td>
      <td><img src="/assets/images/AI_assisted_diagnose/外缘点转换结果.png" alt="外缘点转换结果" style="zoom:48%;" /></td>
      <td><img src="/assets/images/AI_assisted_diagnose/脊柱侧弯后处理-9520406.png" alt="脊柱侧弯后处理" style="zoom:60%;" /></td>
    </tr>
  </table>

  - <span id="训练MaskRCNN">训练MaskRCNN实例分割模型</span>
    - MaskRCNN是基于二阶段的实例分割框架, 通过RPN层网络筛选出前景和背景,其中前景指的是图片中有价值的目标,再对筛选出来的前景信息进行分类、定位和分割任务
    - 卷积方式替换为可变形卷积
    - 骨干网络采用先进的HRNet
    - CascadeRCNN的方式做输出
    <img src="/assets/images/AI_assisted_diagnose/MaskRCNN-9519528.png" alt="MaskRCNN" style="zoom:70%;margin:2em auto;" />{:.align-center}
  - <span id="后处理">后处理</span>
    - 椎体间夹角为上椎体的上缘线和下椎体的下缘线之间的角度
    - 计算每个椎体与其他所有椎体的夹角, 最大的角即为cobb角
  - <span id="结果评估">结果评估</span>
    - 1000张测试图片找到脊柱侧弯97个, 医生保留97个, 新增0个, 召回率100%, 假阳率0%
    - 选取了248张图片, 由两位医生进行手工标注, 医生之间的平均绝对误差在2.2度，算法和医生之间的平均绝对误差在3.32度
  - <span id="结果展示">结果展示</span>
    <img src="/assets/images/AI_assisted_diagnose/脊柱侧弯结果展示.png" alt="脊柱侧弯结果展示" style="zoom:50%;" />{:.align-center}

#### 肺结节是如何判断的?
  - 肺结节在胸部 X 射线检查中呈现为圆形的白色阴影, 大小通常约为5 毫米至30 毫米之间, 肺结节可能是癌症的前兆
  - 处理流程
    - 单侧肺部图像 --> FasterRCNN目标检测 --> 切图 --> 分类模型 --> 结果
  - 训练FasterRCNN目标检测模型
    - 整体架构与MaskRCNN基本一致, 去掉MaskRCNN的Mask分支
    - 骨干网络采用MobileNetV3, 相比HRNet, 算力要求更低, 推理速度更快
  
  <table>
    <tr>
      <th>单侧肺部图像</th>
      <th>目标检测结果</th>
      <th>切图后分类</th>
    </tr>
    <tr>
      <td><img src="/assets/images/AI_assisted_diagnose/单侧肺部图像.png" alt="单侧肺部图像" style="zoom:48%;" /></td>
      <td><img src="/assets/images/AI_assisted_diagnose/目标检测结果-9527352.png" alt="目标检测结果" style="zoom:48%;" /></td>
      <td><img src="/assets/images/AI_assisted_diagnose/切图后分类-9527670.png" alt="切图后分类" style="zoom:100%;" /></td>
    </tr>
  </table>

#### 肺结节判定时FasterRCNN本身会对推荐区域做二分类, 为什么还要单独训练分类模型?
  - <span id="问题">问题</span>
    - 实验中发现, 在保证高召回率(98%)时, FasterRCNN的假阳率非常高(>60%), 尝试优化网络结构或者补充数据, 假阳率只有轻微下降
  - <span id="猜想">猜想</span>
    - 卷积过程中的信息丢失: 特征提取网络本身的局限性导致的
    - 周边信息及与周边的关联性信息丢失: FasterRCNN的分类器只能对RPN限定的推荐区域做分类
  - <span id="分析">分析</span>
    - 普通物体的目标检测时, 本身具有明显特征, 不容易与其他物体混淆, 并且作为独立物体与周边区域关联性不强, 这导致卷积过程的部分信息丢失和周边关联性信息丢失对结果影响不大
    - 肺结节的判定过程与普通物体不同, 卷积时的部分信息丢失和周边关联性信息丢失会导致它与其他相似结构差异过小无法区分, 这导致假阳率始终无法降低
  - <span id="证据">证据</span>
    - 咨询医生判断肺结节的过程, 除了观察可疑区域的外观特征, 还会观察目标的纹理、周围血管的分布密度和走向等特征, 甚至会结合患者的过往病例、生活作息等信息综合判断
  - <span id="解决方案">解决方案</span>
    - 自定义分类网络, 接受多尺寸多维度的输入, 作为FasterRCNN的补充
    - 模拟医生阅片方法, 提取可疑区域的几何、纹理、上下文等信息输入模型
    - FasterRCNN作为“初级筛选”, 分类模型专注于降低假阳率
    <img src="/assets/images/AI_assisted_diagnose/肺结节分类模型.png" alt="肺结节分类模型" style="zoom:48%;" />{:.align-center}
  - <span id="结果评估_">结果评估</span>
    - 1000张测试图片找到肺结节疑似病灶162个, 医生保留117个, 新增4个
    - 召回率97%, 假阳率28%, 假阳率大幅下降

------

### 备注

#### 可变形卷积(Deformable Convolutional Networks)
  - 不同于传统固定尺寸的卷积核,可变卷积的尺寸会跟随特征信息进行改变
  - 对物体的形变和尺度建模的能力比较强
  - 感受野比一般卷积大很多
  - 对目标的形状和大小更加敏感
  <img src="/assets/images/AI_assisted_diagnose/可变形卷积-9526064.png" alt="可变形卷积" style="zoom:67%;" />{:.align-center}

#### HRNet
  - 不同于resnet系列从高分辨率到低分辨率,低维到高纬的特点作为特征提取的方式, HRnet能在整个过程中保持高分辨率的表征. 专门用于提高分割的精确度
  - 考虑到我们X射线的图片尺寸通常在2000*2000以上的分辨率,除了输入的resize尺寸调整, 整个特征提取器保持高分辨率的表征可以提高分割的精准度
  <img src="/assets/images/AI_assisted_diagnose/HRNet.png" alt="HRNet" style="zoom:50%;" />{:.align-center}

#### MobileNet
  - 应用深度可分离卷积、反残差结构、注意力机制等优化, 在保证准确率的前提下大幅降低特征提取的计算量, 对简单目标(肺部区域)有足够好的表现同时降低显存占用
  <img src="/assets/images/AI_assisted_diagnose/MobileNet.png" alt="MobileNet" style="zoom:48%;" />{:.align-center}

#### CascadeRCNN
  - 只有proposal自身的阈值和训练器训练用的阈值较为接近的时候，训练器的性能才最好. 而CascadeRCNN[2]采用3层迭代式box回归, 能够提高检测效果
  <img src="/assets/images/AI_assisted_diagnose/CascadeRCNN.png" alt="CascadeRCNN" style="zoom:48%;" />{:.align-center}

