

##### 20191219

- 图片清晰模型 dr0_0_Clear_formal.py 3rd_edition, 发现有些侧视DR图像被误判为合格, 原因是训练集中侧视图不够多, 数据增强这部分图像重新训了第三版, 并且训练的时候加入了weight和weight_decay

- pth: /data/steven/project/Object_Detection_coastal/Classfication/output/1_img_qualification/T3_qualification_MNetV3_train_stage2_inchannel3_3edi/models/T3_qualification_MNetV3_train_stage2_inchannel3_3edi_mobilenet_v3_E39_F1_0.873612.pth

- 结果: 线上原来错误分类为合格的侧视图都判定为不合格了

##### 1212

- 肺结节 更新pth和config文件, 使用drL_drZ_2019_FJJ_withBIG_clear_0.5_neg数据训练

- /data/steven/project/Object_Detection_coastal/mmdetection_project/output/2_FJJ/FJJ_dataset_drL_drZ_2019_FJJ_withBIG_clear_0.5_neg/cascade_rcnn_dconv_c3_c5_r50_fpn_1x_alub/epoch_24.pth
- /data/steven/project/Object_Detection_coastal/mmdetection_project/saved_cfgs/2_FJJ/drL_drZ_2019_FJJ_withBIG_clear_0.5_neg/cascade_rcnn_dconv_c3_c5_r50_fpn_1x_alub.py

- 数据分布: drL_drZ_2019_FJJ_withBIG_clear_0.5_neg

​		name_EN   	t_bx 	v_bx 	a_bx 	t_im 	v_im 	a_im

​		FJJ       	5096	224		5320	4369	190		4559

​		WYY       	10494	2746	13240	10494	2746	13240

- result: drL_drZ_0813_combine_3200_Test


FJJ_Recall  FJJ_FalPos  FJJ_FPsPI  FJJ_Precis  FJJ_img_count  FJJ_pred_all  FJJ_pred_r  FJJ_pred_w  FJJ_gt_all  FJJ_gt_shot

0.863       0.825      	0.943       0.175          560.0         640.0       112.0       528.0       131.0        113.0



##### 1128

- 肺结节 dr2_11_FeiJieJie_formal.py 加入马乐/王林的肺结节数据, 数据量更大, 在验证集和测试集上的假阳率更低, 但是在肺肿瘤测试集上的recall下降了, 后续需要进一步优化肺肿瘤问题

- /data/steven/project/Object_Detection_coastal/mmdetection_project/output/2_FJJ/FJJ_dataset_drL_drZ_1120_6500_FJJ_mix/cascade_rcnn_dconv_c3_c5_r50_fpn_1x_alub/epoch_13.pth
- /data/steven/project/Object_Detection_coastal/mmdetection_project/saved_cfgs/2_FJJ/drL_drZ_1120_6500_FJJ_mix/cascade_rcnn_dconv_c3_c5_r50_fpn_1x_alub.py



- 数据分布: drL_drZ_1120_6500_FJJ_mix

name_EN   	t_bx 	v_bx 	a_bx 	t_im 	v_im 	a_im

WYY       	18466	1838	20304	18466	1838	20304

FJJ       	7968	319		8287	6865	288		7153

- result:
  - drL_drZ_0813_combine_3200_Test
    FJJ_Recall  FJJ_FalPos  FJJ_FPsPI  FJJ_Precis  FJJ_img_count  FJJ_pred_all  FJJ_pred_r  FJJ_pred_w  FJJ_gt_all  FJJ_gt_shot
    0.824       0.752       0.573       0.248          560.0         427.0       106.0       321.0       131.0        108.0
  - drL_drZ_1120_6500_FJJ_mix_val
    FJJ_Recall  FJJ_FalPos  FJJ_FPsPI  FJJ_Precis  FJJ_img_count  FJJ_pred_all  FJJ_pred_r  FJJ_pred_w  FJJ_gt_all  FJJ_gt_shot
    0.727       0.809      0.538       0.191         1838.0        1222.0       234.0       988.0       319.0        232.0



##### 1102

- 4合一 dr1_11_SC_FQZ_FBZ_SZGBK_formal.py 发现将4in1拆开成2个模型跟直接合并预测的结果差不多, 只要4in1进行了合适的权重设置(占比倒数)

- /data/steven/project/Object_Detection_coastal/Classfication/output/9_4in1/T10_4in1_MNetV3_train_stage2_inchannel3_BCELoss_best_weights/models/T10_4in1_MNetV3_train_stage2_inchannel3_BCELoss_best_weights_mobilenet_v3_E32.pth

- 数据分布: T10_bestF1_best_weights

9_4in1		train	val

渗出		 1307	 158

肺气肿		 545	 123

肺不张		 108	 30

上纵膈变宽	  1755	  225

纯阴性		 1802 	 190

图片总数	 5390	 700

- result: T10_bestF1_best_weights

SC 				FQZ				FBZ				SZGBK			mean_f1

recall	fp		recall	fp		recall	fp		recall	fp

0.80	0.33	0.75	0.22	0.46	0.36	0.74	0.21	0.70



##### 1031

- dr1_11_FQZ_FBZ_formal.py|dr1_12_SC_SZGBK_formal.py 将FQZ、FBZ|SC、SZGBK分别用两个二合一模型预测, 因为这两对的训练数据量比较接近, 没有数据不平衡的问题. 
- 尝试过3个或4个病种合并预测, 数据不平衡的问题对结果会有影响. 尝试过复制、random_crop、上采样、下采样等数据平衡方法, 效果不明显. 最有效的还是把数据量差异大的病种分开预测. 
- 其中FQZ_FBZ模型尝试过混入NIHCC公开数据集进行训练, 发现公开数据集的数据一致性很差, 混入训练会影响模型的结果, 放弃

- /data/steven/project/Object_Detection_coastal/Classfication/output/11_FQZ_FBZ/T2.4_FQZ_FBZ_MNetV3_train_without_outerdata_added_more_neg_val/models/T2.4_FQZ_FBZ_MNetV3_train_without_outerdata_added_more_neg_val_mobilenet_v3_E40.pth
- /data/steven/project/Object_Detection_coastal/Classfication/output/12_SC_SZGBK/T1_SC_SZGBK_MNetV3_train_without_outerdata_added/models/T1_SC_SZGBK_MNetV3_train_without_outerdata_added_mobilenet_v3_E14.pth

- 数据分布:

T2.4_without_outerdata_added_more_neg_val

11_FQZ_FBZ		train	val

单肺气肿	 	 542	 120

单肺不张	 	 106	 28

肺气肿+肺不张	   1	  2

纯阴性			1000	 550

总数			1649	 700



T1_without_outerdata_added		
12_SC_SZGBK		train	val
单渗出			 1225	 148
单上纵膈变宽	  1672	  215
渗出+上纵膈变宽	  81	  10
纯阴性			 2000	 327
总数			 4978	 700



- result: T2.4_without_outerdata_added_more_neg_val

FQZ				FBZ				mean_f1

recall	fp		recall	fp	

0.78	0.29	0.43	0.48	0.61



- result: T2.4_without_outerdata_added_more_neg_val

SC				SZGBK			mean_f1

recall	fp		recall	fp	

0.79	0.37	0.78	0.26	0.73



##### 20191028 

- dr0_0_Clear_formal.py 上一版将一小部分合格的图片判定成不合格, 原因是训练用的不合格数据包含了合格数据(不合格定义不严), 重新筛选不合格数据, 然后重新训练了分类模型
- /data/steven/project/Object_Detection_coastal/Classfication/output/1_img_qualification/T2_qualification_MNetV3_train_stage2_inchannel3_2edi/models/T2_qualification_MNetV3_train_stage2_inchannel3_2edi_mobilenet_v3_E25_F1_0.855359.pth



##### 20191024

- dr0_0_Clear_formal.py 重新上线了一个分类模型, 用于图片是否合格的判定, 0:不确定, 1:合格, 2:不合格
- /data/steven/project/Object_Detection_coastal/Classfication/output/1_img_qualification/T1_qualification_MNetV3_train_stage2_inchannel3/models/T1_qualification_MNetV3_train_stage2_inchannel3_mobilenet_v3_E39_F1_0.858925.pth

- 数据分布:

1_img_qualification	Train	val
notsure				818		91
qualified			5702	634
unqualified			1242	139
all					7762	864

- result:

​				notsure			qualified		unqualified		mean_f1
​				recall	fp		recall	fp		recall	fp	
T1_benchmark	0.57	0.22	0.98	0.05	0.95	0.05	0.86



##### 20191021 

- dr2_10_RuTouYing_formal.py 只有单侧预测出乳头影,将其修改为肺结节,模型不变



##### 20191021

- 肺气肿、肺不张、上纵膈变宽, 3合1, dr1_11_SC_FQZ_FBZ_SZGBK_formal.py 去掉渗出, 训练数据减少了接近一半, 数据稍微有点不平衡, 效果相比4合1得到提升
- /data/steven/project/Object_Detection_coastal/Classfication/output/10_3in1/T1_3in1_MNetV3_train_stage2_inchannel3_valthresh_bestf1/models/T1_3in1_MNetV3_train_stage2_inchannel3_valthresh_bestf1_mobilenet_v3_E37.pth

- 数据分布:

9_3in1		Train	val
肺气肿		 544	 121
肺不张	 	 107	 30
上纵膈变宽	  1773	  205
纯阴性		 1792	 200
all			4216	556

- result:

​				FQZ				FBZ				SZGBK			mean_f1
​				recall	fp		recall	fp		recall	fp	
T1_benchmark	0.75	0.16	0.60	0.35	0.90	0.16	0.76



##### 20191018 

- 渗出、肺气肿、肺不张、上纵膈变宽, 4合1,dr1_11_SC_FQZ_FBZ_SZGBK_formal.py, 肺气肿和肺不张数据量少, 数据严重不平衡, 效果不够好
- /data/steven/project/Object_Detection_coastal/Classfication/output/9_4in1/T5_4in1_MNetV3_train_stage2_inchannel3_valthresh_bestf1/models/T5_4in1_MNetV3_train_stage2_inchannel3_valthresh_bestf1_mobilenet_v3_E29.pth



##### 20191015 

- 胸膜增厚第二版

ID   	Name_CN      	Name_EN   box_Count  img_Count
1	无意义             WYY       3170	     3170
2	胸膜增厚           XMZH      5502	      2463
3	胸腔积液           XQJY      969	      764

- config
  - /data/steven/project/Object_Detection_coastal/mmdetection_project/saved_cfgs/5_XMZH_JY/XMZH_JY_1010_3170_crop_stage2_expand_upd_T2/cascade_rcnn_dconv_c3_c5_r50_fpn_1x_num_libra_alub.py
- pth
  - /data/steven/project/Object_Detection_coastal/mmdetection_project/output/5_XMZH_JY/XMZH_JY_dataset_XMZH_JY_1010_3170_crop_stage2_expand_upd_T2/cascade_rcnn_dconv_c3_c5_r50_fpn_1x_num_libra_alub/epoch_29.pth
- 结果
  - ​																											XMZH															JY						
    ​																								val		test		    gt516				val				test			gt96	
    ​																		epoch	FalPos	r		FalPos	r		漏	多		epoch	FalPos	r	FalPos	r	漏	多
    XMZH_JY_0924_1222_crop_stage2_T2		29	0.60	0.90	0.33	0.92	44	719		34	0.62	0.90	0.37	0.83	16	48



##### 20191014

- 器官 器官和椎体合并 959张标注
- config: /data/steven/project/Object_Detection_coastal/mmdetection_project/saved_cfgs/4_QiGuan_Series/1011_ZuiTi_QiGuan_960_T0/cascade_mask_rcnn_dconv_c3-c5_r50_fpn_1x.py
- pth  /data/steven/project/Object_Detection_coastal/mmdetection_project/output/10_QiGuan_series/ZuiTi_QiGuan_dataset_1011_ZuiTi_QiGuan_960_T0/cascade_mask_rcnn_dconv_c3-c5_r50_fpn_1x/epoch_34.pth



##### 20191010

- 乳头影 dr2_10_RuTouYing_formal 增加了客户端分布的标注图片, 提高了模型的泛化性; 修正了一个cut_thresh=0.3下的bug, in_norner_outer_bbox()不再需要outer_bbox进行过滤
- /data/steven/project/Object_Detection_coastal/mmdetection_project/output/0_RTY/RTY_dataset_RTY_1009_4000_combine_Stage2_up020_4000_T3/cascade_rcnn_dconv_c3_c5_r50_fpn_1x_num_libra_alub/epoch_34.pth
- /data/steven/project/Object_Detection_coastal/mmdetection_project/saved_cfgs/0_RTY/RTY_1009_4000_combine_Stage2_up020_4000_T3/cascade_rcnn_dconv_c3_c5_r50_fpn_1x_num_libra_alub.py



##### 20190927

- 胸膜增厚,积液 dr2_13_XMZH_JY_formal: thresh 0.3

- /data/steven/project/Object_Detection_coastal/mmdetection_project/output/5_XMZH_JY/XMZH_JY_dataset_XMZH_JY_0924_1222_crop_stage2_T0/cascade_rcnn_dconv_c3_c5_r50_fpn_1x_num_libra_alub/epoch_33.pth


- XMZH_JY_0924_1222_crop_stage2_T1 epoch 33		

name_EN   	t_bx 	v_bx 	a_bx 	t_im 	v_im 	a_im

XMZH      	2636	194	2830	2096	148	2244

XQJY      	480	40	520	480	40	520

WYY       	3730	210	3940	3730	210	3940

- 胸膜增厚 

val	 0.56	0.90	
test 0.53	0.83 (0.3)

- 积液

val	 0.28	0.90
test 0.33	0.803 83 (0.3)



##### 20190925

- 肋骨 dr1_1_LeiGu_formal 用mobilenetv1训练了一个找肋骨的目标检测模型代替原有的重量级mask_rcnn实例分割模型, 因为找肋骨的目的是为了确定第2、第4根肋骨的下沿线进行肺的上中下野定位, 轻量级的目标检测即可完成, 显存下降7-8G
- /data/chenwh/mobilenet_faster_rcnn/output/mobile/11_leigu/LG_6cls_clahe40_mixed_train/default/mobile_faster_rcnn_iter_80000.ckpt

09_23_6cls_clahe40_mixed							
Name_CN  name_EN   	t_bx 	v_bx 	a_bx 	t_im 	v_im 	a_im 
左肋骨_1	ZLG_1	1683	297		1980	1683	297		1980
左肋骨_2	ZLG_2	1683	297		1980	1683	297		1980
左肋骨_3	ZLG_3	1683	297		1980	1683	297		1980
左肋骨_4	ZLG_4	1683	297		1980	1683	297		1980
左肋骨_5	ZLG_5	1683	297		1980	1683	297		1980
左肋骨_6	ZLG_6	1683	297		1980	1683	297		1980
右肋骨_1	YLG_1	1683	297		1980	1683	297		1980
右肋骨_2	YLG_2	1683	297		1980	1683	297		1980
右肋骨_3	YLG_3	1683	297		1980	1683	297		1980
右肋骨_4	YLG_4	1683	297		1980	1683	297		1980
右肋骨_5	YLG_5	1683	297		1980	1683	297		1980
右肋骨_6	YLG_6	1683	297		1980	1683	297		1980



- 肺结节 dr2_11_FeiJieJie_formal 添加肺肿瘤标注(大号肺结节), 原有的肺结节标注及参数不变, 重新训练了模型, 目的是解决漏诊肺肿瘤的情况. 目前出现了一些肺肿瘤假阳, 待解决

- /data/steven/project/Object_Detection_coastal/mmdetection_project/output/2_FJJ/FJJ_dataset_drL_drZ_0921_5100_FJJ_FZK_T0/cascade_rcnn_dconv_c3_c5_r50_fpn_1x_alub/epoch_11.pth

​	val 0.86	0.78	
​	test 0.81	0.82	
​	testset:
​		/data/steven/project/Object_Detection_coastal/dataser_raw/3_feijiejie/COCOjson/train_val/drL_drZ_0813_combine_3200_stage2_T0_test.json
​		thresh:0.1, gt:131, miss:23, extra:477
​		thresh:0.2, gt:131, miss:31, extra:292

- drL_drZ_0921_5100_FJJ_FZK_T0						

name_EN   	t_bx 	v_bx 	a_bx 	t_im 	v_im 	a_im
WYY       	11161	1374	12535	11161	1374	12535
FJJ       	5077	210		5287	4349	189		4538



##### 20190918

- 乳头影 dr2_10_RuTouYing_formal rutouying_crop框太上了, 导致较多误诊, 修改8in1模型up0.2,down0(原up0.52,down0.13), 并且拿cut_thresh_w=0.3裁剪之后的图片去训练(原训练图片没有经过cut_thresh裁剪)
- /data/steven/project/Object_Detection_coastal/mmdetection_project/output/0_RTY/RTY_dataset_RTY_0829_2400_AIcombine_Stage2_up020_T8/cascade_rcnn_dconv_c3_c5_r50_fpn_1x_num_libra_alub/epoch_30.pth

val  0.19 0.93

test 0.23 0.92

testset:

​	/data/steven/project/Object_Detection_coastal/dataser_raw/1_rutouying/COCOjson/train_val/RTY_0829_2400_AIcombine_T6_test.json

​	thresh 0.1

​	gt:151 miss:11 extra:42



- RTY_0829_2400_AIcombine_Stage2_up020_T8						

name_EN   	t_bx 	v_bx 	a_bx 	t_im 	v_im 	a_im
RTY       	2967	360		3327	2967	360		3327
WYY       	5752	842		6594	5752	842		6594

- 斑片影 dr2_3_BanPianYing_formal 增加alub提高泛化能力, 增加数据量, 总数据量达到3300
- /data/steven/project/Object_Detection_coastal/mmdetection_project/output/1_BPY/BPY_dataset_BPY_0913_3300_crop_stage2_T1/cascade_rcnn_dconv_c3_c5_r50_fpn_1x_alub/epoch_21.pth
  val  0.85	0.8	
  test 0.84	0.83
  testset:
  	/data/steven/project/Object_Detection_coastal/dataser_raw/2_banpianying/COCOjson/train_val/BPY_0903_2800_crop_stage2_T1_test.json
  	thresh 0.1 gt:192 miss:32 extra:994

- BPY_0913_3300_crop_stage2_T1						

name_EN   	t_bx 	v_bx 	a_bx 	t_im 	v_im 	a_im
WYY       	6604	308		6912	6604	308		6912
BPY       	4627	250		4877	3334	186		3520



##### 20190911

- 血管壁钙化
  - dr2_5_XueGuanBiGaiHua_formal
  - /data/steven/project/Object_Detection_coastal/mmdetection_project/output/3_XGBGH/XGB_dataset_XGBGH_0820_2900_2stage_T0a_Alltrain/cascade_rcnn_dconv_c3_c5_r50_fpn_1x_num_libra_alub/epoch_22.pth

​	val	0.28	0.89

​	test 0.37	0.90

​	XGBGH_0820_2900_2stage_T0a_Alltrain						

​	name_EN   	t_bx 	v_bx 	a_bx 	t_im 	v_im 	a_im

​	XGBGH     	4095	343		4438	3233	270		3503

​	WYY       	6756	564		7320	6756	564		7320

- 乳头影 
  - dr2_10_RuTouYing_formal 训练的时候增加alub处理, 随机改变明暗度, 提高模型泛化能力, 增加了迭代次数, 调整了学习率衰减策略
  - /data/steven/project/Object_Detection_coastal/mmdetection_project/output/0_RTY/RTY_dataset_RTY_0829_2400_AIcombine_Stage2_T6/cascade_rcnn_dconv_c3_c5_r50_fpn_1x_num_libra_alub_2x/epoch_24.pth

​	val  0.18	0.88

​	test 0.16	0.88

​	testset:

​		/data/steven/project/Object_Detection_coastal/dataser_raw/1_rutouying/COCOjson/train_val/RTY_0829_2400_AIcombine_T6_test.json

​		thresh 0.1

​		with corner_outer_filter

​		gt:151 miss:18 extra:25

	RTY_0829_2400_AIcombine_Stage2_T6 expand						
	name_EN   	t_bx 	v_bx 	a_bx 	t_im 	v_im 	a_im
	WYY       	8404	842		9246	8404	842		9246
	RTY       	4527	376		4903	4527	376		4903

- 肺结节
  - dr2_11_FeiJieJie_formal
  - /data/steven/project/Object_Detection_coastal/mmdetection_project/output/2_FJJ/FJJ_dataset_drL_drZ_0905_4200_allAI_pred_T3/cascade_rcnn_dconv_c3_c5_r50_fpn_1x_alub/epoch_16.pth

​	val  0.90   0.73

​	test 0.82	0.79

​	testset:

​		/data/steven/project/Object_Detection_coastal/dataser_raw/3_feijiejie/COCOjson/train_val/drL_drZ_0813_combine_3200_stage2_T0_test.json

​		thresh 0.1 gt:131 miss:27 extra:473

​		thresh 0.2 gt:131 miss:31 extra:311

	drL_drZ_0905_4200_allAI_pred_T3						
	name_EN   	t_bx 	v_bx 	a_bx 	t_im 	v_im 	a_im
	FJJ       	3431	148		3579	2967	135		3102
	WYY       	8520	1256	9776	8520	1256	9776

- 斑片影
  - dr2_3_BanPianYing_formal 增加alub
  - /data/steven/project/Object_Detection_coastal/mmdetection_project/output/1_BPY/BPY_dataset_BPY_0903_2800_crop_stage2_T1/cascade_rcnn_dconv_c3_c5_r50_fpn_1x_alub/epoch_30.pth

​	val  0.76	0.68	

​	test 0.75	0.72

​	testset:

​		/data/steven/project/Object_Detection_coastal/dataser_raw/2_banpianying/COCOjson/train_val/BPY_0903_2800_crop_stage2_T1_test.json

​		thresh 0.1 gt:192 miss:54 extra:461

	BPY_0903_2800_crop_stage2_T1						
	name_EN   	t_bx 	v_bx 	a_bx 	t_im 	v_im 	a_im
	WYY       	1866	308		2174	1866	308		2174
	BPY       	1554	204		1758	1272	159		1431

- 肋软骨钙化
  - dr2_12_LeiRuanGuGaiHua_formal
  - /data/chenwh/mobilenet_faster_rcnn/output/mobile/10_leiruangugaihua/0910_500_LRGgaihua/default/mobile_faster_rcnn_iter_30000.ckpt

​	val  0.25	0.982

​	testset:

​		/data/steven/project/Object_Detection_coastal/dataser_raw/11_LRG_gaihua/COCOjson/train_val/0910_500_LRGgaihua_val.json

​		thresh 0.1 gt:192 miss:3 extra:58

	0910_500_LRGgaihua_T0						
	name_EN   	t_bx 	v_bx 	a_bx 	t_im 	v_im 	a_im 
	WYY       	1818	232	2050	1818	232	2050
	LRGGH     	1303	171	1474	1299	171	1470



##### 20190905

- 斑片影
  - dr2_3_BanPianYing_formal
  - /data/steven/project/Object_Detection_coastal/mmdetection_project/output/1_BPY/BPY_dataset_BPY_0903_2800_crop_stage2_T1/cascade_rcnn_dconv_c3_c5_r50_fpn_1x_num_libra/epoch_18.pth

​	val 0.75	0.64	

​	test 0.73	0.64	

​	testset: 

​		'/data/steven/project/Object_Detection_coastal/dataser_raw/2_banpianying/COCOjson/train_val/BPY_0903_2800_crop_stage2_T1_test.json'

​		thresh 0.1 

​		gt:192 miss:69 extra:347

	BPY_0903_2800_crop_stage2_T1						
	name_EN   	t_bx 	v_bx 	a_bx 	t_im 	v_im 	a_im
	WYY       	1866	308		2174	1866	308		2174
	BPY       	1554	204		1758	1272	159		1431

- 乳头影
  - dr2_10_RuTouYing_formal
  - /data/steven/project/Object_Detection_coastal/mmdetection_project/output/0_RTY/RTY_dataset_RTY_0829_2400_AIcombine_Stage2_T6/cascade_rcnn_dconv_c3_c5_r50_fpn_1x_num_libra/epoch_23.pth

​	val 0.23	0.91

​	test 0.31	0.91

​	testset:

​		/data/steven/project/Object_Detection_coastal/dataser_raw/1_rutouying/COCOjson/train_val/RTY_0829_2400_AIcombine_T6_test.json

​		thresh 0.2 

​		with corner_outer_filter

​		gt:151 miss:14 extra:63

	RTY_0829_2400_AIcombine_Stage2_T6 expand						
	name_EN   	t_bx 	v_bx 	a_bx 	t_im 	v_im 	a_im
	WYY       	8404	842		9246	8404	842		9246
	RTY       	4527	376		4903	4527	376		4903



##### 20190822

- dr1_10_8in1_crop_formal.ckpt 
  - 8合1模型, 共用了650张图片
  - 相比前一版增加了50多张新图片及标注进行训练, 
  - 微调了左右肺, 左右肺门及乳头影的bbox框位置, 使更加准确
  - 上一版有些标注肺门在同一侧, 或者bbox缺失, 这次都补上了

- 肺结节
  - dr2_11_FeiJieJie_formal 
  - /data/steven/project/Object_Detection_coastal/mmdetection_project/output/2_FJJ/FJJ_drLin_1068_7_22_drL_drZ_0813_combine_3200_stage2_random_erasing_T1/cascade_rcnn_dconv_c3_c5_r50_fpn_1x/epoch_5.pth
  - 使用 drL_drZ_0813_combine_3200_stage2_random_erasing_T1 epoch5 

​	val 0.845	0.733

​	test 0.884	0.824  

​	thre 0.1 input large img_scale=(1900, 1200),

	name_EN   	t_bx 	v_bx 	a_bx 	t_im 	v_im 	a_im
		WYY       	9288	776	10064	9288	776	10064
		FJJ       	3860	240	4100	3340	211	3551

- 班片影
  - dr2_3_BanPianYing_formal
  - /data/steven/project/Object_Detection_coastal/mmdetection_project/output/1_BPY/BPY_DRl_1799_7_31_1946_0817_stage2_crop_T1/cascade_rcnn_dconv_c3_c5_r50_fpn_1x/epoch_5.pth
  - 使用 BPY_1946_0817_crop_stage2_comb_T1 cascade_rcnn_dconv_c3_c5_r50_fpn_1x_exg epoch5

​	val 0.91	0.81

​	test 0.906	0.791	

	name_EN   	t_bx 	v_bx 	a_bx 	t_im 	v_im 	a_im
	BPY       	2628	105	2733	2145	80	2225
	WYY       	2516	452	2968	2516	452	2968

- 乳头影
  - dr2_10_RuTouYing_formal
  - /data/steven/project/Object_Detection_coastal/mmdetection_project/output/0_RTY/RTY_drLin_805_7_22_stage2_with_clahe_0803_2000_half_8in1_T0/cascade_rcnn_dconv_c3-c5_r50_fpn_1x/epoch_24.pth
  - 使用 RTY_drLin_805_7_22_stage2_with_clahe_0803_2000_half_8in1_T0 epoch24

​	val 0.20	0.91

​	test 0.205	0.848

	name_EN   	t_bx 	v_bx 	a_bx 	t_im 	v_im 	a_im
	WYY       	11160	700	11860	11160	700	11860
	RTY       	4103	278	4381	4103	278	4381

- 血管壁钙化
  - dr2_5_XueGuanBiGaiHua_formal
  - /data/steven/project/Object_Detection_coastal/mmdetection_project/output/3_XGBGH/XGB_drLin_904_Stage2_XGBGH_0820_2900_2stage_T0c_notAlltrain/cascade_rcnn_dconv_c3_c5_r50_fpn_1x/epoch_7.pth'
  - 使用XGBGH_0820_2900_2stage_T0c_notAlltrain _new8in1 epoch7

​	val	0.39	0.92

​	test 0.521	0.922

	name_EN   	t_bx 	v_bx 	a_bx 	t_im 	v_im 	a_im
		XGBGH     	2266	376	2642	1752	294	2046
		WYY       	3681	616	4297	3681	616	4297