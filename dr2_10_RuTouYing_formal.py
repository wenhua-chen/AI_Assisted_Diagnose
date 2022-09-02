import os
import numpy as np
import pandas as pd
import cv2
import math
import mmcv
import torch
from mmdet.apis import init_detector, inference_detector

def truncate(number, digits):
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

def cal_intersect(boxA, boxB):
    if len(boxA)==0 or len(boxB)==0: return 0
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return max(interArea/boxBArea, interArea/boxAArea)

def drop_inside_bboxes(boxes, clss,iou_the=0.3):
    '''
    boxes: [[2187, 666, 2238, 723, 0.56], [2070, 602, 2185, 759, 0.56], [572, 1462, 709, 1588, 0.7], [2312, 1564, 2482, 1671, 0.9]]
    cls = ['BanPianYing', 'BanPianYing', 'BanPianYing', 'BanPianYing']
    '''
    assert len(boxes) == len(clss)
    d = {}
    for i in range(len(boxes)):
        d[str(boxes[i][0:4])] = clss[i]

    original_bboxes = boxes[:]
    boxes.sort(key=lambda x: x[4], reverse=True)
    group = np.array(boxes)

    out_array = []
    out_cls = []
    while len(group) != 0:
        del_idx = []
        for i, item in enumerate(group):
            min_intersection_ratio = cal_intersect(group[0][0:4], group[i][0:4])
            if min_intersection_ratio >= iou_the:
                del_idx.append(i)
        out_array.append([x for x in group[0][0:4].tolist()] + [group[0][-1]])
        group = np.delete(group, del_idx, axis=0)

    for element in out_array:
        out_cls.append(d[str(element[0:4])])
    return out_array, out_cls

def in_norner_outer_bbox(img,box,cut_thresh,index,w_ratio=0.1,h_ratio=0.5):
    [w_thresh,h_thresh] = cut_thresh
    [h,w,_] = img.shape
    [x1,y1,x2,y2] = box[:4]
    # if y1 < h_thresh*h:
    #     return False
    # elif (index==0 and x2<(1-w_thresh)*w) or (index==1 and x1>w_thresh*w):
    #     return True
    # return False
    if index==0 and x1<w*w_ratio*10/7 and y1<h*h_ratio:
        return True
    elif index==1 and x2>(1-w_ratio*10/7)*w and y1<h*h_ratio:
        return True
    else:
        return False

# pred small box
def pred(model,img_path,crop,cut_thresh=[0.3,0],thresh=0.1,iou_s=0.1):
    cls_names = ['RTY','WYY']
    [rutouying_crop,right_feimen_crop,left_feimen_crop] = crop
    img_original = cv2.imread(img_path)
    img_crop = img_original[rutouying_crop[1]:rutouying_crop[3],rutouying_crop[0]:rutouying_crop[2]]
    img_w = img_crop.shape[1]
    img1 = img_crop[:,:int(img_w/2*(1-cut_thresh[0]))]
    img2 = img_crop[:,int(img_w/2*(1+cut_thresh[0])):]
    boxesr_two_sides = []
    clses_two_sides = []
    def restore_coordinate(boxes_s_original,index):
        boxes_s_restore = []
        for box in boxes_s_original:
            if index==0:
                boxes_s_restore.append([int(box[0]+rutouying_crop[0]),int(box[1]+rutouying_crop[1]),int(box[2]+rutouying_crop[0]),int(box[3]+rutouying_crop[1]),truncate(box[4], 3)])
            elif index==1:
                boxes_s_restore.append([int(box[0]+rutouying_crop[0]+img_w/2*(1+cut_thresh[0])),int(box[1]+rutouying_crop[1]),int(box[2]+rutouying_crop[0]+img_w/2*(1+cut_thresh[0])),int(box[3]+rutouying_crop[1]),truncate(box[4], 3)])
        return boxes_s_restore

    def add_box(boxesr_two_sides,box,index):
        if index == 0:
            boxesr_two_sides.append([int(box[0]+rutouying_crop[0]),int(box[1]+rutouying_crop[1]),int(box[2]+rutouying_crop[0]),int(box[3]+rutouying_crop[1]),truncate(box[4], 3)])
        else:
            boxesr_two_sides.append([int(box[0]+rutouying_crop[0]+img_w/2*(1+cut_thresh[0])),int(box[1]+rutouying_crop[1]),int(box[2]+rutouying_crop[0]+img_w/2*(1+cut_thresh[0])),int(box[3]+rutouying_crop[1]),truncate(box[4], 3)])
    
    for index, img in enumerate([img1, img2]):
        result = inference_detector(model, img)
        bbox_result, segm_result = result, None
        # bboxes_np = np.vstack(bbox_result)
        bboxes_np = np.concatenate(bbox_result)
        labels_np = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
            ]
        # labels_np = np.concatenate(labels_np)
        labels_np = np.hstack(labels_np)
        inds = np.where(bboxes_np[:, -1] >= thresh)[0]
        boxes_s = bboxes_np[inds, :].tolist()
        clses_idx = labels_np[inds].tolist()
        clses = [cls_names[i] for i in clses_idx]
        boxes_s_original, clses_original = drop_inside_bboxes(boxes_s, clses,iou_the=iou_s)
        boxes_s_restore = restore_coordinate(boxes_s_original,index)
        boxes_temp = []
        clses_temp = []
        for i, clse in enumerate(clses_original):
            if clse not in ['WYY','meaningless'] and cal_intersect(boxes_s_restore[i],right_feimen_crop)==0 and cal_intersect(boxes_s_restore[i],left_feimen_crop)==0:
                boxes_temp.append(boxes_s_original[i])
                clses_temp.append(clses_original[i])
        boxes_inside = []
        for box in boxes_temp:
            if not in_norner_outer_bbox(img,box,cut_thresh,index):
                boxes_inside.append(box)
        if len(boxes_inside) > 0:
            max_val = max([item[4] for item in boxes_inside])
            boxes_s = [item for item in boxes_inside if item[4]==max_val]
            add_box(boxesr_two_sides,boxes_s[0],index)
            clses_two_sides.append('RTY')
    if len(clses_two_sides) == 1:
        clses_two_sides[0] = 'FeiJieJie'
    return boxesr_two_sides, clses_two_sides
    
def vis(img_path,boxesr,clses,crop,output_dir,cut_thresh=[0.3,0],corner_ratio=[0.1,0.5],score=0.1):
    assert len(boxesr) == len(clses)
    [rutouying_crop,right_feimen_crop,left_feimen_crop] = crop
    [big_box_x1,big_box_y1,big_box_x2,big_box_y2] = rutouying_crop
    w,h = big_box_x2-big_box_x1,big_box_y2-big_box_y1
    [w_thresh,h_thresh] = cut_thresh
    [w_ratio,h_ratio] = corner_ratio
    img = cv2.imread(img_path)
    cv2.rectangle(img,(big_box_x1,big_box_y1),(big_box_x2,big_box_y2),(255,0,0),2)
    if len(right_feimen_crop)>0:
        cv2.rectangle(img,(right_feimen_crop[0],right_feimen_crop[1]),(right_feimen_crop[2],right_feimen_crop[3]),(0,255,255),2)
    if len(left_feimen_crop)>0:
        cv2.rectangle(img,(left_feimen_crop[0],left_feimen_crop[1]),(left_feimen_crop[2],left_feimen_crop[3]),(0,255,255),2)
    cv2.rectangle(img,(int(big_box_x1),int(big_box_y1+h_thresh*h)),(int(big_box_x1)+int((1-w_thresh)*w*0.5),int(big_box_y2)),(0,255,0),2)
    cv2.rectangle(img,(int(big_box_x2-(1-w_thresh)*w*0.5),int(big_box_y1+h_thresh*h)),(int(big_box_x2),int(big_box_y2)),(0,255,0),2)
    cv2.rectangle(img,(int(big_box_x1),int(big_box_y1)),(int(big_box_x1+w_ratio*w*0.5),int(big_box_y1+h_ratio*h)),(0,255,255),2)
    cv2.rectangle(img,(int(big_box_x2-w_ratio*w*0.5),int(big_box_y1)),(int(big_box_x2),int(big_box_y1+h_ratio*h)),(0,255,255),2)
    for i, box in enumerate(boxesr):
        if box[4] < score:
            continue
        [x1, y1, x2, y2] = box[:4]
        cv2.putText(img, clses[i]+'_'+str(box[4]), (x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1,cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255),2)
    img_name = os.path.basename(img_path)
    cv2.imwrite(os.path.join(output_dir,img_name), img)

if __name__ == '__main__':
    from timeit import default_timer as timer
    import json
    import shutil
    from tqdm import tqdm
    from pycocotools.coco import COCO

    config_file = '/data/steven/project/Object_Detection_coastal/dr_wrapper/DR_models_configs/dr2_10_RuTouYing_formal_cfg.py'
    pth_model = '/data/steven/project/Object_Detection_coastal/dr_wrapper/DR_models_configs/dr2_10_RuTouYing_formal.pth'
    
    # try_time = 'RTY_1009_4000_combine_Stage2_up020_4000_T3'
    # config_file = '/data/steven/project/Object_Detection_coastal/mmdetection_project/saved_cfgs/0_RTY/{}/cascade_rcnn_dconv_c3_c5_r50_fpn_1x_num_libra_alub.py'.format(str(try_time))
    # pth_model = '/data/steven/project/Object_Detection_coastal/mmdetection_project/output/0_RTY/RTY_dataset_{}/cascade_rcnn_dconv_c3_c5_r50_fpn_1x_num_libra_alub/epoch_34.pth'.format(str(try_time))

    # config_file = '/data/steven/project/Object_Detection_coastal/mmdetection_project/saved_cfgs/0_RTY/RTY_0829_2400_AIcombine_Stage2_up020_T8/cascade_rcnn_dconv_c3_c5_r50_fpn_1x_num_libra_alub.py'
    # pth_model = '/data/steven/project/Object_Detection_coastal/mmdetection_project/output/0_RTY/RTY_dataset_RTY_0829_2400_AIcombine_Stage2_up020_T8/cascade_rcnn_dconv_c3_c5_r50_fpn_1x_num_libra_alub/epoch_30.pth'

    # config_file = '/data/steven/project/Object_Detection_coastal/mmdetection_project/saved_cfgs/0_RTY/RTY_0803_2000_combine_Stage2_combine_clahe3_8in1_T1/cascade_rcnn_dconv_c3-c5_r50_fpn_1x.py'
    # pth_model = '/data/steven/project/Object_Detection_coastal/mmdetection_project/output/0_RTY/RTY_drLin_805_7_22_stage2_with_clahe_0803_2000_half_8in1_T0/cascade_rcnn_dconv_c3-c5_r50_fpn_1x/{}.pth'.format(epoch)

    # pth_model = './output/0_RTY/RTY_drLin_805_7_22_stage2_with_clahe_0803_2000_half_8in1_T0/cascade_rcnn_dconv_c3-c5_r50_fpn_1x/epoch_22.pth'
    # out_dir = './inerence_cal_test_check/0_RTY/RTY_drLin_805_7_22_stage2_with_clahe_0803_2000_half_8in1_T0/cascade_rcnn_dconv_c3-c5_r50_fpn_1x/thresh_{}_e22_f17_r90/'.format(str(thresh))

    # pth_model = './output/0_RTY/RTY_drLin_805_7_22_stage2_with_clahe_0803_2000_half_8in1_T0/cascade_rcnn_dconv_c3-c5_r50_fpn_1x/epoch_24.pth'
    # out_dir = './inerence_cal_test_check/0_RTY/RTY_drLin_805_7_22_stage2_with_clahe_0803_2000_half_8in1_T0/cascade_rcnn_dconv_c3-c5_r50_fpn_1x/thresh_{}_e24_f20_r91/'.format(str(thresh))

    start = timer()
    model = init_detector(config_file, pth_model, device='cuda:0')
    elapsed_time = round(timer() - start,2)
    print('{} model loaded {}s {}'.format('-'*20,elapsed_time,'-'*20))

    single_test = False
    if single_test:
        im_name = '/data/steven/project/Object_Detection_coastal/dr_wrapper/1.3.12.2.1107.5.3.33.4700.11.201901030924570375-1.jpg'
        pts1 = [350, 1105, 2417, 2028]
        boxesr, clses = pred(model,im_name,pts1,rty2fjj_thresh)
        print('_'*100)
        print(boxesr, clses)
        vis(im_name, boxesr, clses,pts1)
    else:
        import tensorflow as tf
        from dr1_10_8in1_crop_formal import init_sess,find_model
        tfmodel = '/data/steven/project/Object_Detection_coastal/dr_wrapper/DR_models_configs/dr1_10_8in1_crop_formal.ckpt'
        sess, net = init_sess(tfmodel)
        print('Loaded network {:s}'.format(tfmodel))

        # input_dir = '/data/chenwh/mobilenet_faster_rcnn/8in1_crop/0816_zhuyisheng_xiugai'
        # output_dir = '/data/steven/project/Object_Detection_coastal/dr_wrapper/test_data/2_all_left_test_data_800/2_new_8in1_and_rutouying_{}_score_{}_cut_{}_{}_model_without_symmetry'.format(str(epoch),str(score),str(cut_thresh[0]),str(cut_thresh[1]))

        # input_dir = '/data/steven/project/Object_Detection_coastal/dataser_raw/1a_AI_pred/images_raw_anno/orginal_img'
        # output_dir = '/data/steven/project/Object_Detection_coastal/dr_wrapper/test_data/2_all_left_test_data_800/2_new_8in1_and_rutouying_0905edition_with_corner_outer_filter'

        # input_dir = '/data/steven/project/Object_Detection_coastal/dr_wrapper/test_data/0906_test_dark_img/original'
        # output_dir = '/data/steven/project/Object_Detection_coastal/dr_wrapper/test_data/0906_test_dark_img/original_rutouying_result_0909_newmodel'

        # input_dir = '/data/steven/project/Object_Detection_coastal/dr_wrapper/test_data/right_dark_3_img/original'
        # output_dir = '/data/steven/project/Object_Detection_coastal/dr_wrapper/test_data/right_dark_3_img/result_rutouying'

        input_dir = '/data/steven/project/Object_Detection_coastal/dataser_raw/1_rutouying/images_raw_anno/0913_zhuyisheng_test_imgs'
        output_dir = '/data/steven/project/Object_Detection_coastal/dr_wrapper/test_data/0913_zhuyisheng_test_imgs_pred_rutouying_1021_up02_down0_e34'

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        img_list = os.listdir(input_dir)
        vis_img_count = 0
        for img_name in tqdm(img_list):
            img_path = os.path.join(input_dir,img_name)
            output_dict = find_model(sess, net, img_path)
            crop = [output_dict['rutouying_crop'],
            output_dict['right_feimen_crop'],output_dict['left_feimen_crop']]
            if len(crop[0])>0:
                boxesr, clses = pred(model,img_path,crop)
                print('_'*100)
                print('boxesr: ', boxesr, 'clses: ', clses)
                if len(boxesr)>0:
                    vis(img_path,boxesr,clses,crop,output_dir)
                    vis_img_count += 1
            else:
                print('!'*100)
                print(img_name, ' has no rutouying_crop!')
