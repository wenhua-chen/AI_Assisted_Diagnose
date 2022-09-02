import sys,os
import cv2
import numpy as np
import pandas as pd
import _init_paths
from mobilenet_faster_rcnn.lib.model.config import cfg
from mobilenet_faster_rcnn.lib.model.test import im_detect
from mobilenet_faster_rcnn.lib.nets.mobilenet_v1 import mobilenetv1
import tensorflow as tf
from dr1_10_8in1_crop_formal import init_sess,find_model
import shutil
from tqdm import tqdm
from timeit import default_timer as timer
import math

def truncate(number, digits):
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

def init_sess_detect(tfmodel,num_class=3):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    tfconfig = tf.ConfigProto(gpu_options=gpu_options)
    tfconfig.gpu_options.allow_growth=True

    # init session
    # sess = tf.Session(config=tfconfig)
    sess = tf.Session()
    # load network
    net = mobilenetv1()
    net.create_architecture("TEST", num_class, tag='default', anchor_scales=[4,8,16,32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)
    return sess, net

def cal_intersect(boxA, boxB):
    if len(boxA)==0 or len(boxB)==0: return 0
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return max(interArea/boxBArea, interArea/boxAArea)

def drop_inside_bboxes(boxes, clss,iou_the=0.3):
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

def expand_box(original_SZG,img_w,w_ratio=0.8):
        [x1,y1,x2,y2] = original_SZG
        w = x2-x1
        x1 -= int(w*w_ratio)
        x1 = x1 if x1>0 else 0
        x2 += int(w*w_ratio)
        x2 = x2 if x2<img_w else img_w
        return [x1,y1,x2,y2]

def pred(sess,net,img_path,crop_box,thresh = 0.1,iou_s=0.000099):
    def add_box(boxesr_two_sides,box,index):
        if index == 0:
            boxesr_two_sides.append([int(box[0]+crop_box[0]),int(box[1]+crop_box[1]),int(box[2]+crop_box[0]),int(box[3]+crop_box[1]),truncate(box[4], 3)])
        else:
            boxesr_two_sides.append([int(box[0]+crop_box[0]+(img_w/2)),int(box[1]+crop_box[1]),int(box[2]+crop_box[0]+(img_w/2)),int(box[3]+crop_box[1]),truncate(box[4], 3)])

    boxesr_two_sides,clses_two_sides,final_return = [],[],[]
    
    if len(crop_box)==0: return boxesr_two_sides,clses_two_sides,final_return

    cls_names = ['__background__','LRGGH','WYY']
    img_original = cv2.imread(img_path)
    img_w = img_original.shape[1]
    crop_box = expand_box(crop_box,img_w)
    img_crop = img_original[crop_box[1]:crop_box[3],crop_box[0]:crop_box[2]]
    img_w = img_crop.shape[1]
    img1 = img_crop[:,:int(img_w/2)]
    img2 = img_crop[:,int(img_w/2):]

    for index, img in enumerate([img1, img2]):
        scores, boxes = im_detect(sess, net, img)
        clses,boxesr=[],[]
        for cls_ind, cls in enumerate(cls_names[1:]):
            if cls in ['WYY','meaningless']:
                continue
            output_bbox_dict = {}
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                            cls_scores[:, np.newaxis])).astype(np.float32)
            inds = np.where(dets[:, -1] >= thresh)[0]
            dets = dets[inds, :]
            output_bbox_dict[cls] = dets
            for det in dets.tolist():
                boxesr.append(det)
                clses.append(cls)
        boxesr, clses = drop_inside_bboxes(boxesr, clses,iou_the=iou_s)
        if len(boxesr) > 0:
            max_val = max([item[4] for item in boxesr])
            boxes_s = [item for item in boxesr if item[4]==max_val]
            add_box(boxesr_two_sides,boxes_s[0],index)
            clses_two_sides.append('LRGGH')

    for i in range(len(clses_two_sides)):
        dict_i={}
        cls_name=clses_two_sides[i]
        box=boxesr_two_sides[i]

        xmin,ymin,xmax,ymax,score_i = box[0],box[1],box[2],box[3],box[4]
        cord4 = [str(int(xmin))+','+str(int(ymin)),
                 str(int(xmax))+','+str(int(ymin)),
                 str(int(xmax))+','+str(int(ymax)),
                 str(int(xmin))+','+str(int(ymax))]

        dict_i['Name']=cls_name
        dict_i['Cords4']=cord4
        dict_i['Score']=score_i
        final_return.append(dict_i)

    return boxesr_two_sides, clses_two_sides,final_return

# def vis(im,boxes,cls_name,sum_dict_in=None,pred=True):
#     if pred:
#         box_color = (100,200,50)
#         txt_color = (100,200,50)
#     else:
#         box_color = (100,100,250)
#         txt_color = (100,100,250)

#     for i in range(len(boxes)):
#         box = boxes[i]
#         if pred:
#             score = str(round(box[4],2))
#             xmin,ymin,xmax,ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
#             cv2.putText(im, cls_name+' '+score, (xmin,ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, txt_color,1,cv2.LINE_AA)
#         else:
#             xmin,ymin,bw,bh = int(box[0]), int(box[1]), int(box[2]), int(box[3])
#             xmax,ymax = xmin+bw,ymin+bh
#             cv2.putText(im, cls_name, (xmin,ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, txt_color,1,cv2.LINE_AA)
#         cv2.rectangle(im, (xmin,ymin), (xmax,ymax),box_color ,1)
#     return im

def vis(img_path,boxesr,clses,shangzongge_crop,output_dir,score=0.1):
    assert len(boxesr) == len(clses)
    img = cv2.imread(img_path)
    img_w = img.shape[1]
    shangzongge_crop = expand_box(shangzongge_crop,img_w)
    [big_box_x1,big_box_y1,big_box_x2,big_box_y2] = shangzongge_crop
    cv2.rectangle(img,(big_box_x1,big_box_y1),(big_box_x2,big_box_y2),(0,255,0),2)
    for i, box in enumerate(boxesr):
        if box[4] < score:
            continue
        [x1, y1, x2, y2] = box[:4]
        cv2.putText(img, clses[i]+'_'+str(box[4]), (x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1,cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255),2)
    img_name = os.path.basename(img_path)
    cv2.imwrite(os.path.join(output_dir,img_name), img)

if __name__ == '__main__':
    thresh = 0.1
    iou_fiter = 0.1

    # load model_8in1
    tfmodel_8in1 = '/data/steven/project/Object_Detection_coastal/dr_wrapper/DR_models_configs/dr1_10_8in1_crop_formal.ckpt'
    start = timer()

    # load model_detect
    tfmodel_detect = './DR_models_configs/dr2_12_LeiRuanGuGaiHua_formal.ckpt'
    # tf.reset_default_graph()

    # set input_dir and output_dir
    # input_dir = '/data/steven/project/Object_Detection_coastal/dr_wrapper/test_data/right_dark_3_img/original'
    # output_dir = '/data/steven/project/Object_Detection_coastal/dr_wrapper/test_data/right_dark_3_img/result_LRGgaihua'
    input_dir = '/data/steven/project/Object_Detection_coastal/dr_wrapper/test_data/5_LRG_drclient_failed/temp3/'
    output_dir = '/data/steven/project/Object_Detection_coastal/dr_wrapper/test_data/5_LRG_drclient_failed/temp3_pred/'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    img_list = os.listdir(input_dir)
    for i,img_name in tqdm(enumerate(img_list)):
        img_path = os.path.join(input_dir,img_name)

        start = timer()
        tf.reset_default_graph()
        sess_8in1, net_8in1 = init_sess(tfmodel_8in1)
        elapsed_time = round(timer() - start,2)
        print('{} model_8in1 loaded {}s {}'.format('-'*20,elapsed_time,'-'*20))
        print(img_name)
        output_dict = find_model(sess_8in1, net_8in1, img_path)
        shangzongge_crop = output_dict['shangzongge_crop']
        print(shangzongge_crop)
        tf.reset_default_graph()
        sess_8in1.close()

        start = timer()
        sess_detect, net_detect = init_sess_detect(tfmodel_detect)
        elapsed_time = round(timer() - start,2)
        print('{} model_detect loaded {}s {}'.format('-'*20,elapsed_time,'-'*20))

        if len(shangzongge_crop):
            boxesr, clses,final_return = pred(sess_detect,net_detect,img_path,shangzongge_crop,thresh = thresh,iou_s=iou_fiter)
            print('_'*100)
            print('shangzongge_crop: ', shangzongge_crop)
            print('boxesr: ', boxesr, 'clses: ', clses)
            print('final_return=', final_return)
            if len(boxesr)>0:
                vis(img_path,boxesr,clses,shangzongge_crop,output_dir)
        else:
            print('!'*100)
            print(img_name, ' has no shangzongge_crop!')


