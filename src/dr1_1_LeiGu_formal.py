import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'

import cv2
import numpy as np
import _init_paths
from mobilenet_faster_rcnn.lib.model.config import cfg
from mobilenet_faster_rcnn.lib.model.test import im_detect
from mobilenet_faster_rcnn.lib.nets.mobilenet_v1 import mobilenetv1
import tensorflow as tf
from timeit import default_timer as timer
from tqdm import tqdm
import shutil

def init_sess(tfmodel):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    tfconfig = tf.ConfigProto(gpu_options=gpu_options)
    tfconfig.gpu_options.allow_growth=True
    sess = tf.Session(config=tfconfig)
    net = mobilenetv1()
    net.create_architecture("TEST",13,tag='default',anchor_scales=[4,8,16,32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)
    return sess, net

def trans2dict(boxesr, clses):
    cls_names = ['ZLG_2','ZLG_4','YLG_2','YLG_4']
    re_drict = {}
    for cls_name in cls_names:
        idxs = [i for i, e in enumerate(clses) if e == cls_name]
        boxesr_n = []
        if len(idxs) > 0:
            boxesr_n = [boxesr[j] for j in idxs]
        re_drict[cls_name] = boxesr_n
    return re_drict

def pred(sess, net, image_name,thresh=0.1,iou_s=0.000099):
    CLASSES=['__background__','ZLG_1','ZLG_2','ZLG_3','ZLG_4','ZLG_5','ZLG_6','YLG_1','YLG_2','YLG_3','YLG_4','YLG_5','YLG_6']
    im = cv2.imread(image_name)
    scores, boxes = im_detect(sess, net, im)
    clses,boxesr=[],[]
    for cls_ind, cls in enumerate(CLASSES[1:]):
        if cls != 'ZLG_2' and cls != 'ZLG_4' and cls != 'YLG_2' and cls != 'YLG_4':
            continue
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        inds = np.where(dets[:, -1] >= thresh)[0]
        dets = dets[inds, :]
        for det in dets.tolist():
            boxesr.append(det)
            clses.append(cls)
    pred_dict = trans2dict(boxesr, clses)
    
    cls_names = ['ZLG_2','ZLG_4','YLG_2','YLG_4']
    ye_boundry = {}
    out_box_list = []
    for i,cls_name in enumerate(cls_names):
        pred_bboxes = pred_dict[cls_name]
        if len(pred_bboxes) == 0:
            if i<2:
                symmetry_boxes = pred_dict[cls_names[i+2]]
            else:
                symmetry_boxes = pred_dict[cls_names[i-2]]
            if len(symmetry_boxes) == 0:
                return 'No_result','No_result'
            else:
                pred_bboxes =  symmetry_boxes
        if len(pred_bboxes)>1:
            output_bboxes = pred_bboxes
            max_val = max([item[4] for item in output_bboxes])
            output_bbox = [item for item in output_bboxes if item[4]==max_val]
            pred_bboxes = output_bbox

        x1,y1,x2,y2 = int(pred_bboxes[0][0]),int(pred_bboxes[0][1]),int(pred_bboxes[0][2]),int(pred_bboxes[0][3])
        string_box = [str(x1)+','+str(y1),str(x2)+','+str(y1),str(x2)+','+str(y2),str(x1)+','+str(y2)]
        out_box_list.append(string_box)

        if cls_name == 'ZLG_2':
            ye_boundry['left2'] = y2
        elif cls_name == 'ZLG_4':
            ye_boundry['left4'] = y2
        elif cls_name == 'YLG_2':
            ye_boundry['right2'] = y2
        else:
            ye_boundry['right4'] = y2
    return ye_boundry,out_box_list

def vis(im,boxes,cls_name,sum_dict_in=None,pred=True):
    if pred:
        box_color = (100,200,50)
        txt_color = (100,200,50)
    else:
        box_color = (100,100,250)
        txt_color = (100,100,250)

    for i in range(len(boxes)):
        box = boxes[i]
        if pred:
            score = str(round(box[4],2))
            xmin,ymin,xmax,ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.putText(im, cls_name+' '+score, (xmin,ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, txt_color,1,cv2.LINE_AA)
        else:
            xmin,ymin,bw,bh = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            xmax,ymax = xmin+bw,ymin+bh
            cv2.putText(im, cls_name, (xmin,ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, txt_color,1,cv2.LINE_AA)
        cv2.rectangle(im, (xmin,ymin), (xmax,ymax),(0,0,255) ,2)
    return im

if __name__ == '__main__':
    thresh = 0.1
    iou_fiter = 0.95
    pred_max_only=True
    draw_img = True
    imgFolder = '/data/steven/project/Object_Detection_coastal/dr_wrapper/test_data/0913_zhuyisheng_test_imgs'
    # out_dir =  '/data/steven/project/Object_Detection_coastal/dr_wrapper/test_data/0913_zhuyisheng_test_imgs_pred_leigu_6cls_clahe40_mixed_iter80000'
    out_dir =  '/data/steven/project/Object_Detection_coastal/dr_wrapper/temp'
    tfmodel_path = '/data/steven/project/Object_Detection_coastal/dr_wrapper/DR_models_configs/dr1_1_LeiGu_formal.ckpt'
    
    # if os.path.exists(out_dir):
    #     shutil.rmtree(out_dir)
    # os.makedirs(out_dir)

    start = timer()
    tf.reset_default_graph()
    sess, net = init_sess(tfmodel_path)
    elapsed_time = round(timer() - start,2)

    for i, im_name in tqdm(enumerate(os.listdir(imgFolder))):
        if im_name != '1.3.12.2.1107.5.3.33.4700.11.201905240848310718-1.jpg':
            continue
        in_path = os.path.join(imgFolder, im_name)
        out_path = os.path.join(out_dir, im_name)

        im = cv2.imread(in_path)
        try:
            ye_boundry,out_box_list,pred_dict = pred(sess,net,in_path,thresh = thresh,iou_s=iou_fiter)
        except:
            print(pred(sess,net,in_path,thresh = thresh,iou_s=iou_fiter))
        print('*'*200)
        print('im_name: ',im_name)
        print('ye_boundry: ',ye_boundry)
        print('out_box_list: ',out_box_list)

        cls_names = ['ZLG_2','ZLG_4','YLG_2','YLG_4']
        for cls_name in cls_names:
            pred_bboxes = pred_dict[cls_name]

            if pred_max_only and len(pred_bboxes)>1:
                output_bboxes = pred_bboxes
                max_val = max([item[4] for item in output_bboxes])
                output_bbox = [item for item in output_bboxes if item[4]==max_val]
                pred_bboxes = output_bbox
            print('pred_bboxes: ',pred_bboxes)
            if draw_img and len(pred_bboxes)>0:
                im = vis(im,pred_bboxes,cls_name,pred=True)
        if draw_img:
            cv2.imwrite(out_path, im)