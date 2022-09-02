import tensorflow as tf
import numpy as np
import os, cv2

from mobilenet_faster_rcnn.lib.nets.mobilenet_v1 import mobilenetv1
from mobilenet_faster_rcnn.lib.model.config import cfg
from mobilenet_faster_rcnn.lib.model.test import im_detect
import shutil

def trans2dict(boxesr,clses):
    assert len(boxesr) == len(clses)
    re_drict = {}
    for i, cls in enumerate(clses):
        re_drict[cls] = boxesr[i]
    return re_drict

def find_model(sess, net, image_name):
    def move_crop_box_rutouying(crop_box,up_ratio=0.2):
        [x1,y1,x2,y2] = crop_box
        y1 -= int(up_ratio*(y2-y1))
        y1 = y1 if y1>0 else 0
        return [x1,y1,x2,y2]

    def move_crop_box_shangzongge(crop_box,img_w,ratio=[0.2,0.3]):
        [w_ratio,h_ratio] = ratio
        if len(crop_box) == 0:
            return crop_box
        [x1,y1,x2,y2] = crop_box
        box_w,box_h = x2-x1,y2-y1
        x1 -= int(box_w*w_ratio)
        x1 = x1 if x1>0 else 0
        x2 += int(box_w*w_ratio)
        x2 = x2 if x2<img_w else img_w
        y1 -= int(box_h*h_ratio)
        y1 = y1 if y1>0 else 0
        return [x1,y1,x2,y2]

    """Detect object classes in an image using pre-computed object proposals."""
    CLASSES = ['__background__', 'rutouying_crop', 'xueguanbi_crop', 'zhudongmai_crop', 'shangzongge_crop',
                'left_lung_crop', 'right_lung_crop', 'left_feimen_crop', 'right_feimen_crop']
    # Load the demo image
    im = cv2.imread(image_name)
    img_h,img_w,_ = im.shape
    # Detect all object classes and regress object bounds
    scores, boxes = im_detect(sess, net, im)

    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.3
    thresh = 0.000001

    boxesr = []
    clses = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        output_bbox_dict = {}
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        inds = np.where(dets[:, -1] >= thresh)[0]
        dets = dets[inds, :]
        output_bbox_dict[cls] = dets

        num_bbox = output_bbox_dict[cls].shape[0]
        output_bbox = []
        if num_bbox != 0:
            output_bboxes = [list(output_bbox_dict[cls][i,:]) for i in range(num_bbox)]

            # select the bboxes with highest score
            max_val = max([item[4] for item in output_bboxes])
            output_bbox = [item for item in output_bboxes if item[4]==max_val]
            output_bbox = [int(item) for item in output_bbox[0]][0:4]
            if cls == 'rutouying_crop':
                output_bbox = move_crop_box_rutouying(output_bbox)
            if cls == 'shangzongge_crop':
                output_bbox = move_crop_box_shangzongge(output_bbox,img_w)
        boxesr.append(output_bbox)
        clses.append(cls)
    output_dict = trans2dict(boxesr, clses)
    return output_dict

def init_sess(tfmodel):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    tfconfig = tf.ConfigProto(gpu_options=gpu_options)
    tfconfig.gpu_options.allow_growth=True

    # init session
    # sess = tf.Session(config=tfconfig)
    sess = tf.Session()
    # load network
    net = mobilenetv1()
    net.create_architecture("TEST", 9, tag='default', anchor_scales=[4,8,16,32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)
    return sess, net

def vis(img_path,im_name_out,output_dict):
    im = cv2.imread(img_path)
    img_name = img_path.split('/')[-1]
    for cls in output_dict.keys():
        if len(output_dict[cls]) > 0:
            [x1, y1, x2, y2] = output_dict[cls]
            cv2.putText(im, cls, (x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1,cv2.LINE_AA)
            cv2.rectangle(im, (x1, y1), (x2, y2), (0,0,255),2)
    cv2.imwrite(im_name_out, im)

if __name__ == '__main__':
    import pandas as pd
    from tqdm import tqdm
    # os.environ["CUDA_VISIBLE_DEVICES"]="0"
    # gpu_options = tf.GPUOptions(allow_growth=True)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    tfmodel = '/data/steven/project/Object_Detection_coastal/dr_wrapper/DR_models_configs/dr1_10_8in1_crop_formal.ckpt'
    sess, net = init_sess(tfmodel)
    print('Loaded network {:s}'.format(tfmodel))

    single_test = False
    if single_test:
        raise NotImplementedError
    else:
        # im_name = '/data/steven/project/Object_Detection_coastal/dr_wrapper/1.3.12.2.1107.5.3.33.4700.11.201901030924570375-1.jpg'
        input_dir = '/data/steven/project/Object_Detection_coastal/dr_wrapper/test_data/3_ruwan_hostipal_data_1121/1_false_negtive/'
        output_dir = '/data/steven/project/Object_Detection_coastal/dr_wrapper/test_data/3_ruwan_hostipal_data_1121/1_false_negtive_pred_8in1/'

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        img_list = os.listdir(input_dir)
        for img_name in img_list:
            im_name_in = os.path.join(input_dir,img_name)
            im_name_out = os.path.join(output_dir,img_name)
            output_dict = find_model(sess, net, im_name_in)
            vis(im_name_in,im_name_out,output_dict)
            print('_'*100)
            print(img_name)
            print('output_bboxes ', output_dict)
