import sys
sys.path.append('./tools/')
sys.path.append('./lib/')

import os, cv2
import numpy as np
import pandas as pd

import _init_paths
from model.config import cfg
from model.test import im_detect
from nets.mobilenet_v1 import mobilenetv1

import tensorflow as tf




def init_sess(tfmodel,num_class):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    tfconfig = tf.ConfigProto(gpu_options=gpu_options)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    net = mobilenetv1()
    net.create_architecture("TEST", num_class, tag='default', anchor_scales=[4,8,16,32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)
    return sess, net

def getPred(self):
    images_dict = coco.loadImgs(coco.getImgIds())
    allPred = {}
    for i, im_name in enumerate(images_dict):
        image_id = im_name['id']
        im_name = os.path.join(imgFolder, im_name['file_name'])

        im = cv2.imread(im_name)
        scores, boxes = im_detect(sess, net, im)

        output_bbox_dict = {}
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                  cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            inds = np.where(dets[:, -1] >= thresh)[0]
            dets = dets[inds, :]
            output_bbox_dict[cls] = dets

        allPred[image_id] = output_bbox_dict
        print (i, '/', len(images_dict), end='\r')

    return allPred

def cal_intersect(boxA, boxB):
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

def trans2dict(boxesr, clses,cls_names):
    re_drict = {}
    for cls_name in cls_names:
        idxs = [i for i, e in enumerate(clses) if e == cls_name]
        boxesr_n = []
        if len(idxs) > 0:
            boxesr_n = [boxesr[j] for j in idxs]
        re_drict[cls_name] = boxesr_n
    return re_drict

def pred(sess, net, image_name, CLASSES=('__background__', 'ZhuDongMai'),thresh = 0.5,iou_s=0.000099):
    """Detect object classes in an image using pre-computed object proposals."""

    im = cv2.imread(image_name)
    scores, boxes = im_detect(sess, net, im)
    clses,boxesr=[],[]
    # print('CLASSES',CLASSES)
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
       # print('cls_scores', cls_scores)
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)

        inds = np.where(dets[:, -1] >= thresh)[0]
        dets = dets[inds, :]
        for det in dets.tolist():
            boxesr.append(det)
            clses.append(cls)
    boxesr, clses = drop_inside_bboxes(boxesr, clses,iou_the=iou_s)
    pred_dict = trans2dict(boxesr, clses,cls_names)
    return pred_dict,boxesr

def trans_box(im,box,target_size):
    # print('box',box)
    height,width = im.shape[0],im.shape[1]
    xmin,ymin,xmax,ymax = box[0],box[1],box[2],box[3]
    n_h,n_w = target_size[0],target_size[1]
    # print(height,width,n_h,n_w)
    # print(box)

    new_xmin = int((xmin/width)*n_w)
    new_xmax = int((xmax/width)*n_w)
    new_ymin = int((ymin/height)*n_h)
    new_ymax = int((ymax/height)*n_h)
    # print(new_xmin,new_xmax,new_ymin,new_ymax)
    return [new_xmin,new_ymin,new_xmax,new_ymax,box[-1]]


def vis(im,boxes,cls_name,sum_dict_in=None,pred=True):
    if pred:
        box_color = (100,200,50)
        txt_color = (100,200,50)
    else:
        box_color = (100,100,250)
        txt_color = (100,100,250)


    # height,width = im.shape[0],im.shape[1]
    # if width <600:
    #     new_width  = 600
    #     new_height = int(new_width * height / width)
    # else:
    #      new_width,new_height=width,height
    # im = cv2.resize(im, (new_width, new_height), interpolation = cv2.INTER_CUBIC)

    for i in range(len(boxes)):
        box = boxes[i]
        # box = trans_box(im,box,(new_height,new_width))
        if pred:
            score = str(round(box[4],2))
            xmin,ymin,xmax,ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.putText(im, cls_name+' '+score, (xmin,ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, txt_color,1,cv2.LINE_AA)
        else:
            xmin,ymin,bw,bh = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            xmax,ymax = xmin+bw,ymin+bh
            cv2.putText(im, cls_name, (xmin,ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, txt_color,1,cv2.LINE_AA)
        cv2.rectangle(im, (xmin,ymin), (xmax,ymax),box_color ,1)
    return im

def init_InfoStats_dict():
    return {
        'pred_r':0,
        'pred_all':0,
        'Precis':0,
        'FalPos':0,
        'gt_shot':0,
        'gt_all':0,
        'Recall':0,
    }

def init_cls_stats_dict(cls_names):
    '''
    return sth like, dict with dict value:
    {'ban': {'pred_r': 0, 'pred_all': 0, 'gt_shot': 0, 'gt_all': 0},
     'chloasma': {'pred_r': 0, 'pred_all': 0, 'gt_shot': 0, 'gt_all': 0},
     'douyin': {'pred_r': 0, 'pred_all': 0, 'gt_shot': 0, 'gt_all': 0}}
    '''
    ulti_dict = {}
    ulti_dict['All'] = init_InfoStats_dict()
    for cls_name in cls_names:
        if cls_name == 'bakcground':
            continue
        ulti_dict[cls_name] = init_InfoStats_dict()
    return ulti_dict

def core_val(pred_boxes,gt_boxes,sum_dict,cls_name_in='cls_name',iou_the=0.1,gt_box_all=None):
    pred_eval_list = [0]*len(pred_boxes)
    gt_eval_list = [0]*len(gt_boxes)

    for i, gt_box in enumerate(gt_boxes):
        for j, pred_box in enumerate(pred_boxes):
            gt_box_xyxy = [gt_box[0],gt_box[1],gt_box[0]+gt_box[2],gt_box[1]+gt_box[3]]
            m_iou = cal_intersect(gt_box_xyxy, pred_box)
            if m_iou >= iou_the:
                # print('-'*10+'m_iou'+'-'*10,m_iou)
                gt_eval_list[i]=1
                pred_eval_list[j]=1
    pred_r,pred_w = pred_eval_list.count(1),pred_eval_list.count(0)
    gt_shot,gt_miss = gt_eval_list.count(1),gt_eval_list.count(0)
    assert pred_r+pred_w == len(pred_eval_list)
    assert gt_shot+gt_miss == len(gt_eval_list)

    all_dict = sum_dict['All']
    all_dict['pred_r'] += pred_r
    all_dict['pred_all'] += len(pred_boxes)
    if all_dict['pred_all'] != 0:
        all_dict['Precis'] = round(all_dict['pred_r']/all_dict['pred_all'],3)
        all_dict['FalPos'] = round((all_dict['pred_all']-all_dict['pred_r'])/all_dict['pred_all'],3)
    else:
        all_dict['Precis'] = 'Na'
        all_dict['FalPos'] = 'Na'
    all_dict['gt_shot'] += gt_shot
    all_dict['gt_all'] += len(gt_boxes)
    if all_dict['gt_all'] != 0:
        all_dict['Recall'] = round(all_dict['gt_shot']/all_dict['gt_all'],3)
    else:
        all_dict['Recall'] = 'Na'
    sum_dict['All'] = all_dict

    the_dict = sum_dict[cls_name_in]
    the_dict['pred_r'] += pred_r
    the_dict['pred_all'] += len(pred_boxes)
    if the_dict['pred_all'] != 0:
        the_dict['Precis'] = round(the_dict['pred_r']/the_dict['pred_all'],3)
        the_dict['FalPos'] = round((the_dict['pred_all']-the_dict['pred_r'])/the_dict['pred_all'],3)
    else:
        the_dict['Precis'] = 'Na'
        the_dict['FalPos'] = 'Na'
    the_dict['gt_shot'] += gt_shot
    the_dict['gt_all'] += len(gt_boxes)
    if the_dict['gt_all'] != 0:
        the_dict['Recall'] = round(the_dict['gt_shot']/the_dict['gt_all'],3)
    else:
        the_dict['Recall'] = 'Na'
    sum_dict[cls_name_in] = the_dict
    return sum_dict

def iou_pure(pred_boxes,gt_boxes,iou_the=0.3):
    pred_eval_list = [0]*len(pred_boxes)
    gt_eval_list = [0]*len(gt_boxes)

    for i, gt_box in enumerate(gt_boxes):
        for j, pred_box in enumerate(pred_boxes):
            gt_box_xyxy = [gt_box[0],gt_box[1],gt_box[0]+gt_box[2],gt_box[1]+gt_box[3]]
            m_iou = cal_intersect(gt_box_xyxy, pred_box)
            if m_iou >= iou_the:
                # print('-'*10+'m_iou'+'-'*10,m_iou)
                gt_eval_list[i]=1
                pred_eval_list[j]=1
    pred_r,pred_w = pred_eval_list.count(1),pred_eval_list.count(0)
    gt_shot,gt_miss = gt_eval_list.count(1),gt_eval_list.count(0)
    return round(gt_shot/len(gt_boxes))


def init_pdframe(cls_names_t):
    tmp_dict = {}
    tmp_dict['Iters'] = []
    tmp_dict['All_FalPos'] = []
    tmp_dict['All_Recall'] = []
    for cls_name in cls_names_t:
        if cls_name == 'bakcground':
            continue
        tmp_dict[cls_name+'_FalPos'] = []
        tmp_dict[cls_name+'_Recall'] = []
    return pd.DataFrame(tmp_dict)

def mergin2pdFrame(sum_dict,sum_pd,iters,gt_shot_r):
    tmp_dict = {}
    tmp_dict['Iters'] = [iters]
    tmp_dict['gt_shot_r'] = [gt_shot_r]
    for cls_name in list(sum_dict.keys()):
        dict_i = sum_dict[cls_name]
        tmp_dict[cls_name+'_FalPos'] = [dict_i['FalPos']]
        tmp_dict[cls_name+'_Recall'] = [dict_i['Recall']]

    tmp_pd = pd.DataFrame(tmp_dict)
    sum_pd = sum_pd.append(tmp_pd,ignore_index=False)[tmp_pd.columns.tolist()]
    return sum_pd

def write_txt(txt_path,sum_dict,iters):
    with open(txt_path,'a') as file:
        file.write('-'*10+iters+'-'*10+'\n')
        for cls_name in list(sum_dict.keys()):
            file.write('-'*5+cls_name+'-'*5+'\n')
            file.write(str(sum_dict[cls_name])+'\n')
        file.write('\n')
        file.write('\n')

def get_cls_name(json_file,return_en=True):
    '''
    ['B_QBHQSZ','Z_Z','CC_QZNB','SSX_SSJT']
    '''
    with open(json_file, "r", encoding="utf-8") as f:
        keyword_dict = json.load(f)
    catagerts_names = list(keyword_dict.keys())
    list_tmp = []
    for k,v in keyword_dict.items():
        list_i = [k,v[1],v[0]]
        list_tmp.append(list_i)
    list_tmp = sorted(list_tmp,key=lambda x: x[2], reverse=False)
    en_name=[x[1] for x in list_tmp]
    cn_name=[x[0] for x in list_tmp]
    en_name.insert(0, "background")
    cn_name.insert(0, "background")
    if return_en:
        return en_name
    else:
        return cn_name
    # return [list_tmp,cn_name,en_name]

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out

if __name__ == '__main__':
    from timeit import default_timer as timer
    import json
    from tqdm import tqdm
    import tensorflow as tf

    # os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    from pycocotools.coco import COCO

    thresh = 0.1
    iou_fiter = 0.0001

    try_times = 'Ban_T0'
    train_code = try_times+'_train'
    val_code = try_times+'_val'

    imgFolder = '/home/qiaoran/Project/mobilenet_fastercnn/dataset_raw/4_19_JWrelabel_203/images_raw_anno/orginal_img/'
    annoVal = '/home/qiaoran/Project/mobilenet_fastercnn/dataset_raw/4_19_JWrelabel_203/COCOjson/train_val/4_19_JW_relabel_Drlian_Ban_T0_val.json'
    keyword_json = '/home/qiaoran/Project/mobilenet_fastercnn/dataset_raw/4_19_JWrelabel_203/COCOjson/T_329_d_BanZ_T1.json'

    checkpoint_dir = './output/mobile/4_19_JW_relabel_DrL/{}/default/'.format(train_code,try_times)
    cls_names = get_cls_name(keyword_json,return_en=True) #must en class name

    out_dir =  './inerence_cal_test_check/4_19_JW_relabel_DrL/{}/'.format(try_times)
    txt_path = './output/mobile/4_19_JW_relabel_DrL/{}/{}_thresh{}.txt'.format(train_code,try_times,str(thresh))
    csv_path = './output/mobile/4_19_JW_relabel_DrL/{}/{}_thresh{}.csv'.format(train_code,try_times,str(thresh))
    draw_img = True

    if os.path.exists(txt_path):
        os.remove(txt_path)
    if os.path.exists(csv_path):
        os.remove(csv_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    num_class = len(cls_names)
    sum_pd = init_pdframe(cls_names)

    # tfmodels = ["mobile_faster_rcnn_iter_37500.ckpt",
    #            "mobile_faster_rcnn_iter_37500.ckpt"]

    tfmodels = tf.train.get_checkpoint_state(
        checkpoint_dir,
        latest_filename=None
        ).all_model_checkpoint_paths

    # tfmodels = chunkIt(tfmodels, 4)[0]
    for i in tqdm(range(0, len(tfmodels), 1)):
        # tfmodel_path = checkpoint_dir+tfmodels[i]
        tfmodel_path = tfmodels[i]
        tfmodel =os.path.basename(tfmodel_path)

        start = timer()
        # print('{} {} start initialized {}'.format('-'*20,tfmodel,'-'*20))
        tf.reset_default_graph()
        sess, net = init_sess(tfmodel_path,num_class)
        elapsed_time = round(timer() - start,2)
        # print('{} model loaded {}s {}'.format('-'*20,elapsed_time,'-'*20))

        coco = COCO(annoVal)
        images_dict = coco.loadImgs(coco.getImgIds())

        sum_dict = init_cls_stats_dict(cls_names)

        for i, im_name in enumerate(images_dict):
            image_id = im_name['id']
            in_path = os.path.join(imgFolder, im_name['file_name'])
            out_path = os.path.join(out_dir, im_name['file_name'])
            anno_ids = coco.getAnnIds(imgIds=image_id)
            annos = coco.loadAnns(ids=anno_ids)

            im = cv2.imread(in_path)
            pred_dict,pred_boxesr = pred(sess, net, in_path, CLASSES=cls_names,thresh = thresh,iou_s=iou_fiter)
            gt_shot_all =  [item ['bbox'] for item in annos]
            gt_shot_r = iou_pure(pred_boxesr,gt_shot_all)

            for cls_name in cls_names:
                if cls_name == 'bakcground':
                    continue
                pred_bboxes = pred_dict[cls_name]
                gt_bboxes = [item ['bbox'] for item in annos if coco.loadCats(ids=item['category_id'])[0]['name'] == cls_name]
                sum_dict = core_val(pred_bboxes,gt_bboxes,sum_dict,cls_name_in=cls_name,iou_the=iou_fiter)

                if draw_img:
                    im = vis(im,pred_bboxes,cls_name,pred=True)
                    im = vis(im,gt_bboxes,cls_name,pred=False)
            if draw_img:
                cv2.imwrite(out_path, im)

            # if i>10:
            #     break

        sum_pd = mergin2pdFrame(sum_dict,sum_pd,tfmodel,gt_shot_r)

        print(sum_pd[-1:])
        write_txt(txt_path,sum_dict,tfmodel)
        if os.path.exists(csv_path):
            os.remove(csv_path)
        sum_pd.to_csv(csv_path)
        tf.reset_default_graph()
    print(sum_pd[-3:])
    print(txt_path,'done')
    print(csv_path,'done')
