import sys,os
from collections import defaultdict
import argparse
import cv2
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# from utils.timer import Timer
# import core.test_engine as infer_engine
# import utils.c2 as c2_utils

# from utils.collections import AttrDict
import pycocotools.mask as mask_util
import shutil
import numpy as np

# import sys
# sys.path.append('/data/ubuntu/mask_rcnn_caffe2/Detectron-xRay/Wrapper/')
from dr_tools import visQR as vis_utils
# from dr_tools import dr1_0_JiZhu_process
from dr_tools import dr1_0_XinYin_process
from dr_tools import dr1_0_XinYin_process
from dr_tools import dr1_0_HengGeLine_process
from dr_tools import disease_desc
from timeit import default_timer as timer
import mmcv
from mmdet.apis import init_detector, inference_detector

def trans2caffe2(result,n_cls):
    # n_cls = len(cls_names)
    # if 'background' not in cls_names:
    #     n_cls+=1
    n_cls

    bbox_result, segm_result = result
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)

    segms = mmcv.concat_list(segm_result)

    c_boxes,c_segms,c_keypoints = [],[],None
    for i in range(n_cls):
        c_boxes.append([])
        c_segms.append([])

    for j in range(len(labels)):
        idx = labels[j]
        idx+=1

        # box = np.array(bboxes[i]).astype('float32')
        box = bboxes[j]
        seg = segms[j]

        c_boxes[idx].append(box)
        c_segms[idx].append(seg)

    c_boxes_np = []
    for k in c_boxes:
        if len(k)>0:
            c_boxes_np.append(np.array(k).astype('float32'))
        else:
            c_boxes_np.append(k)

    return c_boxes_np,c_segms,c_keypoints

def convert_from_cls_format(cls_boxes, cls_segms, cls_keyps):
    """Convert from the class boxes/segms/keyps format generated by the testing
    code.
    """
    box_list = [b for b in cls_boxes if len(b) > 0]
    if len(box_list) > 0:
        boxes = np.concatenate(box_list)
    else:
        boxes = None
    if cls_segms is not None:
        segms = [s for slist in cls_segms for s in slist]
    else:
        segms = None
    if cls_keyps is not None:
        keyps = [k for klist in cls_keyps for k in klist]
    else:
        keyps = None
    classes = []
    # print('cls_boxes',cls_boxes)
    for j in range(len(cls_boxes)):
        classes += [j] * len(cls_boxes[j])
    return boxes, segms, keyps, classes

def get_class_string(class_index, score, dataset):
    class_text = dataset_classes[class_index] if dataset is not None else \
        'id{:d}'.format(class_index)
    return class_text + ' {:0.2f}'.format(score).lstrip('0')

def get_qiguan_dataset():
     # from utils.collections import AttrDict
     # ds = AttrDict()
     # # classes = ['__background__','jizhu','xinying','fei'] #50
     # # classes = ['__background__','fei','xinying','jizhu'] #101
     # # classes = ['__background__','fei','xinying','spine']
     # classes = ['__background__','Fei','XinYin','JiZhu','ZuiTi'] #50
     # ds.classes = {i: name for i, name in enumerate(classes)}

     # ds = {'classes': {0: u'__background__', 1: u'Fei', 2: u'XinYin', 3: u'JiZhu', 4: u'ZuiTi'}}
     ds = {0: u'__background__', 1: u'fei', 2: u'xinying', 3: u'jizhu', 4: u'zuiti'}
     return ds

def left_or_right_lung(mask_path):
    # mask = cv2.imread(mask_path)
    mask = mask_path
    mask_half_left = mask[:,:int(mask.shape[1]/2),:]
    mask_half_right = mask[:,int(mask.shape[1]/2):,:]
    num_white_left = np.count_nonzero(mask_half_left != 0)
    num_white_right = np.count_nonzero(mask_half_right != 0)
    if num_white_left > num_white_right:
        return 'right'
    else:
        return 'left'

def find_contours(mask,Raw_Return=True):
    '''
    input:
        mask = cv2.imread(mask)
        cords = find_contours(mask)
    output:
        find all contours and filter biggest contours1
        return [(500, 500), (500, 600), (600, 500), (600, 600)....]
    '''
    lower_blue = np.array([128,128,128])
    upper_blue = np.array([255,255,255])
    mask_blue1 = cv2.inRange(mask, lower_blue, upper_blue)
    # _, contours1, _ = cv2.findContours(mask_blue1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours1,hiers = cv2.findContours(mask_blue1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if Raw_Return:
        return contours1
    else:
        max_len,max_idx=0,0
        for i in range(len(contours1)):
            len_i = len(contours1[i])
            if len_i > max_len:
                max_len = len_i
                max_idx = i
        coords = []
        for i in contours1[max_idx]:
            cod = i.tolist()[0]
            coords.append(tuple(cod))
        return coords

def divded_dai(img_input,oeginal_img_path=None,total_pts=34,n_pts_ingnore=5,n_up_ingnore=1,draw=False,out_path=None):
    '''
    divided dai in three parts
    in_put:
        img = cv2.imread(path)
    output:
        three region, from left to right.
        array([[   0,    0],
               [1161,    0],
               ...
               [ 670, 3001],
               [   0, 3001]])
    '''
    if isinstance(img_input, str):
        img = cv2.imread(img_input)
    else:
        img = img_input
    img_size = img.shape
    ptsl,ptsr=[],[]
    upper_shorter,down_shorter = int(img_size[0]/15),int(img_size[0]/15)
    for height_i in range(upper_shorter,img_size[0]-down_shorter,int(img_size[0]/total_pts)):
        ptl,ptr = [],[]
        line = img[height_i,:,0]
        a = np.where(line==[255])
        if a[0].size > 0:
            width_t = a[0][int(a[0].shape[0]/3)]
            ptl.append(width_t)
            ptl.append(height_i)
            ptsl.append(ptl)

            width_tr = a[0][int(a[0].shape[0]*2/3)]
            ptr.append(width_tr)
            ptr.append(height_i)
            ptsr.append(ptr)
    ptsl_np = np.asarray(ptsl, dtype=np.int32)
    ptsr_np = np.asarray(ptsr, dtype=np.int32)

    upper_shorter,down_shorter = int(len(ptsl_np)/30),int(len(ptsl_np)/30)
    upper_shorter_r,down_shorter_r = int(len(ptsl_np)/30),int(len(ptsl_np)/30)

    ptsl_np = ptsl_np[upper_shorter+n_up_ingnore:len(ptsl_np)-down_shorter-n_pts_ingnore]
    ptsr_np = ptsr_np[upper_shorter_r+n_up_ingnore:len(ptsr_np)-down_shorter_r-n_pts_ingnore]
    up0 = [0,0]
    up1 = [ptsl_np[0][0],0]
    up2 = [ptsr_np[0][0],0]
    up3 = [img_size[1],0]
    down0 = [0,img_size[0]]
    down1 = [ptsl_np[-1][0],img_size[0]]
    down2 = [ptsr_np[-1][0],img_size[0]]
    down3 = [img_size[1],img_size[0]]

    region1 = np.concatenate(([up0, up1],ptsl_np), axis=0)
    region1 = np.concatenate((region1,[down1, down0]), axis=0)

    region2 = np.concatenate(([up1, up2],ptsr_np), axis=0)
    region2 = np.concatenate((region2,[down2, down1]), axis=0)
    region2 = np.concatenate((region2,np.flip(ptsl_np, 0)), axis=0)

    region3 = np.concatenate(([up3, up2],ptsr_np), axis=0)
    region3 = np.concatenate((region3,[down2, down3]), axis=0)

    if draw:
        orginal_img = cv2.imread(oeginal_img_path)
        orginal_img=cv2.drawContours(orginal_img, [region1], -1, (0,180,0), 10)
        orginal_img=cv2.drawContours(orginal_img, [region2], -1, (0,0,180), 10)
        orginal_img=cv2.drawContours(orginal_img, [region3], -1, (150,0,0), 10)

        # pts = ptsl_np
        # ptsr = ptsr_np
        #
        # cv2.polylines(orginal_img,[pts],False,(0,255,0),1)
        # cv2.polylines(orginal_img,[ptsr],False,(0,255,0),1)
        #
        # for pt in pts:
        #     cv2.circle(orginal_img,(int(pt[0]), int(pt[1])), 5, (0,0,255),-1)
        # for ptr in ptsr:
        #     cv2.circle(orginal_img,(int(ptr[0]), int(ptr[1])), 5, (0,0,255),-1)
        cv2.imwrite(out_path,orginal_img)
    return region1.tolist(),region2.tolist(),region3.tolist()

def find_contours_str(mask,Raw_Return=False):
    '''
    input:
        mask = cv2.imread(mask)
        cords = find_contours(mask)
    output:
        find all contours and filter biggest contours1
        return [(500, 500), (500, 600), (600, 500),@ (600, 600)....]
    '''
    lower_blue = np.array([128,128,128])
    upper_blue = np.array([255,255,255])
    mask_blue1 = cv2.inRange(mask, lower_blue, upper_blue)
    # _, contours1, _ = cv2.findContours(mask_blue1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours1,hiers = cv2.findContours(mask_blue1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if Raw_Return:
        return contours1
    else:
        max_len,max_idx=0,0
        for i in range(len(contours1)):
            len_i = len(contours1[i])
            if len_i > max_len:
                max_len = len_i
                max_idx = i
        coords = []
        for i in contours1[max_idx]:
            cod = i.tolist()[0]
            pts = str(int(cod[0]))+','+str(int(cod[1]))
            # coords.append(str(tuple(cod)))
            coords.append(pts)
        return coords

def left_or_right_lung(mask_path):
    if isinstance(mask_path, str):
        mask = cv2.imread(mask_path)
    else:
        mask = mask_path
    mask_half_left = mask[:,:int(mask.shape[1]/2),:]
    mask_half_right = mask[:,int(mask.shape[1]/2):,:]
    num_white_left = np.count_nonzero(mask_half_left != 0)
    num_white_right = np.count_nonzero(mask_half_right != 0)
    if num_white_left > num_white_right:
        return 'right'
    else:
        return 'left'

def boxes_dulp_filter(qiguan_list):
    # std_names = ['Fei_right','Fei_left','XinYin','JiZhu','ZuiTi']
    # std_names = ['fei_right','fei_left','xinying','jizhu']
    std_names = ['fei_right','fei_left','xinying','jizhu','zuiti']
    orginal_names,orgianal_score,keep_idx=[],[],[]
    for qiguan in qiguan_list:
        name = qiguan['Name']
        score = qiguan['Score']
        orginal_names.append(name)
        orgianal_score.append(score)
    for std_name in std_names:
        idx_list = [i for i, e in enumerate(orginal_names) if e == std_name]
        if std_name == 'zuiti':
            keep_idx+=idx_list
            continue

        if len(idx_list) == 1:
            keep_idx.append(idx_list[0])
        elif len(idx_list) > 1:
            orginal_idx,pk_scores = [],[]
            for idx in idx_list:
                pk_scores.append(orgianal_score[idx])
            winner_index = idx_list[pk_scores.index(max(pk_scores))]
            keep_idx.append(winner_index)
        else:
            continue
    # qiguan_list_clear = []
    # for i in keep_idx:
    #     item = qiguan_list[i]
    #     qiguan_list_clear.append(item)
    # assert len(keep_idx) <= 4
    return keep_idx

def get_XinYin_Fer_area(QiGuan_mask_return):
    fei0 = QiGuan_mask_return['fei0']
    fei1 = QiGuan_mask_return['fei1']
    xin1 = QiGuan_mask_return['xinying']

    if isinstance(fei0, str):
        mask1 = cv2.imread(fei0)
        mask2 = cv2.imread(fei1)
        mask3 = cv2.imread(xin1)

    mask1 = fei0
    mask2 = fei1
    mask3 = xin1

    contours1 = find_contours(mask1)
    contours2 = find_contours(mask2)
    contours3 = find_contours(mask3)

    XinYin_xmin_x,XinYin_xmax_x = 10000,0
    for i in range(len(contours1)):
        point_list1 = contours1[i]
        for ii in range(len(point_list1)):
            y=point_list1[ii][0][1]
            x=point_list1[ii][0][0]
            if XinYin_xmin_x >= x:
                XinYin_xmin_x=x
                XinYin_xmin_y=y
            if XinYin_xmax_x <= x:
                XinYin_xmax_x=x
                XinYin_xmax_y=y

    Fei_ymin_y_1 = 10000
    for i in range(len(contours2)):
        point_list2 = contours2[i]
        for ii in range(len(point_list2)):
            y=point_list2[ii][0][1]
            x=point_list2[ii][0][0]
            if Fei_ymin_y_1 >= y:
                Fei_ymin_x_1=x
                Fei_ymin_y_1=y

    Fei_ymin_y_2 = 10000
    for i in range(len(contours3)):
        point_list3 = contours3[i]
        for ii in range(len(point_list3)):
            y=point_list3[ii][0][1]
            x=point_list3[ii][0][0]
    #         print(point_list3[ii])
            if Fei_ymin_y_2 >= y:
                Fei_ymin_x_2=x
                Fei_ymin_y_2=y

    XinYin_xmin = [XinYin_xmin_x,XinYin_xmin_y]
    XinYin_xmax = [XinYin_xmax_x,XinYin_xmax_y]
    fei_ymin_1 = [Fei_ymin_x_1,Fei_ymin_y_1]
    fei_ymin_2 = [Fei_ymin_x_2,Fei_ymin_y_2]

    mask4 = np.zeros(mask1.shape).astype(mask1.dtype)
    a3 = np.array( [[XinYin_xmin,XinYin_xmax,fei_ymin_2,fei_ymin_1]], dtype=np.int32 )
    mask4 = cv2.fillPoly(mask4, a3, (255,255,255))
    mask_inv = mask1+mask2+mask3+mask4
    mask_inv[mask_inv>0]=255
    coords = find_contours(mask_inv,Raw_Return=False)
    return coords

def find_crop_pts_RuTouYing(e, bbox, direction, y=0.5):
    assert direction in ['left', 'right']
    bbox = bbox.tolist()
    bbox = [int(item) for item in bbox]
    y1 = int((bbox[1]+bbox[3])*y)

    tmp = e[y1, :, 0].tolist()

    try:
        x11, x12 = [i for i, item in enumerate(tmp) if item != 0][0], [i for i, item in enumerate(tmp) if item != 0][-1]
        return [bbox, [x11, y1], [x12, y1]]
    except IndexError:
        return [bbox, 'None', 'None']

def find_QiGuan(img_path,model,XinYin_path=None,HengGe_path=None,draw_sample=False,qiguan_thresh=0.5,XinYing_thresh=0.48):
    if not os.path.exists(img_path):
        return 'No_img','No_img'
    im = cv2.imread(img_path)
    img_size = im.shape

    # timers = defaultdict(Timer)
    # with c2_utils.NamedCudaScope(0):
    #     boxes, segms, keypoints = infer_engine.im_detect_all(
    #         model, im, None, timers=timers
    #     )
    result = inference_detector(model, img_path)
    boxes, segms, keypoints = trans2caffe2(result,5)

    if draw_sample:
        output_dir = './test_data/dr1_0_QiGuan_formal_out/'
        mask_out_dir = './test_data/dr1_0_QiGuan_formal_out/'
        dummy_skin_dataset = get_qiguan_dataset()
        mask_dict123 = vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            img_path,
            output_dir,
            boxes,
            segms,
            keypoints,
            dataset=dummy_skin_dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=0.6,
            kp_thresh=2,
            generate_mask=True,
            mask_dir = mask_out_dir,
            show_box=False,
            fill_mask=False,
            # ignore_cls = ['fei','xinying']
            )

    # print('boxes',boxes)
    # print('boxes', boxes)
    # print('')
    # print('segms', segms)
    # print('')
    # print('keypoints', keypoints)
    # print('')

    # assert 1>2
    if isinstance(boxes, list):
        boxes, segms, keypoints, classes = convert_from_cls_format(
            boxes, segms, keypoints)
    # print('classes',classes)
    thresh = qiguan_thresh
    if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh:
        return 'No_result','No_result'

    if segms is not None and len(segms) > 0:
        masks = mask_util.decode(segms)

    dataset_classes = get_qiguan_dataset()
    left_corner,right_corner = [],[]
    mask_dict = {}
    QiGuan_return = {}
    QiGuan_pts_return = []
    mask_dict_old = []
    fei_idx = 0
    QiGuan_mask_return = {}
    qiguan_list = []

    for i in range(len(boxes)):
        one_itemi={}
        bbox = boxes[i, :4]
        score = boxes[i, 4]

        cls = classes[i]
        e = masks[:, :, i]
        e[e!=0]=255
        mask_img = cv2.merge((e,e,e))
        class_text = dataset_classes[cls]
        if class_text == 'fei':
            direction = left_or_right_lung(mask_img)
            one_itemi['Name'] = 'fei_'+direction
        else:
            one_itemi['Name'] = class_text
        one_itemi['Score'] = score
        qiguan_list.append(one_itemi)

    # print('qiguan_list',qiguan_list)
    keep_idx = boxes_dulp_filter(qiguan_list)
    # print('keep_idx',keep_idx)

    Zuiti_pts=[]

    for i in keep_idx:
        one_item = {}
        bbox = boxes[i, :4]
        score = boxes[i, 4]


        if score > qiguan_thresh:
            # score = boxes[i, -1]
            cls = classes[i]
            class_text = dataset_classes[cls]
            e = masks[:, :, i]
            e[e!=0]=255
            mask_img = cv2.merge((e,e,e))
            Segts = find_contours_str(mask_img)
            contours_raw = find_contours(mask_img)
            coords_big = find_contours(mask_img,Raw_Return=False)

            if class_text == 'zuiti':
                Zuiti_pts.append(coords_big)
                continue

            if class_text == 'xinying':
                QiGuan_return['xinying_box'] = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            if class_text == 'fei':
                QiGuan_return['fei_box'] = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]

            if class_text == 'fei':
                xmin,ymin,xmax,ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                left_corner.append([xmin,ymin])
                right_corner.append([xmax,ymax])
                mask_dict[class_text+str(fei_idx)] = mask_img
                padding = 50
                # xmin,ymin,xmax,ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                im_crop_new = im[ymin-padding:ymax+padding,xmin-padding:xmax+padding]
                direction = left_or_right_lung(mask_img)
                QiGuan_return[class_text+'_crop'+str(fei_idx)] = [direction,ymin,ymax,xmin,xmax]
                mask_dict[class_text+'_crop'+str(fei_idx)] = im_crop_new


                region1,region2,region3 = divded_dai(mask_img,oeginal_img_path = img_path,total_pts=34,n_pts_ingnore=5,n_up_ingnore=1,out_path='fei_dai'+str(fei_idx)+'.jpg')
                QiGuan_return[class_text+'_loca_info'+str(fei_idx)] = [direction,coords_big,region1,region2,region3]
                QiGuan_mask_return[class_text+str(fei_idx)] = mask_img
                QiGuan_mask_return[class_text+str(fei_idx)+'_direction'] = direction
                QiGuan_mask_return[class_text+str(fei_idx)+'_box'] = bbox
                fei_idx+=1

                direction = left_or_right_lung(mask_img)
                one_item['Name'] = 'fei_'+direction
                # cv2.imwrite('fei_'+direction+'.jpg',mask_img)
            elif class_text == 'jizhu':
                upper_shorter,down_shorter = int(5),int(5)
                num_segs = 100
                jizhu_midLine = []
                for height_i in range(upper_shorter,img_size[0]-down_shorter,int(img_size[0]/num_segs)):
                    pt = []
                    line = mask_img[height_i,:,0]
                    a = np.where(line==[255])
                    if a[0].size > 0:
                        if height_i <= img_size[0]*0.965:
                            width_t = np.median(a)
                            pt.append(int(width_t))
                            pt.append(height_i)
                            jizhu_midLine.append(pt)
                # pts_np = np.asarray(pts, dtype=np.int32)
                # upper_shorter,down_shorter = int(len(pts_np)/30),int(len(pts_np)/30)
                # pts_np = pts_np[upper_shorter:len(pts_np)-down_shorter]

                mask_dict[class_text] = mask_img
                one_item['Name'] = class_text
                QiGuan_mask_return[class_text] = mask_img
                QiGuan_return['jizhu_midline'] = jizhu_midLine
            else:
                # QiGuan_return[class_text+'_loca_info'] =coords_big
                mask_dict[class_text] = mask_img
                one_item['Name'] = class_text
                QiGuan_mask_return[class_text] = mask_img
                # cv2.imwrite(class_text+'.jpg',mask_img)
            one_item['Segts'] = Segts
            QiGuan_pts_return.append(one_item)
        else:
            continue

    if 'xinying' in QiGuan_mask_return and 'fei0' in QiGuan_mask_return and 'fei1' in QiGuan_mask_return:
        XY_Fei_coords_big = get_XinYin_Fer_area(QiGuan_mask_return)
        QiGuan_return['XinYin_fei'] = XY_Fei_coords_big
        mask_xinying = QiGuan_mask_return['xinying']
        xinying_contour = find_contours(mask_xinying,Raw_Return=False)
        QiGuan_return['XinYin_contour'] = xinying_contour

        mask_fei0 = QiGuan_mask_return['fei0']
        fei0_contour = find_contours(mask_fei0,Raw_Return=False)
        QiGuan_return['fei0_contour'] = fei0_contour

        mask_fei1 = QiGuan_mask_return['fei1']
        fei1_contour = find_contours(mask_fei1,Raw_Return=False)
        QiGuan_return['fei1_contour'] = fei1_contour

    if 'fei0' in QiGuan_mask_return and 'fei0_direction' in QiGuan_mask_return and 'fei0_box' in QiGuan_mask_return and 'fei1' in QiGuan_mask_return  and 'fei1_direction' in QiGuan_mask_return and 'fei1_box' in QiGuan_mask_return:
        fei0_mask = QiGuan_mask_return['fei0']
        fei0_direction = QiGuan_mask_return['fei0_direction']
        fei0_box = QiGuan_mask_return['fei0_box']
        RuTouYing_fei0 = find_crop_pts_RuTouYing(fei0_mask, fei0_box, fei0_direction, y=0.5)
        QiGuan_return['RuTouYing_return_'+fei0_direction] = RuTouYing_fei0

        fei1_mask = QiGuan_mask_return['fei1']
        fei1_direction = QiGuan_mask_return['fei1_direction']
        fei1_box = QiGuan_mask_return['fei1_box']
        RuTouYing_fei1 = find_crop_pts_RuTouYing(fei1_mask, fei1_box, fei1_direction, y=0.5)
        QiGuan_return['RuTouYing_return_'+fei1_direction] = RuTouYing_fei1


    if len(left_corner) == 2:
        if left_corner[0][0] > left_corner[1][0]:
            left_pt = left_corner[1]
        else:
            left_pt = left_corner[0]

        if right_corner[0][0] > right_corner[1][0]:
            right_pt = right_corner[0]
        else:
            right_pt = right_corner[1]
    else:
        left_pt = left_corner[0]
        right_pt = right_corner[0]
    padding=150
    im_feis_crop = im[left_pt[1]-padding:right_pt[1]+padding,left_pt[0]-padding:right_pt[0]+padding]
    QiGuan_return['fei_regions'] = [left_pt,right_pt]

    if mask_dict is None:
        mask_dict = {}
        mask_dict['No_result']='No_result'
    else:

        if 'xinying' in mask_dict and 'fei0' in mask_dict and 'fei1' in mask_dict and 'jizhu' in mask_dict:
            XinYin,XinYin_Rate,XYimg_out = dr1_0_XinYin_process.XinYin_Detect(mask_dict,img_path,img_out=XinYin_path,thether=XinYing_thresh)
            QiGuan_return['XinYin'] = [XinYin,XinYin_Rate,XYimg_out]
            if XinYin_Rate > 0.5:
                xinyin_desc = disease_desc.xinyingzengda(XinYin_Rate)
                QiGuan_return['XinYin_Desc'] = xinyin_desc
        else:
            QiGuan_return['XinYin'] = 'No_XinYin_result'

        if 'xinying' in mask_dict and 'fei0' in mask_dict and 'fei1' in mask_dict:
            assert os.path.exists(img_path)
            HengGe_return = dr1_0_HengGeLine_process.find_HengGe_diff(mask_dict,img_path,img_out=HengGe_path)
            one_item = {}
            one_item['Segts'] = [str(HengGe_return[2][0])+','+str(HengGe_return[2][1]),str(HengGe_return[3][0])+','+str(HengGe_return[3][1])]
            one_item['Name'] = 'Hengge_pt'
            QiGuan_pts_return.append(one_item)

            if HengGe_return[0] > 200:
                HengGe_diff = HengGe_return[0]
                QiGuan_return['HengGe'] = [HengGe_diff,HengGe_path]
                hengGe_desc = disease_desc.HengGe(HengGe_return[1])
                QiGuan_return['HengGe_desc'] = hengGe_desc
            else:
                QiGuan_return['HengGe'] = 'No_HengGe_result'
        else:
            QiGuan_return['HengGe'] = 'No_HengGe_result'
            # print('No_HengGe_result')

    return QiGuan_return,QiGuan_pts_return,Zuiti_pts

if __name__ == "__main__":

    # ???????????????
    # img_path = 'test.jpg'
    img_path = '1.3.12.2.1107.5.3.33.4700.11.201811081519490046-1.jpg'
    config_file = './DR_models_configs/dr1_0_QiGuan_formal_cfg.py'
    checkpoint_file = './DR_models_configs/dr1_0_QiGuan_formal.pth'
    start = timer()

    # ?????????
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    elapsed_time = round(timer() - start,2)
    print('model loaded in {}s'.format(elapsed_time))

    QiGuan_return, QiGuan_pts_return, Zuiti_pts = find_QiGuan(img_path,model,XinYin_path='./test_data/dr1_0_QiGuan_formal_out/XinYin_draw.jpg',HengGe_path='./test_data/dr1_0_QiGuan_formal_out/HengGe_draw.jpg',draw_sample=True)

    if 'No_img' in QiGuan_return:
        print('No_img')
    elif 'No_result' in QiGuan_return:
        print('No_result')
    if 'fei_regions' in QiGuan_return and 'fei_loca_info0' in QiGuan_return and 'fei_loca_info1' in QiGuan_return:
        print('Go_feiJieJie')
    if 'fei_crop0' in QiGuan_return and 'fei_crop1' in QiGuan_return:
        print('Go_JiYe')
    print('pred done')
        # print('QiGuan_return',QiGuan_return)
    if 'No_JiZhu_result' in QiGuan_return:
        print('No_JiZhu_result')
    if 'No_XinYin_result' in QiGuan_return:
        print('No_JiZhu_result')
    print('done')
    print('QiGuan_return=',QiGuan_return)

    print('Zuiti_pts',Zuiti_pts)
