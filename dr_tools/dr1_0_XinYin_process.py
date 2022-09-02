from __future__ import division

import os
import numpy as np
import cv2
# import imutils

def getmask_xywh(img_path):
    # mask = cv2.imread(img_path)
    mask = img_path
    try:
        mask.shape
    except:
        print('Warning ',img_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # mask_new, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    contours,hiers = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    [x, y, w, h] = cv2.boundingRect(mask)
    return [x, y, w, h],contours

def find_extrame_pt_counters(xin_path):
    # cnts = cnts[0]
    # if imutils.is_cv2() else cnts[1]
    # c = max(cnts, key=cv2.contourArea)
    # img = cv2.imread(xin_path)
    img = xin_path
    lower_blue = np.array([128,128,128])
    upper_blue = np.array([255,255,255])
    mask_blue1 = cv2.inRange(img, lower_blue, upper_blue)
    # _, cnts, _ = cv2.findContours(mask_blue1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts,hiers = cv2.findContours(mask_blue1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    c = cnts[0]
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    return extLeft,extRight

def draw_lines(cor,orgin_img,output_img,rate,contoursAll,xin1_path=None):
    im = cv2.imread(orgin_img)
    # im = orgin_img
    fei0C,fei1C,xin1C = cor[0],cor[1],cor[2]
    contoursF0,contoursF1,contoursX0 = contoursAll[0],contoursAll[1],contoursAll[2]
    jizhu_img = contoursAll[3]

    if fei0C[0] > fei1C[0]:
        fei_xmin = fei1C[0]
    else:
        fei_xmin = fei0C[0]

    if fei0C[0]+fei0C[2] > fei1C[0]+fei1C[2]:
        fei_xmax = fei0C[0]+fei0C[2]
    else:
        fei_xmax = fei1C[0]+fei1C[2]

    if fei0C[1] > fei1C[1]:
        fei_ymin = fei1C[1]
    else:
        fei_ymin = fei0C[1]

    if fei0C[1]+fei0C[3] > fei1C[1]+fei1C[3]:
        fei_ymax = fei0C[1]+fei0C[3]
    else:
        fei_ymax = fei1C[1]+fei1C[3]

    fei_comb_x1 = (fei_xmin,fei_ymin)
    fei_comb_x2 = (fei_xmax,fei_ymin)
    fei_comb_x3 = (fei_xmax,fei_ymax-300)
    fei_comb_x4 = (fei_xmin,fei_ymax-300)

    xin_comb_x1 = (xin1C[0],xin1C[1])
    xin_comb_x2 = (xin1C[0]+xin1C[2],xin1C[1])
    xin_comb_x3 = (xin1C[0]+xin1C[2],xin1C[1]+xin1C[3])
    xin_comb_x4 = (xin1C[0],xin1C[1]+xin1C[3])

    img_size=im.shape
    img_height,img_width = img_size[0],img_size[1]
    #draw fei
    line_widths = 10
    # colcor_line = (255, 0, 0)
    colcor_line = (0, 0, 255) #red
    circle_size = 10
    cv2.line(im, (fei_comb_x3), (fei_comb_x4), colcor_line, line_widths)

    cv2.line(im, (fei_xmin,0), (fei_xmin,img_height), (93,92,91), line_widths)
    cv2.line(im, (fei_xmax,0), (fei_xmax,img_height), (93,92,91), line_widths)

    cv2.circle(im, (fei_comb_x3), circle_size, (0, 255, 0), -1)
    cv2.circle(im, (fei_comb_x4), circle_size, (0, 255, 0), -1)
    #draw XinYin
    extLeft,extRight = find_extrame_pt_counters(xin1_path)
    extLeftY = extLeft[1]

    line = jizhu_img[extLeftY,:,0]
    a = np.where(line==[255])
    if a[0].size > 0:
        jizhu_x = np.median(a)
    else:
        for i in range(-200,200):
            line = jizhu_img[extLeftY+i,:,0]
            a = np.where(line==[255])
            if a[0].size > 0:
                jizhu_x = np.median(a)
                break
    #jizhu line
    jizhu_x = int(jizhu_x)
    # if extLeft[1] <= extRight[1]:
    #     cv2.line(im, (jizhu_x,extLeft[1]-500), (jizhu_x,extRight[1]+100), colcor_line, line_widths)
    # else:
    #     cv2.line(im, (jizhu_x,extLeft[1]+100), (jizhu_x,extRight[1]-500), colcor_line, line_widths)

    cv2.line(im, (jizhu_x,0), (jizhu_x,fei_ymax+200), colcor_line, line_widths)

    cv2.line(im, (extLeft), (jizhu_x,extLeft[1]), colcor_line, line_widths)
    cv2.line(im, (extRight), (jizhu_x,extRight[1]), colcor_line, line_widths)
    cv2.circle(im, (extLeft), circle_size, (0, 255, 0), -1)
    cv2.circle(im, (extRight), circle_size, (0, 255, 0), -1)
    cv2.circle(im, (jizhu_x,extLeft[1]), circle_size, (0, 255, 0), -1)
    cv2.circle(im, (jizhu_x,extRight[1]), circle_size, (0, 255, 0), -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(im, str(round(rate,3)), (jizhu_x+40,fei_comb_x4[1]+100), font, 3, colcor_line, 6)
    cv2.imwrite(output_img,im,[int(cv2.IMWRITE_JPEG_QUALITY), 30])

def XinYin_Detect(mask_dict,orgin_img,img_out=None,thether=0.5):
    fei0 = mask_dict['fei0']
    fei1 = mask_dict['fei1']
    xin1 = mask_dict['xinying']
    jizhu1 = mask_dict['jizhu']

    fei0C,contoursF0 = getmask_xywh(fei0)
    fei1C,contoursF1 = getmask_xywh(fei1)
    xin1C,contoursX0 = getmask_xywh(xin1)

    if fei0C[0] > fei1C[0]:
        fei_xmin = fei1C[0]
    else:
        fei_xmin = fei0C[0]

    if fei0C[0]+fei0C[2] > fei1C[0]+fei1C[2]:
        fei_xmax = fei0C[0]+fei0C[2]
    else:
        fei_xmax = fei1C[0]+fei1C[2]

    fei_width = fei_xmax-fei_xmin
    xin_width = xin1C[2]
    XinYin_Rate = xin_width/fei_width
    XinYin_Rate = round(XinYin_Rate, 2)
    if XinYin_Rate > thether:
        XinYin = 'XinYin_Yes'
    else:
        XinYin = 'XinYin_No'

    cor=[fei0C,fei1C,xin1C]
    contoursAll = [contoursF0,contoursF1,contoursX0,jizhu1]
    # print('cor',cor)

    filename, file_extension = os.path.splitext(orgin_img)
    if not img_out:
        img_out = '/home/steven/mask_rcnn_caffe2/Detectron-xRay/Wrapper/Final_Out/_XinYinZengDa.jpg'
    draw_lines(cor,orgin_img,img_out,XinYin_Rate,contoursAll,xin1_path=xin1)
    return XinYin,XinYin_Rate,img_out

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    import sys
    sys.path.append('/home/steven/mask_rcnn_caffe2/Detectron-xRay/')
    import time

    from caffe2.python import workspace
    from core.config import assert_and_infer_cfg
    from core.config import cfg
    from core.config import merge_cfg_from_file
    from utils.io import cache_url
    from utils.timer import Timer
    import core.test_engine as infer_engine
    import utils.c2 as c2_utils
    import utils.logging

    c2_utils.import_detectron_ops()
    cv2.ocl.setUseOpenCL(False)
    print('model loading...')

    workspace.GlobalInit(['caffe2', '--caffe2_log_level=1'])
    model_cfg = './DR_models_configs/dr1_0_QiGuan_Res101_3cls.yaml'
    model_weights = './DR_models_configs/dr1_0_QiGuan_Res101_3cls.pkl'

    merge_cfg_from_file(model_cfg)
    cfg.NUM_GPUS = 1
    model_weights = cache_url(model_weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg()
    model = infer_engine.initialize_model_from_cfg(model_weights)

    print('model loaded')
    img_path = 'test.jpg'
    from dr1_0_QiGuan_formal import find_QiGuan
    mask_dict = find_QiGuan(img_path,model)
    if 'No_img' in mask_dict:
        print('No_img')
    elif 'No_result' in mask_dict:
        print('No_result')
    else:
        print('pred done')
        fei_regions = mask_dict['fei_regions']
        fei0 = mask_dict['fei0']
        fei1 = mask_dict['fei1']
        xinying = mask_dict['xinying']
        jizhu = mask_dict['jizhu']
        XinYin,XinYin_Rate,img_out = XinYin_Detect(mask_dict,img_path,img_out='test_out/xinyin_draw.jpg',thether=0.5)
        print('XinYin',XinYin)
        print('XinYin_Rate',XinYin_Rate)
        print('done')
