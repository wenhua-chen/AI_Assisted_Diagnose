from __future__ import division

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import math
from random import randint

# img_in = 'PN025067.jpg'
# img_in_real = 'PN025067_img.jpg'
# img_out = 'PN025067_2.jpg'

# mask_path = 'mask/'
# img_path = 'img/'
# mask_list = os.listdir(mask_path)
# img_list = os.listdir(img_path)

def get_angeles(a,b,c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    if cosine_angle <-1:
        cosine_angle=-1
    angle = np.arccos(cosine_angle)
    angle_d = np.degrees(angle)
    angle_jus = abs(180 - angle_d)
    return angle_jus

def get_vertcal_pts_simply(p1,p2,img_size):
    x1,x2,y1,y2 = p1[0],p2[0],p1[1],p2[1]
    x3,y3 = (x1 + x2)/2, (y1 + y2)/2
    if x1 == x2:
        x4 = x3+int(img_size[1]/7)
        y4 = y3
        return [x3,y3],[x4,y4]
    else:
        m1 = (y2-y1)/(x2-x1)
        m2 = -1/m1
        x4 = x3-int(img_size[1]/7)
        y4 = m2*(x4-x3)+y3
        return [x3,y3],[x4,y4]

def get_vertcal_pts(p1,p2,pen_length,direction=None):
    x1,x2,y1,y2 = p1[0],p2[0],p1[1],p2[1]
    x3,y3 = (x1 + x2)/2, (y1 + y2)/2

    m1 = (y2-y1)/(x2-x1)
    m2 = -1/m1

    if direction == 'left':
        x4 = x3+pen_length
    else:
        x4 = x3-pen_length
    y4 = m2*(x4-x3)+y3
    return [x3,y3],[x4,y4]

def get_intersection(p1,p2,p3,p4):
    def line(p1, p2):
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0]*p2[1] - p2[0]*p1[1])
        return A, B, -C

    L1 = line(p1,p2)
    L2 = line(p3,p4)

    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False

def get_vertcal_pts2(p1,p2,y,Up=True,direction=None,the=50):
    x1,x2,y1,y2 = p1[0],p2[0],p1[1],p2[1]
    x3,y3 = (x1 + x2)/2, (y1 + y2)/2

    if y2 != y1 and x2 != x1:
        m1 = (y2-y1)/(x2-x1)
    else:
        m1 = 0.000001

    m2 = -1/m1
    for i in range(0,1000,1):
        if direction =='left':
            if Up:
                if y2 > y1:
                    x4 = x3-i
                else:
                    x4 = x3+i
            else:
                if y2 > y1:
                    x4 = x3+i
                else:
                    x4 = x3-i
        else:
            if Up:
                if y2 > y1:
                    x4 = x3+i
                else:
                    x4 = x3-i
            else:
                if y2 > y1:
                    x4 = x3-i
                else:
                    x4 = x3+i
        y4 = m2*(x4-x3)+y3

        if Up:
            if y4 + the > y:
                break
        else:
            if y4 - the < y:
                break
    return [x3,y3],[x4,y4],m2

def get_trangle_pts(p1,p2,img_size,trginle_size,direction=None,position=None):
    x1,x2,y1,y2 = p1[0],p2[0],p1[1],p2[1]
    x3,y3 = (x1 + x2)/2, (y1 + y2)/2

    if y2 != y1 and x2 != x1:
        m1 = (y2-y1)/(x2-x1)
    else:
        m1 = 0.000001

    m2 = -1/m1
    x_m2 = x3+int(img_size[1]/7)
    y_m2 = m2*(x_m2-x3)+y3

    m3 = (m1*m2-1+math.sqrt(m1*m1+1)*math.sqrt(m2*m2+1))/(m1+m2)

    if direction == 'left':
        x4 = x3+trginle_size
        if position in [1,2]:
            if y1 < y2:
                y4 = -1/m3*(x4-x3)+y3
            else:
                y4 = m3*(x4-x3)+y3
        elif position in [3,4]:
            if y2 < y1:
                y4 = -1/m3*(x4-x3)+y3
            else:
                y4 = m3*(x4-x3)+y3
    else:
        x4 = x3-trginle_size
        if position in [1,2]:
            if y1 < y2:
                y4 = -1/m3*(x4-x3)+y3
            else:
                y4 = m3*(x4-x3)+y3
        elif position in [3,4]:
            if y2 < y1:
                y4 = -1/m3*(x4-x3)+y3
            else:
                y4 = m3*(x4-x3)+y3

    x5,y5=perpen_point(p1,p2,[x4,y4])
    x6,y6=perpen_point([int(x3),int(y3)],[int(x_m2),int(y_m2)],[x4,y4])

    spt1 = [int(x3),int(y3)]
    spt3 = [int(x4),int(y4)]
    spt2 = [int(x5),int(y5)]
    spt4 = [int(x6),int(y6)]
    spts = np.asarray([spt1,spt4,spt3,spt2],dtype=np.int32)
    return spts

def perpen_point(p1,p2,p3):
    x1,y1,x2,y2,x3,y3=p1[0],p1[1],p2[0],p2[1],p3[0],p3[1],
    px = x2-x1
    py = y2-y1
    dAB = px*px + py*py
    u = ((x3 - x1) * px + (y3 - y1) * py) / dAB
    x4 = x1 + u * px,
    y4 = y1 + u * py
    return x4[0],y4

def get_angles_4pts(p1,p2,p3,p4,img_size):
    if p1[0] == p2[0] == p1[0] == p2[0]:
        return False
    else:
        v1,v2 = get_vertcal_pts_simply(p1,p2,img_size)
        v3,v4 = get_vertcal_pts_simply(p3,p4,img_size)

        inters = get_intersection(v1,v2,v3,v4)
        if inters:
            inters = [inters[0],inters[1]]
            inters = np.asarray(inters)
            angle_d = get_angeles(v1,inters,v3)
            if angle_d > 90:
                angle_d = 180 - angle_d
            return angle_d
        else:
            return False

# def selected_index_filter(selected_index,permutations_idx_np,all_angles_np):
#     selected_idx_adj=[selected_index[0]]
#     for i in selected_index:
#         if i in selected_idx_adj:
#             continue
#         line1 = permutations_idx_np[i][0]
#         line2 = permutations_idx_np[i][1]
#         angle1 = all_angles_np[i]
#         for i_1 in range(len(selected_idx_adj)):
#             j=selected_idx_adj[i_1]
#             line1_adj = permutations_idx_np[j][0]
#             line2_adj = permutations_idx_np[j][1]
#             angle_agj = all_angles_np[j]
#             if line1.tolist() == line1_adj.tolist():
#                 if angle1 > angle_agj:
#                     if i not in selected_idx_adj:
#                         selected_idx_adj.pop(i_1)
#                         selected_idx_adj.append(i)
#                 else:
#                     continue
#             elif line2.tolist() == line2_adj.tolist():
#                 if angle1 > angle_agj:
#                     print(2)
#                     if i not in selected_idx_adj:
#                         selected_idx_adj.pop(i_1)
#                         selected_idx_adj.append(i)
#                 else:
#                     continue
#             else:
#                 if i not in selected_idx_adj:
#                     selected_idx_adj.append(i)
#     return selected_idx_adj

def selected_index_filter(selected_idx_adj,selected_index,permutations_idx_np):
    n_pass = 0
    line1_t = permutations_idx_np[selected_index][0].tolist()
    line2_t = permutations_idx_np[selected_index][1].tolist()
    for i in selected_idx_adj:
        line1_i = permutations_idx_np[i][0].tolist()
        line2_i = permutations_idx_np[i][1].tolist()
        if line1_t == line1_i or line2_t == line2_i or line1_t == line1_i or line2_t == line1_i:
            n_pass += 1
    if n_pass > 0:
        return selected_idx_adj
    else:
        selected_idx_adj.append(selected_index)
        return selected_idx_adj

def cobb_procss(mask_dict,img_in_real,img_out=None,thether=10):
    num_segs = 17
    angle_the = 0.5
    n_maximum = 3
    draw_supplyline = True
    top_angles = []
    mask_in=mask_dict['jizhu']
    # img = cv2.imread(mask_in)
    img = mask_in
    img_real = cv2.imread(img_in_real)
    img_size = img.shape

    if not img_out:
        filename, file_extension = os.path.splitext(img_in_real)
        img_out = '/home/steven/mask_rcnn_caffe2/Detectron-xRay/Wrapper/Final_Out/'+filename+'_JiZhuCheWan.jpg'

    pts=[]
    '''
    #0, Get the mid points
    start the points beyond upper and donner a little
    then out the points' both side by some rae (30 here)
    '''
    # upper_shorter,down_shorter = int(img_size[0]/15),int(img_size[0]/15)
    upper_shorter,down_shorter = int(5),int(5)
    for height_i in range(upper_shorter,img_size[0]-down_shorter,int(img_size[0]/num_segs)):
        pt = []
        line = img[height_i,:,0]
        a = np.where(line==[255])
        if a[0].size > 0:
            width_t = np.median(a)
            pt.append(width_t)
            pt.append(height_i)
            pts.append(pt)

    pts_np = np.asarray(pts, dtype=np.int32)
    upper_shorter,down_shorter = int(len(pts_np)/30),int(len(pts_np)/30)
    pts_np = pts_np[upper_shorter:len(pts_np)-down_shorter]

    #draw it
    pts = pts_np
    img_1 = cv2.polylines(img_real,[pts],False,(0,255,0),3)

    for pt in pts:
        cv2.circle(img_real,(int(pt[0]), int(pt[1])), 5, (255,0,0),-1)

    '''
    1, get angles between lines, find the max angles most curve
    '''
    lines_pts,lines_pts_idx = [],[]
    for i in range(len(pts)-1):
        pt1 = pts[i]
        pt2 = pts[i+1]
        a,b=[],[]
        a.append(pt1)
        a.append(pt2)
        lines_pts.append(a)
        b.append(i)
        b.append(i+1)
        lines_pts_idx.append(b)

    from itertools import combinations
    permutations = list(combinations(lines_pts, 2))
    permutations_idx = list(combinations(lines_pts_idx, 2))

    all_angles = []
    for four_pts in permutations:
        pt1 = four_pts[0]
        pt2 = four_pts[1]

        p1,p2 = pt1[0],pt1[1]
        p3,p4 = pt2[0],pt2[1]
    #     print(p1,p2,p3,p4)
        if p4[1] >= p3[1] >= p2[1] >= p1[1]:
    #         print(p1,p2,p3,p4)
            angel = get_angles_4pts(p1,p2,p3,p4,img_size)
            all_angles.append(angel)
        else:
            print('p4[1] >= p3[1] >= p2[1] >= p1[1]')
            print(p1,p2,p3,p4)
            # all_angles.append(angel)

    all_angles_np = np.asarray(all_angles)
    permutations_np = np.asarray(permutations)
    permutations_idx_np = np.asarray(permutations_idx)
    all_angles_np = np.nan_to_num(all_angles_np)

    selected_idx_adj = [all_angles_np.argsort()[-1:][::-1][0]]
    for i in range(2,50,1):
        selected_index = (-all_angles_np).argsort()[:i][-1]
    #     print(selected_index)
        selected_idx_adj = selected_index_filter(selected_idx_adj,selected_index,permutations_idx_np)
    #     print(selected_idx_adj)
        if len(selected_idx_adj) >= n_maximum:
            break


    selected_max = permutations_np[selected_idx_adj]
    selected_idx = permutations_idx_np[selected_idx_adj]
    selected_angles = all_angles_np[selected_idx_adj]

    '''
    4, perpendicular of two points of each group
     1   __   2
        |__|
     3        4
    '''
    pen_lengths = [int(img_size[1]/7),int(img_size[1]/6),int(img_size[1]/5),int(img_size[1]/8),int(img_size[1]/9)]
    trginle_sizes = [25,30,35,40,45]
    colorslist = [
        (0,154,255), #orange
        (255,191,0), #Deep Sky Blue
        (147,20,255), #Deep Pink
        (34,139,34), #Forest Green
        (240,32,160), #Purple
                  ]

    for i in range(len(selected_max)):
        a = selected_max[i][0].tolist()+selected_max[i][1].tolist()
        pts_np = np.asarray(a, dtype=np.int32)
        for pt in pts_np:
            cv2.circle(img_real,(int(pt[0]), int(pt[1])), 5, (0,0,255),-1)

        color = colorslist[i]
        line_width1 = 3
        line_width2 = 2
        cv2.line(img_real, (pts_np[0][0], pts_np[0][1]), (pts_np[1][0], pts_np[1][1]), color, line_width1)
        cv2.line(img_real, (pts_np[2][0], pts_np[2][1]), (pts_np[3][0], pts_np[3][1]), color, line_width1)

        if pts_np[1][0] == pts_np[0][0]:
            pts_np[1][0] += 1
        if pts_np[3][0] == pts_np[2][0]:
            pts_np[3][0] += 1

        if i % 2 == 0:
            dire = 'left'
        else:
            dire = 'right'

        # if pts_np[1][0] < pts_np[0][0]:
        #     dire = 'left'
        # elif pts_np[1][0] > pts_np[0][0]:
        #     dire = 'right'

        pen_length = pen_lengths[i]
        v1,v2 = get_vertcal_pts(pts_np[0],pts_np[1],pen_length,direction=dire)
        v3,v4 = get_vertcal_pts(pts_np[2],pts_np[3],pen_length,direction=dire)

        cv2.line(img_real, (int(v1[0]), int(v1[1])), (int(v2[0]), int(v2[1])), color, line_width1)
        cv2.line(img_real, (int(v3[0]), int(v3[1])), (int(v4[0]), int(v4[1])), color, line_width1)

        v1h,v2h,m2u = get_vertcal_pts2(v1,v2,v4[1],Up=True,direction=dire,the=50)
        v3h,v4h,m2d = get_vertcal_pts2(v3,v4,v2[1],Up=False,direction=dire,the=50)

        inters = get_intersection(v1h,v2h,v3h,v4h)

        if inters:
            x4u = v1h[0]+(inters[1]+150-v1h[1])/m2u
            x4d = v3h[0]+(inters[1]-150-v3h[1])/m2d
            if draw_supplyline:
                cv2.line(img_real, (int(v1h[0]), int(v1h[1])), (int(x4u), int(inters[1]+100)), color, line_width2)
                cv2.line(img_real, (int(v3h[0]), int(v3h[1])), (int(x4d), int(inters[1]-100)), color, line_width2)

            inters = [inters[0],inters[1]]
            if [int(v1h[0]),int(v1h[1])] == [int(inters[0]),int(inters[1])] or [int(v3h[0]),int(v3h[1])] == [int(inters[0]),int(inters[1])] or [int(v1h[0]),int(v1h[1])] == [int(v3h[0]),int(v3h[1])]:
                angle_d = 0.01
            else:
                angle_d = get_angeles(v1h,np.asarray(inters),v3h)

            if angle_d > 90:
                angle_d = abs(180-angle_d)

            top_angles.append(angle_d)
            if dire == 'right':
                cv2.putText(img_real, str(round(angle_d,2))+' d', (int(v2[0]),int(v1[1])+int((v3[1]-v1[1])/2)), cv2.FONT_HERSHEY_SIMPLEX, 3, color,3,cv2.LINE_AA)
            else:
                cv2.putText(img_real, str(round(angle_d,2))+' d', (int(inters[0]),int(inters[1])+10), cv2.FONT_HERSHEY_SIMPLEX, 3, color,3,cv2.LINE_AA)

            trginle_size = trginle_sizes[i]
            spts_P1 = get_trangle_pts(pts_np[0],pts_np[1],img_size,trginle_size,direction=dire,position=1,)
            spts_P2 = get_trangle_pts(v1,v2,img_size,trginle_size,direction=dire,position=2)
            spts_P3 = get_trangle_pts(v3,v4,img_size,trginle_size,direction=dire,position=3)
            spts_P4 = get_trangle_pts(pts_np[2],pts_np[3],img_size,trginle_size,direction=dire,position=4)

            img_real = cv2.polylines(img_real,[spts_P1],False,(255,0,0),1)
            img_real = cv2.polylines(img_real,[spts_P3],False,(255,0,0),1)

            if draw_supplyline:
                img_real = cv2.polylines(img_real,[spts_P2],False,(255,0,0),1)
                img_real = cv2.polylines(img_real,[spts_P4],False,(255,0,0),1)
        else:
            continue

    cv2.imwrite(img_out,img_real,[int(cv2.IMWRITE_JPEG_QUALITY), 30])
    if max(top_angles)>thether:
        JiZhuResult = 'JiZhuCheWan_Yes'
    else:
        JiZhuResult = 'JiZhuCheWan_No'
    return JiZhuResult,top_angles,img_out

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
    img_path = '/home/steven/mask_rcnn_caffe2/Detectron-xRay/Wrapper/imgs/PN039833.jpg'
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
        JiZhuResult,top_angles,img_out = cobb_procss(mask_dict,img_path,img_out='test_out/jizhu_draw.jpg',thether=10)
        print('JiZhuResult',JiZhuResult)
        print('top_angles',top_angles)
        print('done')
