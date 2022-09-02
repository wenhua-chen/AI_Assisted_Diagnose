import cv2
import numpy as np
from shapely.geometry import LineString, Point
import os

def find_contours(mask,Raw_Return=False):
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
    _, contours1, _ = cv2.findContours(mask_blue1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
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

def find_ymaxmin_pt(coords,loc='max'):
    '''
    income is return of find_contours(mask,Raw_Return=False)
    like [(500, 500), (500, 600), (600, 500), (600, 600)....]

    return the coutours extrame pts in y axle
    like (100,100)
    '''
    a1 = [i[1] for i in coords]
    if loc =='ymax':
        target_idx = a1.index(max(a1))
    else:
        target_idx = a1.index(min(a1))
    # b=a[target_idx:target_idx+100]
    target_pt=coords[target_idx]
    pts_np = np.asarray(target_pt, dtype=np.int32)
    return pts_np.tolist()

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

def draw_pts(in_path=None,pts_line=None,out_path=None,color=None):
    img = cv2.imread(in_path)
    pts_np = np.asarray(pts_line, dtype=np.int32)
    pts = pts_np
    # img_1 = cv2.polylines(img,[pts],False,(0,255,0),5)
    for pt in pts:
        cv2.circle(img,(int(pt[0]), int(pt[1])), 3, color,-1)
    pt1 = pts[0]
    cv2.circle(img,(int(pt1[0]), int(pt1[1])), 3, (0,255,0),-1)
    pt2 = pts[-1]
    cv2.circle(img,(int(pt2[0]), int(pt2[1])), 3, (255,0,0),-1)
    cv2.imwrite(out_path,img)

def draw_pt(in_path=None,pt=None,out_path=None,color=None):
    img = cv2.imread(in_path)
    cv2.circle(img,tuple(pt), 15, color,-1)
    cv2.imwrite(out_path,img)

def draw_line(in_path=None,pt1=None,pt2=None,out_path=None,color=None,distance=None):
    yline1,yline2 = pt1[1],pt2[1]
    xline1,xline2 = pt1[0],pt2[0]
    img = cv2.imread(in_path)
    height,width = img.shape[0],img.shape[1]
    # if xline1 < xline2:
    for i in range(300,10,-1):
        if xline1-i > 0 and xline2-i > 0 and xline1+i < width and xline2+i < width:
            thehold = i
            break
    # cv2.line(img, (xline1-thehold, yline1), (xline1+thehold, yline1), color, 3)
    # cv2.line(img, (xline2-thehold, yline2), (xline2+thehold, yline2), color, 3)

    cv2.line(img, (xline1-thehold, yline1), (xline2+thehold, yline1), color, 3)
    cv2.line(img, (xline1-thehold, yline2), (xline2+thehold, yline2), (42,141,255), 3)
    cv2.putText(img, str(distance)+' pix', (xline2-250, yline2+180), cv2.FONT_HERSHEY_SIMPLEX, 6, (42,141,255),7,cv2.LINE_AA)
    out_path_new = out_path.replace('.jpg','_'+str(distance)+'.jpg')
    os.remove(in_path)
    cv2.imwrite(out_path_new,img,[int(cv2.IMWRITE_JPEG_QUALITY), 30])

def get_fei_bottom_line(mask,start_pt=None,direction=None):
    pts = []
    img_y,img_x = mask.shape[0],mask.shape[1]
    if direction == 'right':
        for width_i in range(start_pt[0],int(img_x*0.5),5):
            pt = []
            line1 = mask[:,width_i,0]
            a1 = np.where(line1==[255])
            if a1[0].size > 0:
                height_i1 = np.max(a1)
                pt.append(width_i)
                pt.append(height_i1)
                pts.append(pt)
    else:
        for width_i in range(start_pt[0],int(img_x*0.5),-5):
            pt = []
            line1 = mask[:,width_i,0]
            a1 = np.where(line1==[255])
            if a1[0].size > 0:
                height_i1 = np.max(a1)
                pt.append(width_i)
                pt.append(height_i1)
                pts.append(pt)
    return pts

def get_xinying_pts(xinying_pts):
    a1 = [i[1] for i in xinying_pts]
    ymin_idx = a1.index(min(a1))
    ymax_idx = a1.index(max(a1))
    xin_right=xinying_pts[ymin_idx:ymax_idx]
    xin_left=xinying_pts[ymax_idx:-1]

    xin_right = np.asarray(xin_right, dtype=np.int32)
    xin_left = np.asarray(xin_left, dtype=np.int32)

    return xin_right.tolist(),xin_left.tolist()

def getCloset_Pt(maskline=None,XinYin_line=None):
    line = LineString(XinYin_line)
    for the in range(100,1000):
        final_ptc=0
        for idx in range(len(maskline)):
            pt = maskline[idx]
            p = Point(pt)
            dis = p.distance(line)
            if dis < the:
                final_pt = pt
                final_ptc+=1
                break
        if final_ptc > 0:
            break
    return final_pt

def get_ymin_line(maskline=None,start_pt=None,dis_pt=None):
    left_s_idx,left_f_idx = maskline.index(dis_pt),maskline.index(start_pt)
    left_select_line = maskline[left_f_idx:left_s_idx+1]
    fin_pt = find_ymaxmin_pt(left_select_line,loc='ymin')
    return fin_pt

def diff2pt_inY(pt1,pt2):
    y1,y2 = pt1[1],pt2[1]
    if y2 > y1:
        return int(y2-y1)
    else:
        return int(y1-y2)

def find_HengGe_diff(mask_dict,orinal_path,img_out=None):

    fei0_mask = mask_dict['fei0']
    fei1_mask = mask_dict['fei1']
    xinying_mask = mask_dict['xinying']

    fei0_pts_series = find_contours(fei0_mask)
    fei1_pts_series = find_contours(fei1_mask)
    xinyin2_pts_series = find_contours(xinying_mask)

    fei0_start_pt = find_ymaxmin_pt(fei0_pts_series,loc='ymax')
    fei1_start_pt = find_ymaxmin_pt(fei1_pts_series,loc='ymax')

    fei0_dire = left_or_right_lung(fei0_mask)
    fei0_line = get_fei_bottom_line(fei0_mask,start_pt=fei0_start_pt,direction=fei0_dire)

    fei1_dire = left_or_right_lung(fei1_mask)
    fei1_line = get_fei_bottom_line(fei1_mask,start_pt=fei1_start_pt,direction=fei1_dire)

    xin_right,xin_left = get_xinying_pts(xinyin2_pts_series)

    if img_out:
        draw_pts(in_path=orinal_path,pts_line=xin_right,out_path=img_out,color=(29,172,188))
        assert os.path.exists(img_out)
        draw_pts(in_path=img_out,pts_line=xin_left,out_path=img_out,color=(29,172,188))
        draw_pts(in_path=img_out,pts_line=fei0_line,out_path=img_out,color=(179,27,204))
        draw_pts(in_path=img_out,pts_line=fei1_line,out_path=img_out,color=(179,27,204))

    if fei0_dire == 'left':
        distin_pt_fei0 = getCloset_Pt(maskline=fei0_line,XinYin_line=xin_left)
        end_pt_fei0 = get_ymin_line(maskline=fei0_line,start_pt=fei0_start_pt,dis_pt=distin_pt_fei0)

        distin_pt_fei1 = getCloset_Pt(maskline=fei1_line,XinYin_line=xin_right)
        end_pt_fei1 = get_ymin_line(maskline=fei1_line,start_pt=fei1_start_pt,dis_pt=distin_pt_fei1)

    else:
        distin_pt_fei0 = getCloset_Pt(maskline=fei0_line,XinYin_line=xin_right)
        end_pt_fei0 = get_ymin_line(maskline=fei0_line,start_pt=fei0_start_pt,dis_pt=distin_pt_fei0)

        distin_pt_fei1 = getCloset_Pt(maskline=fei1_line,XinYin_line=xin_left)
        end_pt_fei1 = get_ymin_line(maskline=fei1_line,start_pt=fei1_start_pt,dis_pt=distin_pt_fei1)

    # print('end_pt_fei0,end_pt_fei1',end_pt_fei0,end_pt_fei1)
    diff2line = diff2pt_inY(end_pt_fei0,end_pt_fei1)
    if img_out:
        draw_pt(in_path=img_out,pt=distin_pt_fei0,out_path=img_out,color=(255,41,133))
        draw_pt(in_path=img_out,pt=distin_pt_fei1,out_path=img_out,color=(72,255,8))
        draw_pt(in_path=img_out,pt=end_pt_fei0,out_path=img_out,color=(138,0,255))
        draw_pt(in_path=img_out,pt=end_pt_fei1,out_path=img_out,color=(42,141,255))
        draw_line(in_path=img_out,pt1=end_pt_fei0,pt2=end_pt_fei1,distance=diff2line,out_path=img_out,color=(138,0,255))


    return diff2line
