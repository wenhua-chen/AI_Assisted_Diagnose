from shapely.geometry import Point, Polygon
import cv2
import numpy as np
from scipy.spatial.distance import pdist
import disease_desc

def final_process(boxes,cls_str,QiGuan_return=None,segmts=None,ye_out=None,region_within_filter=True,XinYin_within_filter=False):
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

    def get_XinYin_Fer_area(QiGuan_return):
        # fei0 = QiGuan_return['fei0_contours_raw_return']
        # fei1 = QiGuan_return['fei1_contours_raw_return']
        # xin1 = QiGuan_return['xinying_contours_raw_return']

        # if isinstance(fei0, basestring):
        #     mask1 = cv2.imread(fei0)
        #     mask2 = cv2.imread(fei1)
        #     mask3 = cv2.imread(xin1)
        #
        # mask1 = fei0
        # mask2 = fei1
        # mask3 = xin1

        # contours1 = find_contours(mask1)
        # contours2 = find_contours(mask2)
        # contours3 = find_contours(mask3)

        contours1 = QiGuan_return['fei0_contours_raw_return']
        contours2 = QiGuan_return['fei1_contours_raw_return']
        contours3 = QiGuan_return['xinying_contours_raw_return']


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

    def check_polygons_within(coords_big,coords2_small):
        '''
        input:
            coords_big:[(500, 500), (500, 600), (600, 500), (600, 600)]
            coords2_small:[(500, 500), (500, 600), (600, 500), (600, 600)]
        output:
            True/False
        '''
        poly1 = Polygon(coords_big)
        poly2 = Polygon(coords2_small)
        TorF = poly1.intersects(poly2)
        return TorF

    def calculate_intersection(coords_big,coords2_small,theath=0.3):
        # cords = trans_box(coords2_small,full_boxes=False)
        p1 = Polygon(coords_big)
        p2 = Polygon(coords2_small)
        inter_sec_area = p1.intersection(p2).area
        overlap = inter_sec_area/p2.area
        if overlap > theath:
            return 'True'
        else:
            return 'False'

    def trans_box(boxes,full_boxes=True):
        '''
        full_boxes means a list of box
        input:
            [[7.69001038e+02 1.44156030e+03 8.20211121e+02 1.48525391e+03 6.36455059e-01]
                [8.27617493e+02 1.53489282e+03 9.54969788e+02 1.62012524e+03 8.95488933e-02]]
        output:
            [[(769.001038, 1441.5603),(820.211121, 1441.5603),
              (769.001038, 1485.25391),(820.211121, 1485.25391)],
             [(827.617493, 1534.89282),(954.969788, 1534.89282),
              (827.617493, 1620.12524),(954.969788, 1620.12524)]]
        '''
        if full_boxes:
            boxes_4cords = []
            for box in boxes:
                box_i = []
                pts = box[:4]
                xmin,ymin,xmax,ymax = pts[0],pts[1],pts[2],pts[3]
                p1=(pts[0],pts[1])
                p2=(pts[2],pts[1])
                p3=(pts[0],pts[3])
                p4=(pts[2],pts[3])
                box_i.append(p1)
                box_i.append(p2)
                box_i.append(p3)
                box_i.append(p4)
                boxes_4cords.append(box_i)
            return boxes_4cords
        else:
            box_i = []
            pts = boxes[:4]
            xmin,ymin,xmax,ymax = pts[0],pts[1],pts[2],pts[3]
            p1=(pts[0],pts[1])
            p2=(pts[2],pts[1])
            p3=(pts[0],pts[3])
            p4=(pts[2],pts[3])
            box_i.append(p1)
            box_i.append(p2)
            box_i.append(p3)
            box_i.append(p4)
            return box_i

    def check_box_in_area(mask_path,box):
        if isinstance(mask_path, basestring):
            mask = cv2.imread(mask_path)
            coords_big = find_contours(mask,Raw_Return=False)
        else:
            coords_big = mask_path
        cords = trans_box(box,full_boxes=False)
        result = check_polygons_within(coords_big,cords)
        return result

    def left_or_right_lung(mask_path):
        if isinstance(mask_path, basestring):
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

    def get_longest_line_from_polygon(str_coutrs):
        import ast
        cords_new = []
        for str_i in str_coutrs:
            tuple_cord = ast.literal_eval(str_i)
            cords_new.append(tuple_cord)
        x = np.array(cords_new)
        max_length = np.max(pdist(x))
        return max_length

    def divded_dai(img_input,total_pts=34,n_pts_ingnore=5,n_up_ingnore=1):
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
        if isinstance(img_input, basestring):
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
        return region1,region2,region3

    def check_dai_region(box,region1,region2,region3,out_path=None,draw=False):
        '''
        find the max intersetion between box and regions
        input:
            three regions from divded_dai
            box=[(500, 500),(500, 600),(600, 600), (600, 500)]
        output:
            int, 1,2,3 = left to right
        '''
        p_box=Polygon(box)
        inters = []
        p1=Polygon(region1)
        p_box = p_box.buffer(0)
        # print('box',box)
        # print('region1',region1)
        # print('p_box',p_box.is_valid)
        # print('p1',p1.is_valid)
        p1_inter=p_box.intersection(p1)

        p2=Polygon(region2)
        p2_inter=p_box.intersection(p2)

        p3=Polygon(region3)
        p3_inter=p_box.intersection(p3)

        if draw:
            # print('draw dai')
            oeginal_img_path = 'test.jpg'
            orginal_img = cv2.imread(oeginal_img_path)
            # print('region1',region1)
            orginal_img=cv2.drawContours(orginal_img, [np.array(region1)], -1, (0,180,0), 10)
            orginal_img=cv2.drawContours(orginal_img, [np.array(region2)], -1, (0,0,180), 10)
            orginal_img=cv2.drawContours(orginal_img, [np.array(region3)], -1, (150,0,0), 10)
            cv2.imwrite(out_path,orginal_img)

        inters.append(p1_inter.area)
        inters.append(p2_inter.area)
        inters.append(p3_inter.area)
        # print('inters',inters)
        which_region = inters.index(max(inters))
        return which_region

    def check_ye_region(box,ye_boundry,dire):
        if dire == 'left':
            y2line = ye_boundry['left2']
            y4line = ye_boundry['left4']
        else:
            y2line = ye_boundry['right2']
            y4line = ye_boundry['right4']

        target_ymax = box[3][1]
        if target_ymax < y2line:
            location = 'top'
            location_int = 0
        elif y2line <= target_ymax <= y4line:
            location = 'mid'
            location_int = 1
        elif y4line < target_ymax:
            location = 'bottom'
            location_int = 2
        return location,location_int

    def position_list_dai(str):
        if str == 'Nei':
            return 0
        elif str == 'Zhong':
            return 1
        elif str == 'Wai':
            return 2

    def combine_results(cls_str,boxes,QiGuan_returns=None,segmts2=None,ye_boundry=None):
        final_results = []
        cord4s = []
        # dai_right_list = ['Nei','Zhong','Wai']
        dai_right_list = ['Wai','Zhong','Nei']
        dai_left_list = ['Nei','Zhong','Wai']
        # assert ye_boundry
        for i in range(len(boxes)):
            one_item={}
            cls_name = cls_str[i]
            box = boxes[i]
            position_list = []
            if 'fei_loca_info0' in QiGuan_returns and 'fei_loca_info1' in QiGuan_returns:
                fei0 = 	QiGuan_returns['fei_loca_info0']
                fei1 = 	QiGuan_returns['fei_loca_info1']

                #0 Check direction0
                direction0 = fei0[0]
                direction1 = fei1[0]
                # direction0 = left_or_right_lung(fei0)

                if direction0 == 'left':
                    coords_big0 = fei0[1]
                    In_Not = check_box_in_area(coords_big0,box)
                    # print('left0',In_Not)
                    if In_Not:
                        position_list.append(0)
                        # print('direction0_left')
                        one_item['Loctin_dire']='Left'
                        box_style = trans_box(box,full_boxes=False)
                        if ye_boundry:
                            ye_loac,location_int = check_ye_region(box_style,ye_boundry,direction0)
                            one_item['Loctin_Ye'] = ye_loac
                            position_list.append(location_int)
                        else:
                            one_item['Loctin_Ye'] = 'None'
                            position_list.append('None')

                        region1,region2,region3 = fei0[2],fei0[3],fei0[4]
                        int_dai = check_dai_region(box_style,region1,region2,region3,out_path='left_dai0.jpg')
                        one_item['Loctin_Dai'] = dai_left_list[int_dai]
                        dai_int = position_list_dai(dai_left_list[int_dai])
                        position_list.append(dai_int)
                else:
                    coords_big1 = fei0[1]
                    In_Not = check_box_in_area(coords_big1,box)
                    if In_Not:
                        position_list.append(1)
                        one_item['Loctin_dire']='Right'
                        box_style = trans_box(box,full_boxes=False)

                        if ye_boundry:
                            ye_loac,location_int = check_ye_region(box_style,ye_boundry,direction0)
                            one_item['Loctin_Ye'] = ye_loac
                            position_list.append(location_int)
                        else:
                            one_item['Loctin_Ye'] = 'None'
                            position_list.append('None')

                        region1,region2,region3 = fei0[2],fei0[3],fei0[4]
                        int_dai = check_dai_region(box_style,region1,region2,region3,out_path='right_dai0.jpg')
                        one_item['Loctin_Dai'] = dai_right_list[int_dai]
                        dai_int = position_list_dai(dai_right_list[int_dai])
                        position_list.append(dai_int)

                if direction1 == 'left':
                    coords_big0 = fei1[1]
                    In_Not = check_box_in_area(coords_big0,box)
                    # print('left1',In_Not)
                    if In_Not:
                        print('left2',In_Not)
                        position_list.append(0)
                        one_item['Loctin_dire']='Left'
                        box_style = trans_box(box,full_boxes=False)
                        if ye_boundry:
                            ye_loac,location_int = check_ye_region(box_style,ye_boundry,direction1)
                            one_item['Loctin_Ye'] = ye_loac
                            position_list.append(location_int)
                        else:
                            one_item['Loctin_Ye'] = 'None'
                            position_list.append('None')

                        region1,region2,region3 = fei1[2],fei1[3],fei1[4]
                        int_dai = check_dai_region(box_style,region1,region2,region3,out_path='left_dai1.jpg')
                        one_item['Loctin_Dai'] = dai_left_list[int_dai]
                        dai_int = position_list_dai(dai_left_list[int_dai])
                        position_list.append(dai_int)
                else:
                    coords_big1 = fei1[1]
                    In_Not = check_box_in_area(coords_big1,box)
                    # print('right1',In_Not)
                    if In_Not:
                        position_list.append(1)
                        one_item['Loctin_dire']='Right'
                        box_style = trans_box(box,full_boxes=False)

                        if ye_boundry:
                            ye_loac,location_int = check_ye_region(box_style,ye_boundry,direction1)
                            one_item['Loctin_Ye'] = ye_loac
                            position_list.append(location_int)
                        else:
                            one_item['Loctin_Ye'] = 'None'
                            position_list.append('None')

                        region1,region2,region3 = fei1[2],fei1[3],fei1[4]
                        int_dai = check_dai_region(box_style,region1,region2,region3,out_path='right_dai1.jpg')
                        one_item['Loctin_Dai'] = dai_right_list[int_dai]
                        dai_int = position_list_dai(dai_right_list[int_dai])
                        position_list.append(dai_int)

            if 'Loctin_dire' not in one_item:
                one_item['Loctin_dire']= 'None'
                one_item['Loctin_Ye'] = 'None'
                one_item['Loctin_Dai'] = 'None'
                position_list = ['None','None','None']
            assert len(position_list) == 3
            position_list_short = position_list[:2]
            # try:
            if cls_name == 'FeiJieJie':
                d_desc = disease_desc.feijiejie(position_list)
            elif cls_name == 'BanPianYing':
                d_desc = disease_desc.banpianying(position_list_short)
            elif cls_name == 'Fei_GaiHua':
                d_desc = disease_desc.feigaihua(position_list_short)
            elif cls_name == 'feizhongkuai':
                d_desc = disease_desc.feizhongkuai(position_list_short)
            else:
                d_desc = ''
                print('d_desc not in list: ',d_desc)
            # except:
            #     d_desc = ''
            one_item['Disease_Desc'] = d_desc
            xmin,ymin,xmax,ymax,score_i = box[0],box[1],box[2],box[3],box[4]
            cord4 = [str(int(xmin))+','+str(int(ymin)),
                     str(int(xmax))+','+str(int(ymin)),
                     str(int(xmax))+','+str(int(ymax)),
                     str(int(xmin))+','+str(int(ymax))]

            one_item['Name']=cls_name
            one_item['Cords4']=cord4
            one_item['Score']=score_i

            width_d,height_d = int(xmax-xmin),int(ymax-ymin)
            if width_d > height_d:
                one_item['Length'] = width_d
            else:
                one_item['Length'] = height_d
            if segmts:
                seg = segmts[i]
                one_item['pts']=seg
                if cls_name == 'FeiJieJie':
                    length = get_longest_line_from_polygon(segmts2[i])
                    one_item['Length']=length
            final_results.append(one_item)
        return final_results

    if region_within_filter:
        if QiGuan_return is not None and 'XinYin_fei' in QiGuan_return:
            # coords_big = get_XinYin_Fer_area(QiGuan_return)
            coords_big = QiGuan_return['XinYin_fei']
            boxes_4cords = trans_box(boxes,full_boxes=True)
            for i in range(len(boxes_4cords)):
                TorF = check_polygons_within(coords_big,boxes_4cords[i])
                if TorF:
                    pass
                else:
                    del boxes[i]
                    del cls_str[i]
                    if segmts:
                        del segmts[i]

    if XinYin_within_filter:
        if QiGuan_return is not None and 'XinYin_contour' in QiGuan_return:
            coords_big = QiGuan_return['XinYin_contour']
            # print('boxes',boxes)
            boxes_4cords = trans_box(boxes,full_boxes=True)
            for i in range(len(boxes_4cords)):
                TorF = calculate_intersection(coords_big,boxes_4cords[i],theath=0.3)
                if TorF:
                    pass
                else:
                    del boxes[i]
                    del cls_str[i]
                    if segmts:
                        del segmts[i]

    if segmts:
        final_results = combine_results(cls_str,boxes,QiGuan_returns=QiGuan_return,segmts2=segmts,ye_boundry=ye_out)
    else:
        final_results = combine_results(cls_str,boxes,QiGuan_returns=QiGuan_return,ye_boundry=ye_out)
    return final_results
