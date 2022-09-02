import numpy as np
import cv2
import os



def cal_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


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


def drop_inside_bboxes(boxes, clss):
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
            if min_intersection_ratio >= 0.9:
                del_idx.append(i)

        out_array.append([int(x) for x in group[0][0:4].tolist()] + [group[0][-1]])
        group = np.delete(group, del_idx, axis=0)

    for element in out_array:
        out_cls.append(d[str(element[0:4])])
    return out_array, out_cls



def drop_inside_bboxes_bak(boxes, clss):
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
            iou = cal_iou(group[0][0:4], group[i][0:4])
            if iou > 0.01:
                del_idx.append(i)

        out_array.append([int(x) for x in group[0][0:4].tolist()] + [group[0][-1]])
        group = np.delete(group, del_idx, axis=0)

    for element in out_array:
        out_cls.append(d[str(element[0:4])])
    return out_array, out_cls


if __name__=='__main__':
    boxes_nms = [[2408, 1765, 2433, 1792, 0.22467452], [671, 1651, 743, 1719, 0.12765914], [1846, 668, 1973, 862, 0.09400957], [2195, 1430, 2442, 1587, 0.087488815], [514, 1563, 767, 1770, 0.06873418], [1936, 1525, 2115, 1786, 0.06615681]]
    cls_str_nms = ['BanPianYing', 'BanPianYing', 'BanPianYing', 'BanPianYing', 'BanPianYing', 'BanPianYing']
    drop_inside_bboxes(boxes_nms, cls_str_nms)
# def combine_gaihua(feigaihua_bbox, xgbgaihua_bbox):
    # boxes = 
# boxes = [[2187, 666, 2238, 723, 0.8571314], [2070, 602, 2185, 759, 0.6040176], [572, 1462, 709, 1588, 0.5786457], [2312, 1564, 2482, 1671, 0.1636323], [1916, 1435, 2508, 1854, 0.1369191], [769, 689, 848, 796, 0.07365652]]
# out_array = drop_inside_bboxes(boxes)
# img_path = 'f2e31ab6efeae5e8f3a29519a0960f27.jpg'
# im = cv2.imread(img_path)
# for box_nms in out_array:
#     cv2.rectangle(im, (box_nms[0], box_nms[1]), (box_nms[2], box_nms[3]), (100,200,50),2)
# cv2.imwrite('f2e31.jpg', im)

# # print(cal_iou([1916, 1435, 2508, 1854, 0.1369191], [2312, 1564, 2482, 1671, 0.1636323]))