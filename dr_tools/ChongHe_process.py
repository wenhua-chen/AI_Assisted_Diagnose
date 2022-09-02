from shapely.geometry import Polygon

def pts2box(pt):
    '''
    input [1576, 972, 1633, 1148, 0.9248891]
    output [(1576, 972), (1633, 972), (1633, 1148), (1576, 1148)]
    '''
    box=[]
    xmin,ymin,xmax,ymax=pt[0],pt[1],pt[2],pt[3]
    box.append(tuple([xmin,ymin]))
    box.append(tuple([xmax,ymin]))
    box.append(tuple([xmax,ymax]))
    box.append(tuple([xmin,ymax]))
    return box

def iou2box(box1,box2):
    '''
    input [(1576, 972), (1633, 972), (1633, 1148), (1576, 1148)]
    '''
    a = Polygon(box1)
    b = Polygon(box2)
    # iou = a.intersection(b).area / a.union(b).area
    iou1 = a.intersection(b).area / a.area
    iou2 = a.intersection(b).area / b.area
    if iou1 > iou2:
        return iou1
    if iou2 > iou1:
        return iou2

def compare2RuTouYing(target_box=None,target_cls=None,RuTouYing_box=None,RuTouYing_cls=None):
    if not RuTouYing_box:
        RuTouYing_box = [[0, 0, 10, 10, 0.3]]
    if not RuTouYing_cls:
        RuTouYing_cls = ['Fei_JJ']
    if not target_box or not target_cls:
        return [],[]

    target_box_new,target_cls_new = [],[]
    not_keep = []
    for i in range(len(RuTouYing_box)):
        RuTouYing_box_i = RuTouYing_box[i]
        for j in range(len(target_box)):
            target_box_i = target_box[j]

            box1 = pts2box(target_box_i)
            box2 = pts2box(RuTouYing_box_i)
            iou = iou2box(box1,box2)
            if iou > 0.1:
                not_keep.append(j)

    for k in range(len(target_box)):
        if k not in not_keep:
            target_box_new.append(target_box[k])
            target_cls_new.append(target_cls[k])
    return target_box_new,target_cls_new

def FeiGaihua_filter(FJJ_box=None,FJJ_cls=None,Gaihua_box=None,Gaihua_cls=None):
    if not FJJ_box:
        FJJ_box = [[0, 0, 10, 10, 0.3]]
    if not FJJ_cls:
        FJJ_cls = ['Fei_JJ']
    if not Gaihua_box or not Gaihua_cls:
        return [],[]

    Gaihua_box_new,Gaihua_cls_new = [],[]
    not_keep = []
    for i in range(len(FJJ_box)):
        FJJ_box_i = FJJ_box[i]
        FJJ_str_i = FJJ_cls[i]
        for j in range(len(Gaihua_box)):
            Gaihua_box_i = Gaihua_box[j]

            box1 = pts2box(FJJ_box_i)
            box2 = pts2box(Gaihua_box_i)
            iou = iou2box(box1,box2)
            if iou > 0.1:
                not_keep.append(j)

    for k in range(len(Gaihua_box)):
        if k not in not_keep:
            Gaihua_box_new.append(Gaihua_box[k])
            Gaihua_cls_new.append(Gaihua_cls[k])
    return Gaihua_box_new,Gaihua_cls_new

def BPY_FZK_filter(BanPian_box=None,BanPian_cls=None,FZK_box=None,FZK_cls=None):
    if not BanPian_box:
        BanPian_box = [[0, 0, 10, 10, 0.3]]
    if not BanPian_cls:
        BanPian_cls = ['Fei_JJ']
    if not FZK_box or not FZK_cls:
        return [],[]

    FZK_box_new,FZK_cls_new = [],[]
    not_keep = []
    for i in range(len(BanPian_box)):
        BP_i = BanPian_box[i]
        for j in range(len(FZK_box)):
            FZK_box_i = FZK_box[j]

            box1 = pts2box(BP_i)
            box2 = pts2box(FZK_box_i)
            iou = iou2box(box1,box2)
            if iou > 0.1:
                not_keep.append(j)

    for k in range(len(FZK_box)):
        if k not in not_keep:
            FZK_box_new.append(FZK_box[k])
            FZK_cls_new.append(FZK_cls[k])

    return FZK_box_new,FZK_cls_new

def Banpian_filter(FJJ_box=None,FJJ_cls=None,Gaihua_box=None,Gaihua_cls=None,BanPian_box=None,BanPian_cls=None):
    if not FJJ_box:
        FJJ_box = [[0, 0, 10, 10, 0.3]]
    if not FJJ_cls:
        FJJ_cls = ['Fei_JJ']
    if not Gaihua_box:
        Gaihua_box = [[0, 0, 10, 10, 0.3]]
    if not Gaihua_cls:
        Gaihua_cls = ['Fei_GaiHua']
    if not BanPian_box or not BanPian_cls:
        return [],[]

    BanPian_box_new,BanPian_cls_new = [],[]
    not_keep = []
    for i in range(len(FJJ_box)):
        FJJ_box_i = FJJ_box[i]
        FJJ_str_i = FJJ_cls[i]
        for j in range(len(Gaihua_box)):
            Gaihua_box_i = Gaihua_box[j]
            for k in range(len(BanPian_box)):
                BanPian_box_i = BanPian_box[k]
                box1 = pts2box(FJJ_box_i)
                box2 = pts2box(Gaihua_box_i)
                box3 = pts2box(BanPian_box_i)
                iou13 = iou2box(box1,box3)
                iou23 = iou2box(box2,box3)
                if iou13 > 0.1:
                    not_keep.append(k)
                if iou23 > 0.1:
                    not_keep.append(k)

    # print('not_keep',not_keep)
    for k in range(len(BanPian_box)):
        if k not in not_keep:
            BanPian_box_new.append(BanPian_box[k])
            BanPian_cls_new.append(BanPian_cls[k])
    return BanPian_box_new,BanPian_cls_new

if __name__ == "__main__":
    feijiejie_box = [[828, 1766, 944, 1877, 0.974905]]
    feijiejie_str = ['feijiejie']

    RuTouYing_box = [[2219, 1870, 2366, 2050, 0.9999752044677734], [816, 1763, 956, 1897, 0.9999743700027466]]
    RuTouYing_box_str = ['RuTouYing_cls','RuTouYing_cls']

    Gaihua_box = [[188, 724, 211, 769, 0.38106567]]
    # Gaihua_box = [[1576, 972, 1633, 1148, 0.9248891]]
    FeiGaiHua_str = ['Fei_GaiHua']
    banpian_box = [[188, 724, 211, 769, 0.38106567]]
    banpian_str = ['Banpian']
    # Gaihua_box_new,Gaihua_cls_new = FeiGaihua_filter(FJJ_box=feijiejie_box,FJJ_cls=feijiejie_str,Gaihua_box=Gaihua_box,Gaihua_cls=FeiGaiHua_str)
    # Gaihua_box_new,Gaihua_cls_new = Banpian_filter(FJJ_box=feijiejie_box,FJJ_cls=feijiejie_str,Gaihua_box=Gaihua_box,Gaihua_cls=FeiGaiHua_str,\
    #                                                        BanPian_box=Gaihua_box,BanPian_cls=FeiGaiHua_str)

    Gaihua_box_new,Gaihua_cls_new = compare2RuTouYing(target_box=feijiejie_box,target_cls=feijiejie_str,RuTouYing_box=RuTouYing_box,RuTouYing_cls=RuTouYing_box_str)
    # Gaihua_box_new,Gaihua_cls_new = Banpian_filter(FJJ_box=None,FJJ_cls=None,Gaihua_box=None,Gaihua_cls=None,\
                                                           # BanPian_box=Gaihua_box,BanPian_cls=FeiGaiHua_str)
    print('Gaihua_box',Gaihua_box)
    print('FeiGaiHua_str',FeiGaiHua_str)
    print('Gaihua_box_new',Gaihua_box_new)
    print('Gaihua_cls_new',Gaihua_cls_new)
