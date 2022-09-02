# -*- coding: utf-8 -*-
import timm
import torch
import torch.nn as nn
import cv2
from PIL import Image
import torchvision.transforms as transforms
import shutil
import time
import random
import numpy as np

def init_model(model_path):
    model = timm.create_model('mobilenetv3_100',pretrained=True,num_classes=2,in_chans=3)
    device = torch.device("cuda")
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def trans_downer(xmin,ymin,xmax,ymax):
    cord4 = [str(int(xmin))+','+str(int(ymin)),
             str(int(xmax))+','+str(int(ymin)),
             str(int(xmax))+','+str(int(ymax)),
             str(int(xmin))+','+str(int(ymax))]
    return cord4

def expand_boxes(feimen_crop,img_h,img_w,ratio=[0.4,0.4]):
    [w_ratio,h_ratio] = ratio
    for k,box in enumerate(feimen_crop):
        if len(box)==0:
            continue
        [x1,y1,x2,y2] = box
        box_w,box_h = x2-x1,y2-y1
        y1 -= int(box_h*h_ratio)
        y2 += int(box_h*h_ratio)
        y1 = y1 if y1>0 else 0
        y2 = y2 if y2<img_h else img_h
        if k == 0:
            x1 -= int(box_w*w_ratio)
            x2 += int(box_w*(w_ratio-0.2))
        else:
            x1 -= int(box_w*(w_ratio-0.2))
            x2 += int(box_w*w_ratio)
        x1 = x1 if x1>0 else 0
        x2 = x2 if x2<img_w else img_w
        feimen_crop[k] = [x1,y1,x2,y2]
    return feimen_crop

def your_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def predict(model,img_path,ptsFeiMen):
    device = torch.device("cuda")
    im = cv2.imread(img_path)
    ptsFeiMen = expand_boxes(ptsFeiMen,img_h,img_w)
    fei0 = ptsFeiMen[0]
    fei1 = ptsFeiMen[1]
    [f0_xmin,f0_ymin,f0_xmax,f0_ymax] = fei0
    [f1_xmin,f1_ymin,f1_xmax,f1_ymax] = fei1
    # f0_direction = 'left'
    # f1_direction = 'right'
    f0_im_crop_new = im[f0_ymin:f0_ymax,f0_xmin:f0_xmax]
    f1_im_crop_new = im[f1_ymin:f1_ymax,f1_xmin:f1_xmax]

    if cv2.cvtColor(f0_im_crop_new,cv2.COLOR_BGR2RGB) is not None and cv2.cvtColor(f1_im_crop_new,cv2.COLOR_BGR2RGB) is not None:
        img0 = Image.fromarray(cv2.cvtColor(f0_im_crop_new,cv2.COLOR_BGR2RGB))
        img1 = Image.fromarray(cv2.cvtColor(f1_im_crop_new,cv2.COLOR_BGR2RGB))

        fei0_downer = trans_downer(f0_xmin,f0_ymin,f0_xmax,f0_ymax)
        fei1_downer = trans_downer(f1_xmin,f1_ymin,f1_xmax,f1_ymax)

        cls_list=[0,1]
        sizex,sizey = 224,224
        transform = transforms.Compose([
            transforms.Resize([sizex,sizey]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        results,preds,scores = [],[0,0],[]
        cls_name = 'FeiMenZDZN'
        for i in [[img0,fei0_downer],[img1,fei1_downer]]:
            img = i[0]
            fei_downer = i[1]
            img_tensor = transform(img)
            img_tensor.unsqueeze_(0)
            img_tensor.shape
            img_tensor = img_tensor.to(device)
            with torch.no_grad():
                y_predict = model(img_tensor)
            out = torch.max(y_predict,1)[1]
            # print('y_predict',y_predict.data.cpu().numpy().tolist()[0])
            score_softmax = y_predict.data.cpu().numpy().tolist()[0]
            score_softmax = your_softmax(score_softmax)

            score_softmax = score_softmax.tolist()
            pred = score_softmax.index(max(score_softmax))
            score = round(score_softmax[pred],3)
            if pred==1:
                one_item={}
                one_item['Cords4'] = fei_downer
                one_item['Name'] = cls_name
                one_item['Score'] = score
                results.append(one_item)
    return results

if __name__ == "__main__":
    import os
    import torch
    import torch.nn as nn
    from tqdm import tqdm

    model_path='/data/steven/project/Object_Detection_coastal/dr_wrapper/DR_models_configs/dr2_8_FeiMenZD_formal.pth'

    print('loading model....')
    timea=time.time()
    model = init_model(model_path)
    timeb=time.time()
    print('model loaded, time: ',timeb-timea,' s')

    single_test=False
    if single_test:
        pred = predict(model,img_path,ptsFeiMen=pts1)
        print('pts1',pts1)
        print('pred',pred)
        timec=time.time()
        print('model loaded, time: ',timec-timeb,' s')
    else:
        import tensorflow as tf
        from dr1_10_8in1_crop_formal import init_sess,find_model
        tfmodel = '/data/steven/project/Object_Detection_coastal/dr_wrapper/DR_models_configs/dr1_10_8in1_crop_formal.ckpt'
        sess, net = init_sess(tfmodel)
        print('Loaded network {:s}'.format(tfmodel))

        input_dir = '/data/steven/project/Object_Detection_coastal/dataser_raw/1_rutouying/images_raw_anno/0913_zhuyisheng_test_imgs'
        img_list = os.listdir(input_dir)
        for i in tqdm(range(len(img_list))):
            img_name = img_list[i]
            # if i>1:
            #     break
            img_path = os.path.join(input_dir,img_name)
            output_dict = find_model(sess, net, img_path)
            feimen_crop = [output_dict['left_feimen_crop'],output_dict['right_feimen_crop']]
            pred = predict(model,img_path,ptsFeiMen=feimen_crop)
            print('pred: ', pred)
