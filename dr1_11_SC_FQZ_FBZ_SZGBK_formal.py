# -*- coding:utf-8 -*- 
# Author: 陈文华(Steven)
# Website: https://wenhua-chen.github.io/
# Github: https://github.com/wenhua-chen
# Date: 2020-03-16 03:59:32
# LastEditTime: 2022-09-02 15:47:43
# Description: 

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
    model = timm.create_model('mobilenetv3_100',pretrained=True,num_classes=4,in_chans=3)
    model.classifier = nn.Sequential(nn.Linear(1280, 4),nn.Sigmoid())
    device = torch.device("cuda")
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model,img_path):
    device = torch.device("cuda")
    im = cv2.imread(img_path)
    sizex,sizey = 224,224
    transform = transforms.Compose([
        transforms.Resize([sizex,sizey]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    results = []
    class_names = ['ShenChu','FQiZ','FeiBuZhang','ShangZongGeBK']
    all_thresholds = [0.8, 0.08, 0.05, 0.29]
    if cv2.cvtColor(im,cv2.COLOR_BGR2RGB) is not None:
        img = Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
        img_tensor = transform(img)
        img_tensor.unsqueeze_(0)
        img_tensor.shape
        img_tensor = img_tensor.to(device)
        with torch.no_grad():
            out = model(img_tensor)
        predlist = out.data.cpu().numpy().tolist()
        for i,cls_name in enumerate(class_names):
            if predlist[0][i] > all_thresholds[i]:
                results.append([cls_name,predlist[0][i]])
    return results

if __name__ == "__main__":
    import os
    import torch
    import torch.nn as nn
    from tqdm import tqdm

    model_path='/data/steven/project/Object_Detection_coastal/dr_wrapper/DR_models_configs/dr1_11_SC_FQZ_FBZ_SZGBK_formal.pth'

    print('loading model....')
    timea=time.time()
    model = init_model(model_path)
    timeb=time.time()
    print('model loaded, time: ',timeb-timea,' s')

    input_dir = '/data/steven/project/Object_Detection_coastal/dataser_raw/1_rutouying/images_raw_anno/0913_zhuyisheng_test_imgs'
    img_list = os.listdir(input_dir)
    count = [0,0,0,0,0]
    for i in tqdm(range(len(img_list))):
        img_name = img_list[i]
        # if i>1:
        #     break
        img_path = os.path.join(input_dir,img_name)
        pred = predict(model,img_path)
        print('pred: ', pred)
        if 'ShenChu' in [x[0] for x in pred]:
            count[1] += 1
        if 'FQiZ' in [x[0] for x in pred]:
            count[2] += 1
        if 'FeiBuZhang' in [x[0] for x in pred]:
            count[3] += 1
        if 'ShangZongGeBK' in [x[0] for x in pred]:
            count[4] += 1
        if len(pred) == 0:
            count[0] += 1
    print(count)
