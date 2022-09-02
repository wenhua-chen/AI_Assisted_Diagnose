# -*- coding:utf-8 -*- 
# Author: 陈文华(Steven)
# Website: https://wenhua-chen.github.io/
# Github: https://github.com/wenhua-chen
# Date: 2020-03-16 05:08:28
# LastEditTime: 2022-09-02 15:43:21
# Description: 判定图片是否清晰合格

from __future__ import print_function, division
import timm
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import shutil
import time
import os
from tqdm import tqdm

# 初始化
def init_model(model_path):
    model = timm.create_model('mobilenetv3_100',pretrained=True,num_classes=3,in_chans=3)
    device = torch.device("cuda")
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# 判定图片是否清晰合格: 0:不确定, 1:合格, 2:不合格
def predict_ZC(model,imgpath):
    device = torch.device("cuda")
    cls_list=[0,1,2]
    img = Image.open(imgpath).convert('RGB')
    sizex,sizey = 224,224
    transform = transforms.Compose([
        transforms.Resize([sizex,sizey]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    img_tensor = transform(img)
    img_tensor.unsqueeze_(0)
    img_tensor.shape
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        y_predict = model(img_tensor)
    print(y_predict.data.cpu().numpy().tolist())
    out = torch.max(y_predict,1)[1]
    out0 = out.data.cpu().numpy().tolist()
    print('out0:',out0)
    rescults = cls_list[out0[0]]
    return rescults

if __name__ == "__main__":

    # model_path='/data/steven/project/Object_Detection_coastal/dr_wrapper/DR_models_configs/dr0_0_Clear_formal.pth'
    # model_path='/data/steven/project/Object_Detection_coastal/Classfication/output/1_img_qualification/T2_qualification_MNetV3_train_stage2_inchannel3_2edi/models/T2_qualification_MNetV3_train_stage2_inchannel3_2edi_mobilenet_v3_E1_F1_0.850914.pth'
    # model_path='/data/steven/project/Object_Detection_coastal/Classfication/output/1_img_qualification/T2_qualification_MNetV3_train_stage2_inchannel3_2edi/models/T2_qualification_MNetV3_train_stage2_inchannel3_2edi_mobilenet_v3_E25_F1_0.855359.pth'
    # model_path='/data/steven/project/Object_Detection_coastal/Classfication/output/1_img_qualification/T3_qualification_MNetV3_train_stage2_inchannel3_3edi/models/T3_qualification_MNetV3_train_stage2_inchannel3_3edi_mobilenet_v3_E39_F1_0.873612.pth'
    model_path='/data/steven/project/Object_Detection_coastal/dr_wrapper/DR_models_configs/dr0_0_Clear_formal.pth'

    # 模型初始化
    print('loading model....')
    timea=time.time()
    model = init_model(model_path)
    timeb=time.time()
    print('model loaded, time: ',timeb-timea,' s')

    # infer
    single_test=False
    if single_test:
        # img_path = '1.3.12.2.1107.5.3.33.4700.11.201905270848080125-1.jpg'
        img_path = '/data/steven/project/Object_Detection_coastal/dr_wrapper/1.3.12.2.1107.5.3.33.4700.11.201901020810040562-1.jpg'
        pred = predict_ZC(model,img_path)
        print('pred',pred)
    else:
        input_dir = '/data/steven/project/Object_Detection_coastal/dr_wrapper/test_data/5_LRG_drclient_failed/temp2/'
        img_list = os.listdir(input_dir)
        for i in tqdm(range(len(img_list))):
            img_name = img_list[i]
            # if i>1:
            #     break
            img_path = os.path.join(input_dir,img_name)
            pred = predict_ZC(model,img_path)
            print(img_name,' pred: ', pred)
