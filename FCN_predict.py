import os.path
from my_utils import save_roi_images,vedio_concatenate_show
import numpy as np
from torchvision.transforms import transforms
import cv2
import torch.cuda
from PIL import Image
from FCNnet import FCNnet
from PortraitDataset import *

def predict4images(images_dir,image_size=(320,256),C=2,weight_file=r"model\best_fcn.pth"):
    device="cuda"if torch.cuda.is_available() else "cpu"
    #构建网络
    net =FCNnet(C).to(device)
    #导入FCN网络权重文件
    is_pretrained=True
    if is_pretrained:
        if os.path.exists(weight_file):
            weight_dict = torch.load(weight_file)
            net_dict = net.state_dict()
            temp_dict = {k: v for k, v in weight_dict.items() if
                         k in net_dict.keys() and net_dict[k].numel() == v.numel()}
            for k, v in temp_dict.items():
                net_dict[k] = v
            net.load_state_dict(net_dict)
            print(f"权重文件存在，模型成功导入参数")
    net.eval()
    #新建抠图前景图片保存文件夹
    if not os.path.exists("predict"):
        os.mkdir("predict")
    #将图片集转换成张量
    images_name_list = os.listdir(images_dir)
    images_name_list = [os.path.join(images_dir,img_name) for img_name in images_name_list]
    #这里对验证集进行图片抠图
    images_name_list = [img_name for img_name in images_name_list if "matte.png" not in img_name.split("\\")[-1].split("_")]
    save_roi_images(net,images_name_list)

def predict4video(image_size=(320,256),video_name=r"data\Testing.mp4",weight_file=r"model\best_fcn.pth",C=2):
    device="cuda"if torch.cuda.is_available() else "cpu"
    # 构建网络
    net = FCNnet(C).to(device)
    # 导入FCN网络权重文件
    is_pretrained = True
    if is_pretrained:
        if os.path.exists(weight_file):
            weight_dict = torch.load(weight_file)
            net_dict = net.state_dict()
            temp_dict = {k: v for k, v in weight_dict.items() if
                         k in net_dict.keys() and net_dict[k].numel() == v.numel()}
            for k, v in temp_dict.items():
                net_dict[k] = v
            net.load_state_dict(net_dict)
            print(f"权重文件存在，模型成功导入参数")
    net.eval()
    vedio_concatenate_show(net,video_name,is_roi_or_mask=False)
if __name__ == "__main__":
    images_dir = r"data\Portrait-dataset-2000\dataset\testing" #测试图片目录
    predict4images(images_dir)
    #predict4video()