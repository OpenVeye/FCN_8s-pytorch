import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import os

from torchvision.transforms import transforms
from my_utils import get_transform,images_show

os.chdir(r"F:\mygit\FCN_8s") #程序目录

class PortraitDataset(Dataset):
    def __init__(self,images_path_list,masks_path_list,transform=None):
        super().__init__()
        self.images_path_list = images_path_list
        self.masks_path_list = masks_path_list
        self.transform = transform
        self.t = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
            ]
        )
    def __len__(self):
        return len(self.images_path_list)
    def __getitem__(self, item):
        image_name = self.images_path_list[item]
        mask_name = self.masks_path_list[item]
        image_np = cv2.imread(image_name)
        mask_np = cv2.imread(mask_name,0)
        if self.transform is not None:
            aug_dict = {"image":image_np,"mask":mask_np}
            albument=self.transform(**aug_dict)
            image_np = albument["image"]
            mask_np = albument["mask"]
        mask_np[mask_np>0]=1
        image = self.t(image_np)
        mask = torch.from_numpy(mask_np).long()

        return image,mask



#获取训练集或者验证集原图像和对应掩膜图像的名称
def get_train_val_data(data_dir):
    #获取文件夹下的.png图像名称
    all_images_name = glob.glob(os.path.join(data_dir,"*.png"))
    #获取原图像名称
    images_name_list = [img_name for img_name in all_images_name if "matte" not in os.path.basename(img_name).split(".")[0].split("_")]
    #获取掩模图名称
    masks_name_list = [img_name.replace(".png","_matte.png") for img_name in images_name_list]
    #乱序
    np.random.seed(42)
    len_images = len(images_name_list)
    random_id = np.random.permutation(len_images)
    images_name_list=np.array(images_name_list)[random_id].tolist()
    masks_name_list=np.array(masks_name_list)[random_id].tolist()
    return images_name_list,masks_name_list


if __name__ =="__main__":
    data_dir = r"data\Portrait-dataset-2000\dataset\testing"    #测试集目录
    images_name_list,masks_name_list=get_train_val_data(data_dir)
    image,mask = next(iter(PortraitDataset(images_name_list,masks_name_list,get_transform())))
    images_show(image.unsqueeze(0),mask.unsqueeze(0))

