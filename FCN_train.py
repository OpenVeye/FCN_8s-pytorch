import torch
import torch.nn as nn
from PortraitDataset import *
from FCNnet import FCNnet
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm
from torchvision.utils import save_image
from matplotlib import cm
from matplotlib.colors import Normalize
from my_utils import val_score,show_confusion_matrix
from seg_loss_fn import MultiLossFunction
import matplotlib.pyplot as plt


def train_one_epoch(model,loss_fn,optimizer,datloader,logger,epoch,max_epoch):
    model.train()
    device="cuda"if torch.cuda.is_available() else"cpu"
    dataloader_tqdm=tqdm.tqdm(datloader)
    aver_loss = 0.
    mean_iou=0.
    accuracy=0.
    aver_accuracy=0.

    for i, (images_data,masks_data) in enumerate(dataloader_tqdm):
        dataloader_tqdm.set_description(f"training......{epoch+1}/{max_epoch}")
        images_data,masks_data = images_data.to(device),masks_data.to(device)
        predict_masks = model(images_data)
        loss_result = loss_fn(predict_masks,masks_data)
        optimizer.zero_grad()
        loss_result.backward()
        optimizer.step()
        aver_loss = aver_loss+loss_result.item()
        pa,mpa,miou,matrix=val_score(predict_masks,masks_data)
        mean_iou +=miou.item()
        accuracy += pa.item()
        aver_accuracy += mpa.item()
        tqdm_dict = {
            "iter":i+1,
            "loss":loss_result.item(),
            "accuracy":pa.item(),
            "mean_accuracy":mpa.item(),
            "miou":miou.item()
        }
        dataloader_tqdm.set_postfix(tqdm_dict)
        dataloader_tqdm.update()
    len_dataloader = len(datloader)
    aver_loss /=len_dataloader
    mean_iou /= len_dataloader
    accuracy /= len_dataloader
    aver_accuracy /= len_dataloader
    logger.add_scalar("train_loss",aver_loss,global_step=epoch+1)
    logger.add_scalar("train_miou",mean_iou,global_step=epoch+1)
    logger.add_scalar("train_accuracy",accuracy,global_step=epoch+1)
    logger.add_scalar("train_aver_accuracy",aver_accuracy,global_step=epoch+1)
    return aver_loss,mean_iou

def val_one_epoch(model,loss_fn,dataloader,logger,epoch,max_epoch,C=2):

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataloader_tqdm = tqdm.tqdm(dataloader)
    aver_loss = 0.
    mean_iou = 0.
    accuracy = 0.
    aver_accuracy=0.
    confusion_matrix = torch.zeros(1)
    with torch.no_grad():
        for i, (images_data,masks_data) in enumerate(dataloader_tqdm):
            dataloader_tqdm.set_description(f"testing....{epoch+1}/{max_epoch}")
            images_data,masks_data = images_data.to(device),masks_data.to(device)
            predict_masks = model(images_data) #predict_masks size (bs,n_class,h,w)
            loss_result = loss_fn(predict_masks,masks_data)
            aver_loss = aver_loss+loss_result.item()
            pa,mpa,miou,matrix = val_score(predict_masks,masks_data,C=C)
            mean_iou += miou.item()
            accuracy += pa.item()
            aver_accuracy += mpa.item()
            confusion_matrix = confusion_matrix+matrix
            tqdm_dict = {
            "iter":i+1,
            "loss":loss_result.item(),
            "accuracy":pa.item(),
            "mean accuracy":mpa.item(),
            "miou":miou.item()
        }
            dataloader_tqdm.set_postfix(tqdm_dict)

            dataloader_tqdm.update()
            if (i + 1) % 10==0 or i + 1 == len(dataloader):
                predict_masks = torch.softmax(predict_masks, dim=1)
                predict_masks = torch.argmax(predict_masks, dim=1).float().unsqueeze(1)
                save_image(predict_masks, os.path.join("val_masks", f"{i}.png"))

    aver_loss /= max(1,len(dataloader))
    mean_iou /= max(1,len(dataloader))
    accuracy /= max(1,len(dataloader))
    aver_accuracy /= max(1,len(dataloader))
    logger.add_scalar("val loss",aver_loss,global_step=epoch+1)
    logger.add_scalar("val miou",mean_iou,global_step=epoch+1)
    logger.add_scalar("val accuracy",accuracy,global_step=epoch+1)
    logger.add_scalar("val aver_accuracy",aver_accuracy,global_step=epoch+1)
    #计算混淆矩阵归一化
    confusion_matrix=confusion_matrix/confusion_matrix.sum()
    return  aver_loss,mean_iou,confusion_matrix.numpy()

def main():
    #新建日志、模型权重文件存放文件夹
    if not os.path.exists("log"):
        os.mkdir("log")
    if not os.path.exists("model"):
        os.mkdir("model")
    if not os.path.exists("val_masks"):
        os.mkdir("val_masks")
    #构建训练验证数据集
    image_size=(320,320)
    batch_size=4
    n_workers = 0
    train_data_path = r"data\Portrait-dataset-2000\dataset\training"
    train_images_name,train_masks_name=get_train_val_data(train_data_path)
    train_dataset = PortraitDataset(train_images_name,train_masks_name,get_transform(True,image_size=image_size))
    train_dataloader=DataLoader(train_dataset,batch_size,True,num_workers=n_workers,pin_memory=True)
    val_data_path = r"data\Portrait-dataset-2000\dataset\testing"
    val_images_name,val_masks_name = get_train_val_data(val_data_path)
    val_dataset = PortraitDataset(val_images_name,val_masks_name,get_transform(False,image_size=image_size))
    val_dataloader = DataLoader(val_dataset,batch_size,False,num_workers=n_workers,pin_memory=True)
    #建立网络
    device = "cuda"if torch.cuda.is_available() else "cpu"
    net = FCNnet(2,True)
    is_pretrained = False
    if is_pretrained:
        weight_file = r"model\best_fcn.pth"
        weight_dict=torch.load(weight_file)
        net_dict = net.state_dict()
        temp_dict = {k:v for k,v in weight_dict.items() if k in net_dict.keys() and net_dict[k].numel()==v.numel()}
        for k,v in temp_dict:
            net_dict[k]=v
        net.load_state_dict(net_dict)
        print(f"权重文件存在，模型成功导入参数")
    net = net.to(device)
    #优化器部分
    lr=0.0001
    optimizer = torch.optim.Adam(net.parameters(),lr=lr,weight_decay=5.e-4,betas=(0.9,0.999))
    #损失函数
    loss_function = MultiLossFunction(type="CE")
    #定义logger,记录网络训练loss、pa、mpa、miou
    logger = SummaryWriter("log")
    max_epoch=2
    train_info = []
    val_info=[]
    best_loss =np.inf
    best_miou=0
    best_matrix = None
    #定义类别名称
    class_name=["background","person"]
    for epoch in range(max_epoch):
        if epoch==40:
            lr=0.00001
        elif epoch==60:
            lr = 0.000001
        for param in optimizer.param_groups:
            param["lr"]=lr
        aver_loss,miou=train_one_epoch(net,loss_function,optimizer,train_dataloader,logger,epoch,max_epoch)
        train_info.append([aver_loss,miou])
        aver_loss,miou,c_matrix=val_one_epoch(net,loss_function,val_dataloader,logger,epoch,max_epoch,C=2)
        val_info.append([aver_loss,miou])
        if aver_loss<best_loss or miou>best_miou:
            if aver_loss<best_loss:
                best_loss=aver_loss
            if miou>best_miou:
                best_miou=miou
                best_matrix=c_matrix
                torch.save(net.state_dict(),r"model\best_fcn.pth",_use_new_zipfile_serialization=False)
    logger.close()
    #loss,miou可视化保存为图片
    fig,axes = plt.subplots(1,2)
    axes[0].plot(list(range(max_epoch)),[n_row[0] for n_row in train_info],label="train")
    axes[0].plot(list(range(max_epoch)),[n_row[0] for n_row in val_info],label="val")
    axes[0].legend()
    axes[0].set_title("aver_loss")
    axes[1].plot(list(range(max_epoch)), [n_row[1] for n_row in train_info], label="train")
    axes[1].plot(list(range(max_epoch)), [n_row[1] for n_row in val_info], label="val")
    axes[1].legend()
    axes[1].set_title("miou")
    fig.savefig(r"val_masks\train_val_plot.png")
    plt.show()
    # 混淆矩阵可视化并保存为图片
    if best_matrix is not None:
        show_confusion_matrix(best_matrix, class_name)

if __name__=="__main__":
    main()


