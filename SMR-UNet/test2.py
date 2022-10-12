import pandas as pd
import argparse
import os
from collections import OrderedDict
import yaml
from load_LIDC_data import LIDC_IDRI
import torch
import numpy as np
from skimage import io
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import albumentations as albu
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from losses import BCEDiceLoss
from metrics import iou_score,dice_coef
from utils import AverageMeter, str2bool
import UNett_batcnnorm
from torch.utils.data import DataLoader,Dataset
import SMRUNet
class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.image_path = os.listdir(self.root_dir)
        self.label_path = os.listdir(self.label_dir)
    def __getitem__(self, idx):                                #如果想通过item去获取图片，就要先创建图片地址的一个列表
        img_name = self.image_path[idx]
        label_name = self.label_path[idx]
        img_item_path = os.path.join(self.root_dir, img_name)  #每个图片的位置
        label_item_path = os.path.join(self.label_dir, label_name)
        image = np.load(img_item_path)
        label = io.imread(label_item_path)
        return image, label
    def __len__(self):
        return len(self.image_path)

colormap = [[0,0,0], [1,1,1]]
dataset = LIDC_IDRI(dataset_location = 'data/') # change path
test_sampler = torch.load('sampler/test_sampler.pth') # change path
test_loader = DataLoader(dataset, batch_size=1, sampler=test_sampler,shuffle=False)
net = SMRUNet.Unet(1,1)
net.load_state_dict(torch.load('checpoint/SMRUNet/bestmodel_LIDC_no_lr_step2.pth'))
net.cuda()

def testdate(test_loader, model):
    avg_meters = {#'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(test_loader))
        for input, target in test_loader:
            input = input.cuda()
            target = target.cuda()
            input = torch.unsqueeze(input,1)
            output = model(input)
            output = torch.squeeze(output)
            target = torch.squeeze(target)
            #loss = criterion(output, target)
            iou = iou_score(output, target)
            dice = dice_coef(output, target)
            #avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                #('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice',avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()
    return OrderedDict([#('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice',avg_meters['dice'].avg)])

if __name__ == '__main__':
    test_log = testdate(test_loader=test_dataloader, model=net)
    print('testdata Dice:{:.4f}, testdata IOU:{:.4f}'.format(test_log['dice'], test_log['iou']))

