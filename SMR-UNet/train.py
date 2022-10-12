import pandas as pd
import argparse
import os
from collections import OrderedDict
import yaml
from load_LIDC_data import LIDC_IDRI
import torch
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
from unety import UNett_batcnnorm
from torch.utils.data import DataLoader
import SMRUNet

def parse_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--name', default="SMRUNet",
                        help='model name: SMRUNet',choices=['SMRUNet','UNet'])
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 6)')
    parser.add_argument('--early_stopping', default=50, type=int,
                        metavar='N', help='early stopping (default: 50)')
    parser.add_argument('--num_workers', default=8, type=int)
    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-5, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')
    # data
    parser.add_argument('--augmentation',type=str2bool,default=False,choices=[True,False])
    config = parser.parse_args()

    return config

def train(train_loader, model, criterion, optimizer,scheduler):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}
    model.train()
    pbar = tqdm(total=len(train_loader))
    for input, target,_ in train_loader:
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        output = torch.squeeze(output)
        target = torch.squeeze(target)
        loss = criterion(output, target)
        iou = iou_score(output, target)
        dice = dice_coef(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['dice'].update(dice, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('dice',avg_meters['dice'].avg)
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    # scheduler.step()
    # lr_exp = scheduler.get_last_lr()[0]
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice',avg_meters['dice'].avg)])

def validate(val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            output = torch.squeeze(output)
            target = torch.squeeze(target)
            loss = criterion(output, target)
            iou = iou_score(output, target)
            dice = dice_coef(output, target)
            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice',avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice',avg_meters['dice'].avg)])

def main():
    # Get configuration
    config = vars(parse_args())
    # Make Model output directory
    if config['augmentation']== True:
        file_name= config['name'] + '_with_augmentation'
    else:
        file_name = config['name'] +'_base'
    os.makedirs('checpoint/{}'.format(file_name),exist_ok=True)
    print("Creating directory called",file_name)

    print('-' * 20)
    print("Configuration Setting as follow")
    for key in config:
        print('{}: {}'.format(key, config[key]))
    print('-' * 20)

    #save configuration
    with open('checpoint/{}/config.yml'.format(file_name), 'w') as f:
        yaml.dump(config, f)

    #criterion = nn.BCEWithLogitsLoss().cuda()
    criterion = BCEDiceLoss().cuda()
    cudnn.benchmark = True

    # create model
    print("=> creating model" )
    if config['name']=='UNet':
        model = UNett.Unet(1,1)
    elif config['name'] == 'SMRUNet':
        model = SMRUNet.Unet(1,1)
    else:
        raise ValueError("Wrong Parameters")
    model = model.cuda()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'], nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    dataset = LIDC_IDRI(dataset_location='data/')
    train_sampler = torch.load('sampler/train_sampler.pth')
    test_sampler = torch.load('sampler/test_sampler.pth')
    val_sampler = torch.load('sampler/val_sampler.pth')
    train_loader = DataLoader(dataset, batch_size=16, sampler= train_sampler,shuffle=False)
    test_loader = DataLoader(dataset, batch_size=1, sampler=test_sampler,shuffle=False)
    val_loader = DataLoader(dataset, batch_size=16, sampler=val_sampler,shuffle=False)

    log= pd.DataFrame(index=[],columns= ['epoch','lr','loss','iou','dice','val_loss','val_iou'])

    best_dice = 0
    trigger = 0

    for epoch in range(config['epochs']):

        # train for one epoch
        train_log = train(train_loader, model, criterion, optimizer,scheduler=exp_lr_scheduler)
        # evaluate on validation set
        val_log = validate(val_loader, model, criterion)

        print('Training epoch [{}/{}], Training BCE loss:{:.4f}, Training DICE:{:.4f}, Training IOU:{:.4f}, Validation BCE loss:{:.4f}, Validation Dice:{:.4f}, Validation IOU:{:.4f}'.format(
            epoch + 1, config['epochs'], train_log['loss'], train_log['dice'], train_log['iou'], val_log['loss'], val_log['dice'],val_log['iou']))

        tmp = pd.Series([
            epoch,
            config['lr'],
            #train_log['lr_exp'],
            train_log['loss'],
            train_log['iou'],
            train_log['dice'],
            val_log['loss'],
            val_log['iou'],
            val_log['dice']
        ], index=['epoch', 'lr', 'loss', 'iou', 'dice', 'val_loss', 'val_iou','val_dice'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('checpoint/{}/log.csv'.format(file_name), index=False)

        trigger += 1

        if val_log['dice'] > best_dice:
            torch.save(model.state_dict(), 'checpoint/{}/bestmodel_LIDC.pth'.format(file_name))
            best_dice = val_log['dice']
            print("=> saved best model as validation DICE is greater than previous best DICE")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
