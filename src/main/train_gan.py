#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************
# @Author  : Qiqi Xiao & Jiaxu Zou
# @Email     : xiaoqiqi177@gmail.com & zoujx96@gmail.com
# @File    : train_gan_ex.py
# **************************************
#
# Modified by Duy Le
#***************************************

import sys
import os
import numpy as np
import random
import copy
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
import albumentations as A

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler

import config_gan as config
from .archs.segformerstar import SegformerStar
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
import argparse
from .util.base_utils import get_datapath
from ..data.lesion_dataset import OneLesionSegmentation
import archs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dir_checkpoint = config.MODELS_DIR
lesions = [config.LESION_NAME]
rotation_angle = config.ROTATION_ANGEL
image_size = config.IMAGE_SIZE
image_dir = config.IMAGE_DIR
batchsize = config.TRAIN_BATCH_SIZE

softmax = nn.Softmax(1)

class DNet(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_size=32):
        super(DNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)

        return x

def eval_model(model, eval_loader):
    model.eval()
    masks_soft = []
    masks_hard = []

    with torch.set_grad_enabled(False):
        for inputs, true_masks in eval_loader:
            inputs = inputs.to(device=device, dtype=torch.float)
            true_masks = true_masks.to(device=device, dtype=torch.float)
            bs, _, h, w = inputs.shape
            # not ignore the last few patches
            h_size = (h - 1) // image_size + 1
            w_size = (w - 1) // image_size + 1
            masks_pred = torch.zeros(true_masks.shape).to(dtype=torch.float)

            for i in range(h_size):
                for j in range(w_size):
                    h_max = min(h, (i + 1) * image_size)
                    w_max = min(w, (j + 1) * image_size)
                    inputs_part = inputs[:,:, i*image_size:h_max, j*image_size:w_max]
                    masks_pred_single = model(inputs_part)[-1]
                    masks_pred[:, :, i*image_size:h_max, j*image_size:w_max] = masks_pred_single

            masks_pred_batch = masks_pred.cpu().numpy()
            masks_soft_batch = masks_pred_batch
            masks_hard_batch = true_masks.cpu().numpy()

            masks_soft.extend(masks_soft_batch)
            masks_hard.extend(masks_hard_batch)

    masks_soft = np.array(masks_soft).transpose((1, 0, 2, 3))
    masks_hard = np.array(masks_hard).transpose((1, 0, 2, 3))
    masks_soft = np.reshape(masks_soft, (masks_soft.shape[0], -1))
    masks_hard = np.reshape(masks_hard, (masks_hard.shape[0], -1))

    ap = average_precision_score(masks_hard[0], masks_soft[0])
    return ap

def denormalize(inputs):
    mean = torch.FloatTensor([0.485, 0.456, 0.406]).to(device)
    std = torch.FloatTensor([0.229, 0.224, 0.225]).to(device)
    return ((inputs * std[None, :, None, None] + mean[None, :, None, None])*255.).to(device=device, dtype=torch.uint8)

def generate_log_images(inputs_t, true_masks_t, masks_pred_softmax_t):
    true_masks = (true_masks_t * 255.).to(device=device, dtype=torch.uint8)
    masks_pred_softmax = (masks_pred_softmax_t.detach() * 255.).to(device=device, dtype=torch.uint8)
    inputs = denormalize(inputs_t)
    bs, _, h, w = inputs.shape
    pad_size = 5
    images_batch = (torch.ones((bs, 3, h, w*3+pad_size*2)) * 255.).to(device=device, dtype=torch.uint8)
    
    images_batch[:, :, :, :w] = inputs
    
    images_batch[:, :, :, w+pad_size:w*2+pad_size] = 0
    images_batch[:, 0, :, w+pad_size:w*2+pad_size] = true_masks[:, 1, :, :]
    
    images_batch[:, :, :, w*2+pad_size*2:] = 0
    images_batch[:, 0, :, w*2+pad_size*2:] = masks_pred_softmax[:, 1, :, :]
    return images_batch

def image_to_patch(image, patch_size):
    bs, channel, h, w = image.shape
    return (image.reshape((bs, channel, h//patch_size, patch_size, w//patch_size, patch_size))
            .permute(2, 4, 0, 1, 3, 5)
            .reshape((-1, channel, patch_size, patch_size)))

def train_model(model, dnet, gan_exist, train_loader, eval_loader, criterion, g_optimizer, g_scheduler, d_optimizer, \
    d_scheduler, batch_size, num_epochs=5, start_epoch=0, start_step=0):
    model.to(device=device)
    dnet.to(device=device)
    tot_step_count = start_step

    best_ap = 0.

    for epoch in range(start_epoch, start_epoch + num_epochs):
        print('Starting epoch {}/{}.\t\n'.format(epoch + 1, start_epoch+num_epochs))
        g_scheduler.step()
        d_scheduler.step()
        model.train()
        dnet.train()

        for batch in train_loader:
            inputs = batch["image"]
            true_masks = batch["mask"]
            inputs = inputs.to(device=device, dtype=torch.float)
            true_masks = true_masks.to(device=device, dtype=torch.float)

            masks_pred = model(inputs)
            loss_ce = criterion(masks_pred, true_masks)
            
            # Save images
            ce_weight = 1.
            g_loss = loss_ce * ce_weight

            # add descriminator loss
            if config.D_MULTIPLY:
                input_real = torch.matmul(inputs, true_masks[:, 1:, :, :])
                input_real = image_to_patch(input_real, config.PATCH_SIZE)
                input_fake = torch.matmul(inputs, masks_pred)
                input_fake = image_to_patch(input_fake, config.PATCH_SIZE)
            else:
                input_real = torch.cat((inputs, true_masks[:, 1:, :, :]), 1)
                input_real = image_to_patch(input_real, config.PATCH_SIZE)
                input_fake = torch.cat((inputs, masks_pred), 1)
                input_fake = image_to_patch(input_fake, config.PATCH_SIZE)
            
            d_real = dnet(input_real)
            d_fake = dnet(input_fake.detach()) #do not backward to generator
            d_real_loss = torch.mean(1 - d_real)
            d_fake_loss = torch.mean(d_fake)
            
            #update d loss
            loss_d = d_real_loss + d_fake_loss
            d_optimizer.zero_grad()
            loss_d.backward()
            d_optimizer.step()
            
            #updage g loss
            d_fake = dnet(input_fake) #do backward to generator
            loss_gan = torch.mean(1 - d_fake)
            g_loss += loss_gan * gan_weight
            
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            
            tot_step_count += 1
        
        if not os.path.exists(dir_checkpoint):
            os.mkdir(dir_checkpoint)

        if (epoch + 1) % 40 == 0:
            eval_ap = eval_model(model, eval_loader)
            with open("ap_during_learning_ex_" + gan_exist + ".txt", 'a') as f:
                f.write("epoch: " + str(epoch))
                f.write("ap: " + str(eval_ap))
                f.write("\n")

            if eval_ap > best_ap:
                best_ap = eval_ap
                if dnet:
                    state = {
                        'epoch': epoch,
                        'step': tot_step_count,
                        'g_state_dict': model.state_dict(),
                        'd_state_dict': dnet.state_dict(),
                        'g_optimizer': g_optimizer.state_dict(),
                        'd_optimizer': d_optimizer.state_dict(),
                        }
                else:
                    state = {
                        'epoch': epoch,
                        'step': tot_step_count,
                        'state_dict': model.state_dict(),
                        'optimizer': g_optimizer.state_dict()
                        }

                torch.save(state, \
                            os.path.join(dir_checkpoint, 'model_' + gan_exist + '.pth.tar'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--gan', type=str, default="True")
    args = parser.parse_args()
    #Set random seed for Pytorch and Numpy for reproducibility
    args.seed = 1999
    np.random.seed(1999)
    random.seed(1999)
    torch.manual_seed(1999)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1999)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.gan == "True":
        gan_weight = config.D_WEIGHT
    else:
        gan_weight = 0.

    model = SegformerStar(backbone="mit_b0", deep_supervision=True, clfhead=True, pretrained=False)
   
    if config.D_MULTIPLY:
        dnet = DNet(input_dim=3, output_dim=1, input_size=config.PATCH_SIZE)
    else:
        dnet = DNet(input_dim=4, output_dim=1, input_size=config.PATCH_SIZE)

    g_optimizer = optim.AdamW(model.parameters(),
                              lr=config.G_LEARNING_RATE,
                              momentum=0.9,
                              weight_decay=0.0005)
    d_optimizer = optim.AdamW(dnet.parameters(),
                              lr=config.D_LEARNING_RATE,
                              momentum=0.9,
                              weight_decay=0.0005)

    resume = config.RESUME_MODEL
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']+1
            start_step = checkpoint['step']
            try:
                model.load_state_dict(checkpoint['state_dict'])
                g_optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                model.load_state_dict(checkpoint['g_state_dict'])
                dnet.load_state_dict(checkpoint['d_state_dict'])
            print('Model loaded from {}'.format(resume))
        else:
            print("=> no checkpoint found at '{}'".format(resume))
    else:
        start_epoch = 0
        start_step = 0


    mask_path = os.path.join(config.IMG_DIR, "2. All Segmentation Groundtruths", "a. Training Set")
    if config.PREPROCESS:
        img_path = os.path.join(config.IMG_DIR, "Images_CLAHE", "a. Training Set")
    else:
        img_path = os.path.join(config.IMAGE_DIR, "1. Original Images", "a. Training Set")
    image_paths, mask_paths = get_datapath(img_path=img_path, mask_path=mask_path, lesion_type="EX")

    indices = np.arange(len(image_paths))
    train_indices, valid_indices = train_test_split(
        indices, test_size=0.2, random_state=1999, shuffle=True)
    np_images = np.array(image_paths)
    train_imgs = np_images[train_indices].tolist()
    valid_imgs = np_images[valid_indices].tolist()

    np_masks = np.array(mask_paths)
    train_masks = np_masks[train_indices].tolist()
    valid_masks = np_masks[valid_indices].tolist()
    preprocess_fn, mean, std = archs.get_preprocessing_fn("IDRiD", False)
    train_transform = A.Compose([
        A.RandomRotate90(),
        A.RandomCrop(512, 512),
        A.Normalize(mean=mean, std=std)
    ])

    train_dataset = OneLesionSegmentation(train_imgs, False, train_masks, transform=train_transform)
    eval_dataset = OneLesionSegmentation(valid_imgs, False, valid_masks, transform=A.Compose([A.Normalize(mean=mean, std=std)]))

    train_loader = DataLoader(train_dataset, batchsize, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batchsize, shuffle=False)

    g_scheduler = lr_scheduler.StepLR(g_optimizer, step_size=10, gamma=0.9)
    d_scheduler = lr_scheduler.StepLR(d_optimizer, step_size=5, gamma=0.9)
    criterion = nn.BCEWithLogitsLoss(weight=torch.FloatTensor(config.CROSSENTROPY_WEIGHTS).to(device))
    
    train_model(model, dnet, args.gan, train_loader, eval_loader, criterion, g_optimizer, g_scheduler, \
        d_optimizer, d_scheduler, batchsize, num_epochs=config.EPOCHES, start_epoch=start_epoch, start_step=start_step)