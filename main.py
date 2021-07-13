import numpy as np
import torch
import torchvision

from torch.utils.tensorboard import SummaryWriter

import random

import os

from PIL import Image

from models import ResUNet
from losses import flatMSE

import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms

import torchvision.transforms.functional as TF

import copy

writer = SummaryWriter()

data_path = 'C:/aerialimagelabeling/AerialImageDataset'

NUM_EPOCHS = 10
BATCH_SIZE = 20

train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []

DEBUG = False

class AerialImageDataSet(Dataset):
    def __init__(self, path):
        self.im_path = '/'.join([path, 'images/'])
        self.gt_path = '/'.join([path, 'gt/'])
        self.im_cache_path = '/'.join([path, 'images_cache/'])
        self.gt_cache_path = '/'.join([path, 'gt_cache/'])
        self.num_ims = len(os.listdir(self.im_path))
        self.samples_per_img = 350

    def __len__(self):
        return self.samples_per_img*self.num_ims
    
    def __getitem__(self, idx):
        #check if cached version exists
        proposed_im_path = self.im_cache_path + str(idx) + '.pt'
        proposed_gt_path = self.gt_cache_path + str(idx) + '.pt'
        im_exists = os.path.isfile(proposed_im_path)
        gt_exists = os.path.isfile(proposed_gt_path)
        if im_exists and gt_exists:
            x = torch.load(proposed_im_path)
            y = torch.load(proposed_gt_path)
            return x, y
        img_idx = idx // self.samples_per_img #get which 5000x5000 image should be used
        remainder = idx - img_idx * self.num_ims #index within 5000x5000 tile
        random.seed(remainder)
        aspect_ratio = random.uniform(0.9, 1.1)
        height = random.randint(100, 500)
        width = int(height * aspect_ratio)
        top = random.randint(1, 4999-height)
        left = random.randint(1, 4999-width)
        h_flip = random.choice((False, True))
        v_flip = random.choice((False, True))


        
        x_name = os.listdir(self.im_path)[img_idx]
        x = Image.open('/'.join([self.im_path, x_name]))
        y = Image.open('/'.join([self.gt_path, x_name]))

        x = TF.resized_crop(x, top, left, height, width, (224, 224), Image.NEAREST)
        y = TF.resized_crop(y, top, left, height, width, (224, 224), Image.NEAREST)
        if h_flip:
            x = TF.hflip(x)
            y = TF.hflip(y)
        if v_flip:
            x = TF.vflip(x)
            y = TF.vflip(y)
        x = TF.to_tensor(x)
        y = TF.to_tensor(y)
        y = y.type(torch.int8)
        x = TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        torch.save(x, proposed_im_path)
        torch.save(y, proposed_gt_path)

        return x, y
def accuracy(output, target):
    with torch.no_grad():
        best_guess = torch.round(output)
        flat_out = torch.flatten(best_guess)
        #print(flat_out)
        flat_targ = torch.flatten(target)
        #print(flat_targ)
        abs_diff = torch.abs(torch.sub(flat_out, flat_targ))
        total_pixels = abs_diff.shape[0]
        misses = torch.sum(abs_diff)
        acc = (total_pixels-misses)/(total_pixels)
        return acc.item()

def train(epoch, data_loader, model, optimizer, criterion):
    accuracyNumerator = 0
    accuracyDenominator = 0
    lossNumerator = 0
    lossDenominator = 0
    for idx, (data, target) in enumerate(data_loader):
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        out = model.forward(data)
        num_entries = out.shape[0]
        loss = criterion.forward(out, target)
        sum_loss = out.shape[0] * loss
        lossNumerator += sum_loss.item()
        lossDenominator += num_entries
        model.zero_grad()
        loss.backward()
        optimizer.step()
        batch_acc = accuracy(out, target)
        print('Batch ' + str(idx) + ' accuracy: ' + str(batch_acc))
        print('Batch ' + str(idx) + ' loss: ' + str(loss))
        num_pos = out.shape[0] * batch_acc
        accuracyNumerator += num_pos
        accuracyDenominator += num_entries
        writer.add_scalar('Loss/batch', loss.item(), idx)
        writer.add_scalar('Acc/batch', batch_acc, idx)
    train_loss_history.append(lossNumerator/lossDenominator)
    train_acc_history.append(accuracyNumerator/accuracyDenominator)
    print('Training Epoch #' + str(epoch)+' - Loss: ' + str(train_loss_history[-1]) + '; Acc: ' + str(train_acc_history[-1]))
    writer.add_scalar('Loss/train', train_loss_history[-1], epoch)
    writer.add_scalar('Acc/train', train_acc_history[-1], epoch)

def writeImage(out, targ, epoch):
    with torch.no_grad():
        roundedOut = torch.round(out)
        gridOut = torchvision.utils.make_grid(roundedOut)
        gridTarg = torchvision.utils.make_grid(targ)
        writer.add_image('val_targ', gridTarg, epoch)
        writer.add_image('val_output', gridOut, epoch)

def validate(epoch, val_loader, model, criterion):
    accuracyNumerator = 0
    accuracyDenominator = 0
    lossNumerator = 0
    lossDenominator = 0
    for idx, (data, target) in enumerate(val_loader):
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        with torch.no_grad():
            out = model.forward(data)
            loss = criterion.forward(out, target)
            lossNumerator += out.shape[0] * loss
            lossDenominator += out.shape[0]
            acc = accuracy(out, target)
            accuracyNumerator += out.shape[0] * acc
            accuracyDenominator += out.shape[0]
            if idx == 0:
                writeImage(out, target, epoch)
    val_loss_history.append(lossNumerator/lossDenominator)
    val_acc_history.append(accuracyNumerator/accuracyDenominator)
    print('Validation Epoch #' + str(epoch)+' - Loss: ' + str(val_loss_history[-1]) + '; Acc: ' + str(val_acc_history[-1]))
    writer.add_scalar('Loss/val', val_loss_history[-1], epoch)
    writer.add_scalar('Acc/val', val_acc_history[-1], epoch)
    return val_acc_history[-1]





def main():
    #print('poopy loopy')
    #transforms_list = [transforms.RandomResizedCrop(224, (0.05, 0.15), (0.75, 1.25), Image.NEAREST),
    #                   transforms.RandomVerticalFlip(),
    #                   transforms.RandomVerticalFlip(),
    #                   transforms.ToTensor()]

    train_and_val_dataset = AerialImageDataSet('/'.join([data_path, 'train']))
    print('train dataset created')
    train_size = int(0.8*(len(train_and_val_dataset)))
    val_size = len(train_and_val_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_and_val_dataset, [train_size, val_size])
    
    print('train dataset contains: ' + str(len(train_dataset)) + ' images')
    print('val dataset contains: ' + str(len(val_dataset)) + ' images')
    #test_dataset = AerialImageDataSet('/'.join([data_path, 'test']))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    print('train loader created')
    if DEBUG:
        for idx, (x, y) in enumerate(train_loader):
            if idx % 1000 == 0:
                print('train iteration: ' + str(idx))
            pass
        print('train cache complete')
        for idx, (x, y) in enumerate(val_loader):
            if idx % 1000 == 0:
                print('val iteration: ' + str(idx))
            pass
        print('val cache complete')
    model = ResUNet()
    if torch.cuda.is_available():
        model = model.cuda()
    criterion = flatMSE()
    optimizer = torch.optim.Adam(model.parameters())
    bestAcc = 0
    best_model = None
    for epoch in range(NUM_EPOCHS):
        train(epoch, train_loader, model, optimizer, criterion)

        accuracy = validate(epoch, val_loader, model, criterion)
        if accuracy > bestAcc:
            bestAcc = accuracy
            best_model = copy.deepcopy(model)
        print('Epoch num: ' + str(epoch)+'; Accuracy: ' + str(accuracy))
    torch.save(best_model.state_dict(), './checkpoints/project.pth')
    writer.close()




if __name__ == '__main__':
    main()