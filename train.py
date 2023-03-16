import os
import time

import numpy as np
import pandas as pd
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm, trange
from util.parameters import argparse

from model.ResNet18 import resnet18


args = argparse()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_accuracy_and_loss(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    cross_entropy = 0.
    for i, (features, targets) in enumerate(data_loader):
            
        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        cross_entropy += F.cross_entropy(logits, targets).item()
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100, cross_entropy/num_examples


def train(train_loader, test_loader):
    model = resnet18(num_classes = 2)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Load existing current model
    start_epoch = 0
    model_path = "./results/model/gender_model.pth"
    if os.path.exists("./results/model/gender_model.pth"):
        model.load_state_dict(torch.load(model_path)['model'])
        optimizer.load_state_dict(torch.load(model_path)['optimizer'])
        start_epoch = torch.load(model_path)['epoch']

    train_acc_lst, valid_acc_lst = [], []
    train_loss_lst, valid_loss_lst = [], []

    for epoch in range(start_epoch+1, args.epoch):
        
        model.train()
        for batch_idx, (features, targets) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch: {epoch+1:03d}/{args.epoch:03d}", ncols=100):
        
            ### PREPARE MINIBATCH
            features = features.to(device)
            targets = targets.to(device)
                
            ### FORWARD AND BACK PROP
            logits, probas = model(features)
            cost = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            
            cost.backward()
            
            ### UPDATE MODEL PARAMETERS
            optimizer.step()
            
        # no need to build the computation graph for backprop when computing accuracy
        model.eval()
        with torch.set_grad_enabled(False):
            # train_acc, train_loss = compute_accuracy_and_loss(model, train_loader, device=device)
            valid_acc, valid_loss = compute_accuracy_and_loss(model, test_loader, device=device)
            # train_acc_lst.append(train_acc)
            valid_acc_lst.append(valid_acc)
            # train_loss_lst.append(train_loss)
            valid_loss_lst.append(valid_loss)
            print(f'Epoch: {epoch+1:03d}/{args.epoch:03d} '
                f' | Validation Acc.: {valid_acc:.2f}%')
        
        # Save model
        state_model = {'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'epoch': epoch}
        save(state_model)

            
def save(state_model):
    save_dir = os.path.join('./results/model')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(state_model, os.path.join(save_dir, f'gender_model.pth'))

def load(self):
    save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

    self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + f'_G.pth'))['model'])
    self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + f'_D.pth'))['model'])
    self.G_optimizer.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + f'_G.pth'))['optimizer'])
    self.D_optimizer.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + f'_D.pth'))['optimizer'])
    self.start_epoch = torch.load(os.path.join(save_dir, self.model_name + f'_D.pth'))['epoch']


if __name__ == '__main__':
    df = pd.read_csv('../data/celeba/list_attr_celeba.csv')

    img_id = df['image_id'].values
    gen = df['Male'].values

    if not os.path.exists('./data/celeba/train/Female'):
        os.makedirs('./data/celeba/train/Female')
        os.makedirs('./data/celeba/train/Male')
        os.makedirs('./data/celeba/test/Female')
        os.makedirs('./data/celeba/test/Male')

        test_len = 40520
        train_len = 162079

        for idx in range(train_len):
            if (gen[idx] == -1):
                shutil.copyfile('../data/celeba/img_align_celeba/'+img_id[idx], './data/celeba/train/Female/'+img_id[idx])
            elif (gen[idx] == 1):
                shutil.copyfile('../data/celeba/img_align_celeba/'+img_id[idx], './data/celeba/train/Male/'+img_id[idx])

        for idx in range(train_len+1, len(img_id)):
            if (gen[idx] == -1):
                shutil.copyfile('../data/celeba/img_align_celeba/'+img_id[idx], './data/celeba/test/Female/'+img_id[idx])
            elif (gen[idx] == 1):
                shutil.copyfile('../data/celeba/img_align_celeba/'+img_id[idx], './data/celeba/test/Male/'+img_id[idx])

    transform = transforms.Compose([transforms.CenterCrop((178, 178)),
                                       transforms.Resize((128, 128)),
                                       #transforms.Grayscale(),                                       
                                       #transforms.Lambda(lambda x: x/255.),
                                       transforms.ToTensor()])
    
    train_dataset = datasets.ImageFolder('./data/celeba/train', transform=transform)
    test_dataset = datasets.ImageFolder('./data/celeba/test', transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    train(train_loader, test_loader)
