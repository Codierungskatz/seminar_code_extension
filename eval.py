import torch
torch.set_printoptions(10)

import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import numpy as np
import torch.optim as optim
import os
import shutil
from matplotlib import pyplot as plt

from models.squid import AE#, QueueMemory
from models.inpaint import InpaintBlock
from models.discriminator import SimpleDiscriminator

import random
import argparse

import importlib

from tools import parse_args, build_disc, log, log_loss, save_image
from alert import GanAlert
import re



args = parse_args()

if not os.path.exists(os.path.join('checkpoints', args.exp)):
    print('exp folder cannot be found!')
    exit()

if not os.path.isfile(os.path.join('checkpoints', args.exp, 'discriminator.pth')):
    print('discriminator ckpt cannot be found!')
    exit()

if not os.path.isfile(os.path.join('checkpoints', args.exp, 'config.py')):
    print('config file cannot be found!')
    exit()

# load config file from exp folder
CONFIG = importlib.import_module('checkpoints.'+args.exp+'.config').Config()

save_path = os.path.join('checkpoints', args.exp, 'test_images')

# log
log_file = open(os.path.join('checkpoints', args.exp, 'eval_log.txt'), 'w')

# build main model from exp folder
MODULE = importlib.import_module('checkpoints.'+args.exp+'.squid')
model = MODULE.AE(1, 32, CONFIG.shrink_thres, num_slots=CONFIG.num_slots, num_patch=CONFIG.num_patch, level=CONFIG.level, 
            ratio=CONFIG.mask_ratio, initial_combine=CONFIG.initial_combine, drop=CONFIG.drop,
            dist=CONFIG.dist, memory_channel=CONFIG.memory_channel, mem_num_slots=CONFIG.mem_num_slots,
            ops=CONFIG.ops, decoder_memory=CONFIG.decoder_memory).cuda()

print('Loading AE...')
ckpt = torch.load(os.path.join('checkpoints',args.exp,'model.pth'))
model.load_state_dict(ckpt)
print('AE loaded!')

# for discriminator
discriminator = build_disc(CONFIG)

print('Loading discriminator...')
ckpt = torch.load(os.path.join('checkpoints',args.exp,'discriminator.pth'))
discriminator.load_state_dict(ckpt)
print('discriminator loaded!')

# alert
alert = GanAlert(discriminator=discriminator, args=args, CONFIG=CONFIG, generator=model)


def evaluation():

    reconstructed, inputs, scores, labels = test(CONFIG.test_loader)
    results = alert.evaluate(scores, labels, collect=True)
        
    # log metrics
    msg = '[TEST metrics] '
    for k, v in results.items():
        msg += k + ': '
        msg += '%.2f ' % v
    log(log_file, msg)

    save_image(os.path.join(save_path, 'test'), zip(reconstructed, inputs))

def test(dataloader):
    model.eval()

    # for reconstructed img
    reconstructed = []
    # for input img
    inputs = []
    # for anomaly score
    scores = []
    # for gt labels
    labels = []

    count = 0
    for i, (img, label) in enumerate(dataloader):
        count += img.shape[0]
        img = img.to(CONFIG.device)
        label = label.cpu()

        out = model(img)
        fake_v = discriminator(out['recon'])

        scores += list(fake_v.detach().cpu().numpy())
        labels += list(label.detach().cpu().numpy())
        reconstructed += list(out['recon'].detach().cpu().numpy())
        inputs += list(img.detach().cpu().numpy())

    return reconstructed, inputs, scores, labels

# load data
file_path = 'checkpoints/zhang_exp14/log.txt'

# extract the data
epochs_1 = []
train_loss = []
recon_loss = []
g_loss = []
d_loss = []
t_recon_loss = []
dist_loss = []


# split data into columns
with open(file_path, 'r') as file:
    for line in file:
        if line.startswith('Epoch:'):
            parts = line.split()
            epoch_1 = int(parts[1])
            t_loss = float(parts[4])
            r_loss = float(parts[6])
            gen_loss = float(parts[8])
            dis_loss = float(parts[10])
            # t_r_loss = float(parts[12])
            # distillation_loss = float(parts[14])

            epochs_1.append(epoch_1)
            train_loss.append(t_loss)
            recon_loss.append(r_loss)
            g_loss.append(gen_loss)
            d_loss.append(dis_loss)
            # t_recon_loss.append(t_r_loss)
            # dist_loss.append(distillation_loss)

# plot the loss curves
plt.figure(figsize=(10, 6))
plt.plot(epochs_1, train_loss, label='Train Loss')
plt.plot(epochs_1, recon_loss, label='Recon Loss')
plt.plot(epochs_1, g_loss, label='G Loss')
plt.plot(epochs_1, d_loss, label='D Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Metrics over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# extract data
auc = []
acc = []
recall = []
f1 = []
precision = []

# split data into columns
with open(file_path, 'r') as file:
    for line in file:
        if line.startswith('[VAL metrics]'):
            parts = line.split()
            auc_curve = float(parts[5])
            accuracy = float(parts[7])
            recall_num = float(parts[11])
            f1_score = float(parts[9])
            precision_num = float(parts[13])

            auc.append(auc_curve)
            acc.append(accuracy)
            recall.append(recall_num)
            f1.append(f1_score)
            precision.append(precision_num)

# plot the change of auc,acc,recall,f1,precision
plt.figure(figsize=(10, 6))
#plt.plot(epochs_1, auc, label='AUC(%)')
plt.plot(epochs_1, acc, label='Accuracy(%)')
#plt.plot(epochs_1, recall, label='Recall(%)')
#plt.plot(epochs_1, f1, label='F1 Score')
#plt.plot(epochs_1, precision, label='Precision(%)')

plt.xlabel('Epoch')
plt.ylabel('Rate(%)')
plt.title('Validation Metrics over Epochs')
plt.legend()
plt.grid(True)
plt.show()

if __name__ == '__main__':
    evaluation()
