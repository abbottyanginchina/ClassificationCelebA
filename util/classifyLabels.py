import os
import shutil
import numpy as np
import pandas as pd

import cv2 as cv
import matplotlib.pyplot as plt

df = pd.read_csv('../data/celeba/list_attr_celeba.csv')

img_id = df['image_id'].values
gen = df['Male'].values
glass = df['Eyeglasses'].values

for idx in range(len(img_id)):

    if (gen[idx] == -1) & (glass[idx] == -1):
        shutil.copyfile('../data/celeba/img_align_celeba/'+img_id[idx], './data/celeba/Female_noglass/'+img_id[idx])
    elif (gen[idx] == -1) & (glass[idx] == 1):
        shutil.copyfile('../data/celeba/img_align_celeba/'+img_id[idx], './data/celeba/Female_glass/'+img_id[idx])
    elif (gen[idx] == 1) & (glass[idx] == -1):
        shutil.copyfile('../data/celeba/img_align_celeba/'+img_id[idx], './data/celeba/Male_noglass/'+img_id[idx])
    elif (gen[idx] == 1) & (glass[idx] == 1):
        shutil.copyfile('../data/celeba/img_align_celeba/'+img_id[idx], './data/celeba/Male_glass/'+img_id[idx])

