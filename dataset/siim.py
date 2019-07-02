from __future__ import print_function

from collections import defaultdict, deque
import datetime
import pickle
import time
import torch.distributed as dist
import errno

import collections
import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image, ImageFile
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
import torchvision

from util import mask2rle, rle2mask

ImageFile.LOAD_TRUNCATED_IMAGES = True

class SIIM_MaskRCNN_Dataset(torch.utils.data.Dataset):
    def __init__(self, df_path, img_dir, lst_dir, fold=0, phrase='train'):
        self.df = pd.read_csv(df_path)
        self.height = 256
        self.width = 256
        self.image_dir = img_dir
        self.image_info = collections.defaultdict(dict)
        
        # combine multiple masks in the same sample
        
        for index, row in tqdm(self.df.iterrows(), total=len(self.df)):
            image_id = row['ImageId']
            
            if image_id in self.image_info.keys():
                # append data to image_info dict
                self.image_info[image_id]['annotations'].append(row[' EncodedPixels'].strip())
            else:
                image_path = os.path.join(self.image_dir, image_id)
                
                if os.path.exists(image_path + '.png') and row[" EncodedPixels"].strip() != '-1':
                    self.image_info[image_id]["image_path"] = image_path
                    self.image_info[image_id]["annotations"] = [row[" EncodedPixels"].strip()]

        list_file = os.path.join(lst_dir, '{}-{}.csv'.format(phrase, fold))
        list_pd = pd.read_csv(list_file)

        self.sample_list = list(list_pd['ImageId'])

    def __getitem__(self, idx):
        image_id = self.sample_list[idx]
        info = self.image_info[image_id]

        img_path = info["image_path"]
        img = Image.open(img_path + '.png').convert("RGB")
        width, height = img.size
        img = img.resize((self.width, self.height), resample=Image.BILINEAR)
        
        boxes = []
        masks = []

        for anno in info['annotations']:
            mask = rle2mask(anno, width, height).T
            mask = Image.fromarray(mask)
            mask = mask.resize((self.width, self.height), resample=Image.BILINEAR)
            mask = np.expand_dims(mask, axis=0)

            pos = np.where(np.array(mask)[0, :, :])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            box = [xmin, ymin, xmax, ymax]

            boxes.append(box)
            masks.append(mask.squeeze())

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        masks = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.float32)
        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        labels = torch.ones((len(areas),), dtype=torch.int64)
        image_id = torch.tensor([idx])
        iscrowd = torch.zeros((len(areas),), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks / 255
        target["image_id"] = image_id
        target["area"] = areas
        target["iscrowd"] = iscrowd

        return transforms.ToTensor()(img), target

    def __len__(self):
        return len(self.sample_list)
