import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import glob
import json


class ChessDataset(torch.utils.data.Dataset):
  def __init__(self, root, transforms=None):
      self.root = root
      self.transforms = transforms
      # load all image files, sorting them to
      # ensure that they are aligned
      self.imgs = list(sorted(glob.glob(os.path.join(root, "*.jpg"))))
      with open(os.path.join(root, "_annotations.coco.json"), 'r') as myfile:
        data = myfile.read()

      self.obj = json.loads(data)
      self.annotations = self.obj['annotations']
      self.categories = self.obj['categories']
      # print(self.categories)
      # print(self.obj['annotations'][0].keys())

      # get range of indices of annotations for each image
      self.indices = []
      # print(self.annotations[0])
      i, j = 0, 0
      while i < len(self.annotations):
        j = i
        while j < len(self.annotations) and self.annotations[j]['image_id'] == self.annotations[i]['image_id']:
          j += 1
        self.indices.append(range(i, j))
        i = j
      # print("imgs len:")
      # print(len(self.imgs))
      # print(self.indices[len(self.indices)-1])
      # print(self.indices)
      # print('indices len:')
      # print(len(self.indices))
      # print(len(self.annotations))

  def __getitem__(self, idx):
      # if idx > len(self.indices) or idx < 0:
      #   print('{} {}'.format(self.root, idx))
      #   ann_range = self.indices[0]
      # else:
      ann_range = self.indices[idx]

      current_annotation = self.annotations[ann_range.start]
      img_obj = self.obj['images'][self.annotations[ann_range.start]['image_id']]
      img_name = img_obj['file_name']
      img = Image.open(os.path.join(self.root, img_name))

      target = {}
      bbox = current_annotation['bbox']
      target["boxes"] = torch.as_tensor([([ann['bbox'][0], ann['bbox'][1], ann['bbox'][0]+ann['bbox'][2], ann['bbox'][1]+ann['bbox'][3]])
                                         for ann in self.annotations[ann_range.start:ann_range.stop]], dtype=torch.float32)
      target["labels"] = torch.tensor(list(
          ann['category_id'] for ann in self.annotations[ann_range.start:ann_range.stop]), dtype=torch.int64)
      target["image_id"] = torch.tensor(
          [self.annotations[ann_range.start]['image_id']])
      target["area"] = torch.tensor(list(
          ann['area'] for ann in self.annotations[ann_range.start:ann_range.stop]), dtype=torch.int64)
      target["iscrowd"] = torch.tensor(list(
          ann['iscrowd'] for ann in self.annotations[ann_range.start:ann_range.stop]), dtype=torch.int8)
      target["masks"] = torch.as_tensor(
          [[[False] * img.height] * img.width] * len(ann_range), dtype=torch.uint8)

      if self.transforms is not None:
          img, target = self.transforms(img, target)

      return img, target

  def __len__(self):
      return len(self.imgs)
