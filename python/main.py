import PIL.ImageShow
# from engine import train_one_epoch, evaluate
import math
# import numpy as np
# from collections import defaultdict
# import scipy.cluster as cluster
# import scipy.spatial as spatial
# from matplotlib import pyplot as plt
# import numpy
from PIL import Image, ImageDraw
# import transforms as T
# import utils
# import torchvision
# import os
import torch
import torch.utils.data
# import glob
# import json
import torch
# from torch import nn
# from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor  # , FastRCNNOutputLayers
# import cv2
# from d2go.export.api import convert_and_export_predictor
# from d2go.export.d2_meta_arch import patch_d2_meta_arch
# from d2go.runner import create_runner, GeneralizedRCNNRunner
# from d2go.model_zoo import model_zoo

# my modules
from dataset import ChessDataset

import model


class MyViewer(PIL.ImageShow.UnixViewer):
    def get_command_ex(self, file, **options):
        command = executable = "sxiv"
        return command, executable


PIL.ImageShow.register(MyViewer, -1)

model, dataset, dataset_test = model.setup_model(ChessDataset)

# torch.save(model2.state_dict(), 'char_model_{}epoch_statedict'.format(num_epochs2))

ind = 3

# pick one image from the test set
img, obj = dataset_test[ind]

model.eval()
with torch.no_grad():
    prediction = model([img.to(torch.device('cpu'))])


import hough

img_path = dataset_test.get_path(ind)

x, y, clusterpoints = hough.grid_points(img_path, ind, obj, dataset, dataset_test)

def get_name(categories, id):
    for c in categories:
        if c['id'] == id:
            return c['name']
    return 'none'


image = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
imgdraw = ImageDraw.Draw(image)
# for bbox, label in zip(obj['boxes'], obj['labels']):
#     shape = [(bbox[0], bbox[1]), (bbox[2], bbox[3])]
#     imgdraw.rectangle(shape)
#     imgdraw.text((bbox[0], bbox[1]), get_name(dataset.categories, label))


class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


points = []
for x_val, y_val in zip(x, y):
    points.append((x_val, y_val))


class Piece:
    def __init__(self, label=0, bbox=((0, 0), (0, 0)), corner_bl=(0, 0),
                 corner_br=(0, 0), square='a1', score=0):
        self.label = label
        self.bbox = bbox
        self.corner_bl = corner_bl
        self.corner_br = corner_br
        self.square = square
        self.score = score


pieces = []

for point in points:
    width = 4
    imgdraw.ellipse(((point[0] - width/2, point[1] - width/2), (point[0] + width/2, point[1] + width/2)), fill='red')


for bbox, label, score in zip(prediction[0]['boxes'], prediction[0]['labels'],
                              prediction[0]['scores']):
    # if score < 0.3:
    #   continue
    # shape = [(bbox[0], bbox[1]), (bbox[2], bbox[3])]
    # imgdraw.rectangle(shape)
    # imgdraw.text((bbox[0], bbox[1]), get_name(dataset.categories, label))
    # imgdraw.text((bbox[0] + 20, bbox[1] + 20), str(score))
    # print(bbox)
    # print(score)

    lb = [bbox[0], bbox[3]]
    rb = [bbox[2], bbox[3]]
    closest_br, closest_bl = -1, -1
    min_br, min_bl = 1e9, 1e9
    for point in clusterpoints:
        br_dist = math.sqrt((rb[0] - point[0])**2 + (rb[1] - point[1])**2)
        bl_dist = math.sqrt((lb[0] - point[0])**2 + (lb[1] - point[1])**2)
        if br_dist < min_br:
            min_br = br_dist
            closest_br = point
        if bl_dist < min_bl:
            min_bl = bl_dist
            closest_bl = point

    new_piece = Piece(label=label, bbox=bbox, corner_bl=closest_bl, corner_br=closest_br, score=score)
    pieces.append(new_piece)

# filter out all of same piece/different labels, keeping only maximum score for each square
piece_dict = {}
for i in range(len(pieces)):
    piece = pieces[i]
    if (piece.corner_bl, piece.corner_br) in piece_dict:
        if piece_dict[(piece.corner_bl, piece.corner_br)].score < piece.score:
            piece_dict[(piece.corner_bl, piece.corner_br)] = piece

    else:
        piece_dict[(piece.corner_bl, piece.corner_br)] = piece

# print(piece_dict)

pieces = []
for piece in piece_dict:
    pieces.append(piece_dict[piece])
    # print(piece)

for piece in pieces:
    # imgdraw.ellipse((piece.corner_bl[0], piece.corner_bl[1],
    #                  piece.corner_bl[0] + 2, piece.corner_bl[1] + 2), fill='red')
    # imgdraw.ellipse((piece.corner_br[0], piece.corner_br[1],
    #                  piece.corner_br[0] + 2, piece.corner_br[1] + 2), fill='blue')
    bbox = piece.bbox
    label = piece.label
    score = piece.score
    shape = [(bbox[0], bbox[1]), (bbox[2], bbox[3])]
    imgdraw.rectangle(shape, outline=(0, 0, 255))
    imgdraw.text((bbox[0], bbox[1]), get_name(dataset.categories, label), fill=(0, 0, 0))
    # imgdraw.text((bbox[0] + 20, bbox[1] + 20), str(round(score.item(), 3)))

image.show()
