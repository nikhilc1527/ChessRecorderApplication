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

piece_model, piece_dataset, piece_dataset_test = model.setup_model(ChessDataset, '..', '_annotations.coco.json')
char_model, char_dataset, char_dataset_test = model.setup_model(ChessDataset, '..', '_characters_annotations.coco.json')

# torch.save(model2.state_dict(), 'char_model_{}epoch_statedict'.format(num_epochs2))

ind = 3

img_path = piece_dataset_test.get_path(ind)

# pick one image from the test set
img, _ = piece_dataset_test[ind]
# img = Image.open(img_path)

# EVERYTHING AFTER THIS IS GOING ------------------------------------------------------------------------

import apply_model
# imgdraw = ImageDraw.Draw(image)

points, pieces = apply_model.apply_model(piece_model, char_model, img_path, img)

img2 = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())

apply_model.draw(img2, points, pieces, piece_dataset)

img2.show()
