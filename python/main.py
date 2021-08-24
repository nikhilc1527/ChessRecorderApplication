import PIL.ImageShow
from engine import train_one_epoch, evaluate
import math
import numpy as np
from collections import defaultdict
import scipy.cluster as cluster
import scipy.spatial as spatial
from matplotlib import pyplot as plt
import numpy
from PIL import Image, ImageDraw
import transforms as T
import utils
import torchvision
# import os
import torch.utils.data
# import glob
# import json
import torch
# from torch import nn
# from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor  # , FastRCNNOutputLayers
import cv2
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


# use our dataset and defined transformations
dataset = ChessDataset('train/', '_annotations.coco.json', get_transform(train=True))
dataset_test = ChessDataset('valid/', '_annotations.coco.json', get_transform(train=False))
char_dataset = ChessDataset('train/', '_characters_annotations.coco.json', get_transform(train=True))
char_dataset_test = ChessDataset('train/', '_characters_annotations.coco.json', get_transform(train=False))

torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
indices_test = torch.randperm(len(dataset_test)).tolist()
dataset_ = torch.utils.data.Subset(dataset, indices[:len(indices) // 2])
dataset_test_ = torch.utils.data.Subset(dataset_test, indices_test)

torch.manual_seed(2)
char_indices = torch.randperm(len(char_dataset)).tolist()
char_indices_test = torch.randperm(len(char_dataset_test)).tolist()
char_dataset_ = torch.utils.data.Subset(char_dataset, char_indices[1:])
char_dataset_test_ = torch.utils.data.Subset(char_dataset_test, char_indices_test[:1])

data_loader = torch.utils.data.DataLoader(
    dataset_, batch_size=2, shuffle=True, num_workers=0,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test_, batch_size=1, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)

char_data_loader = torch.utils.data.DataLoader(
    char_dataset_, batch_size=2, shuffle=True, num_workers=0,
    collate_fn=utils.collate_fn)

char_data_loader_test = torch.utils.data.DataLoader(
    char_dataset_test_, batch_size=1, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)

device = torch.device('cpu')

num_classes = 14
num_classes2 = 18

model = get_instance_segmentation_model(num_classes)
model2 = get_instance_segmentation_model(num_classes2)
# move model to the right device
model.to(device)
model2.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

params2 = [p for p in model2.parameters() if p.requires_grad]
optimizer2 = torch.optim.SGD(params2, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

lr_scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2,
                                               step_size=3,
                                               gamma=0.1)

# number of epochs to train for
num_epochs = 20
# if os.path.isfile('./model_{}epoch_statedict'.format(num_epochs)):
# model.load_state_dict(torch.load(
#     './model_{}epoch_statedict'.format(num_epochs), map_location=device))

num_epochs2 = 3
# model2.load_state_dict(torch.load(
#     './char_model_{}epoch_statedict'.format(num_epochs2), map_location=device))
# else:
for epoch in range(num_epochs2):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, char_data_loader,
                    device, epoch, print_freq=1)
    # update the learning rate
    lr_scheduler2.step()
    # evaluate on the test dataset
    evaluate(model2, char_data_loader_test, device=device)

# torch.save(model2.state_dict(), 'char_model_{}epoch_statedict'.format(num_epochs2))

ind = 3
# for i in range(len(char_dataset_test)):
#     print('{}: {}'.format(i, len(char_dataset_test[i][1]['labels'])))
#     if len(char_dataset_test[i][1]['labels']) > len(char_dataset_test[ind][1]['labels']):
#         ind = i
# print(ind)

# pick one image from the test set
img, obj = char_dataset_test[ind]
# put the model in evaluation mode
model2.eval()
with torch.no_grad():
    chars_prediction = model2([img.to(device)])

model.eval()
with torch.no_grad():
    pieces_prediction = model([img.to(device)])

def get_name(categories, id):
    for c in categories:
        if c['id'] == id:
            return c['name']
    return 'none'


image = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
imgdraw = ImageDraw.Draw(image)
for bbox, label in zip(obj['boxes'], obj['labels']):
    shape = [(bbox[0], bbox[1]), (bbox[2], bbox[3])]
    imgdraw.rectangle(shape)
    imgdraw.text((bbox[0], bbox[1]), get_name(char_dataset.categories, label))


img_path = char_dataset_test.get_path(ind)
print(img_path)

pieces = []
image = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
imgdraw = ImageDraw.Draw(image)

def pil_to_cv(img):
    return cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)


def grayblur(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.blur(gray, (5, 5))
    return gray_blur


def canny_edge(img, sigma=0.33):
    v = np.median(img)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(img, lower, upper)
    return edges


def hough_line(edges, min_line_length=50, max_line_gap=30):
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 110,
                           min_line_length, max_line_gap)
    lines = np.reshape(lines, (-1, 2))
    return lines


def h_v_lines(lines):
    h_lines, v_lines = [], []
    for rho, theta in lines:
        if theta < np.pi / 4 or theta > np.pi - np.pi / 4:
            v_lines.append([rho, theta])
        else:
            h_lines.append([rho, theta])
    return h_lines, v_lines


def line_intersections(h_lines, v_lines):
    points = []
    for r_h, t_h in h_lines:
        for r_v, t_v in v_lines:
            a = np.array([[np.cos(t_h), np.sin(t_h)],
                          [np.cos(t_v), np.sin(t_v)]])
            b = np.array([r_h, r_v])
            inter_point = np.linalg.solve(a, b)
            points.append(inter_point)
    return np.array(points)


def cluster_points(points):
    dists = spatial.distance.pdist(points)
    single_linkage = cluster.hierarchy.single(dists)
    flat_clusters = cluster.hierarchy.fcluster(single_linkage, 15, 'distance')
    cluster_dict = defaultdict(list)
    for i in range(len(flat_clusters)):
        cluster_dict[flat_clusters[i]].append(points[i])
    cluster_values = cluster_dict.values()
    clusters = map(lambda arr: (np.mean(np.array(arr)[:, 0]), np.mean(
        np.array(arr)[:, 1])), cluster_values)
    return sorted(list(clusters), key=lambda k: [k[1], k[0]])


cv_img = cv2.imread(img_path)
edges = canny_edge(cv_img)

hough_lines = hough_line(edges)
h_lines, v_lines = h_v_lines(hough_lines)
intersections = line_intersections(h_lines, v_lines)
clusterpoints = cluster_points(intersections)

lines_x = numpy.array(list(point[0] for point in hough_lines))
lines_y = numpy.array(list(point[1] for point in hough_lines))

x = list(point[0] for point in clusterpoints)
y = list(point[1] for point in clusterpoints)

x2 = list(point[0] for point in intersections)
y2 = list(point[1] for point in intersections)

# plt.scatter(x,y)
plt.scatter(x, y)

for line in h_lines:
    x_vals = [0, 500]
    y_vals = [(line[0] - x_vals[0] * math.cos(line[1])) / math.sin(line[1]),
              (line[0] - x_vals[1] * math.cos(line[1])) / math.sin(line[1])]
    # plt.plot(x_vals, y_vals)

for line in v_lines:
    y_vals = [0, 500]
    x_vals = [(line[0] - y_vals[0] * math.sin(line[1])) / math.cos(line[1]),
              (line[0] - y_vals[1] * math.sin(line[1])) / math.cos(line[1])]
    # plt.plot(x_vals, y_vals)

# plt.show()

# print(h_lines)

# print(len(intersections))
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


for bbox, label, score in zip(chars_prediction[0]['boxes'], chars_prediction[0]['labels'],
                              chars_prediction[0]['scores']):
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

new_points = []

for point in points:
    lt = False
    rt = False
    lb = False
    rb = False
    for piece in pieces:
        if lt and rt and lb and rb:
            break
        piece = piece.bbox
        # print(piece)
        if piece[0].item() < point[0] and piece[1].item() < point[1]:
            lt = True
        if piece[0].item() < point[0] and piece[3].item() > point[1]:
            lb = True
        if piece[2].item() > point[0] and piece[1].item() < point[1]:
            rt = True
        if piece[2].item() > point[0] and piece[3].item() > point[1]:
            rb = True
    if lt and rt and lb and rb:
        new_points.append(point)

points = new_points
del new_points


bottom_row = points.copy()
bottom_row.sort(key=lambda x: x[1], reverse=True)
bottom_row = bottom_row[0:7]

left_row = points.copy()
left_row.sort(key=lambda x: x[0], reverse=False)
left_row = left_row[0:7]

top_row = points.copy()
top_row.sort(key=lambda x: x[1], reverse=False)
top_row = top_row[0:7]

right_row = points.copy()
right_row.sort(key=lambda x: x[0], reverse=True)
right_row = right_row[0:7]

for point in points:
    width = 4
    imgdraw.ellipse(((point[0] - width/2, point[1] - width/2), (point[0] + width/2, point[1] + width/2)), fill='red')

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
    imgdraw.text((bbox[0], bbox[1]), get_name(char_dataset.categories, label), fill=(0, 0, 0))
    # imgdraw.text((bbox[0] + 20, bbox[1] + 20), str(round(score.item(), 3)))

# ------------------------------------------------------------------------------------------------------------------------

for bbox, label, score in zip(pieces_prediction[0]['boxes'], pieces_prediction[0]['labels'],
                              pieces_prediction[0]['scores']):
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
    imgdraw.text((bbox[0], bbox[1]), get_name(char_dataset.categories, label), fill=(0, 0, 0))
    # imgdraw.text((bbox[0] + 20, bbox[1] + 20), str(round(score.item(), 3)))

image.show()