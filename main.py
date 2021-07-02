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
import os
import torch.utils.data
import glob
import json
import torch
from torch import nn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2


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
            while j < len(self.annotations) and self.annotations[j]['image_id'] is self.annotations[i]['image_id']:
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
        ann_range = self.indices[idx]

        img_obj = self.obj['images'][self.annotations[ann_range.start]
                                     ['image_id']]
        img_name = img_obj['file_name']
        img = Image.open(os.path.join(self.root, img_name))

        target = {}
        target["boxes"] = torch.as_tensor([([ann['bbox'][0], ann['bbox'][1],
                                             ann['bbox'][0]+ann['bbox'][2],
                                             ann['bbox'][1]+ann['bbox'][3]])
                                           for ann in self.annotations
                                           [ann_range.start:ann_range.stop]],
                                          dtype=torch.float32)
        target["labels"] = torch.tensor(list(
            ann['category_id'] for ann in self.annotations
            [ann_range.start:ann_range.stop]), dtype=torch.int64)
        target["image_id"] = torch.tensor(
            [self.annotations[ann_range.start]['image_id']])
        target["area"] = torch.tensor(list(
            ann['area'] for ann in self.annotations
            [ann_range.start:ann_range.stop]), dtype=torch.int64)
        target["iscrowd"] = torch.tensor(list(
            ann['iscrowd'] for ann in self.annotations
            [ann_range.start:ann_range.stop]), dtype=torch.int8)
        target["masks"] = torch.as_tensor(
            [[[False] * img.height] * img.width] *
            len(ann_range), dtype=torch.uint8)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def get_path(self, idx):
        img_obj = self.obj['images'][self.annotations[idx]['image_id']]
        img_name = img_obj['file_name']
        return os.path.join(self.root, img_name)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)


def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.position_predictor = MLP()

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


# use our dataset and defined transformations
dataset = ChessDataset('train/', get_transform(train=True))
dataset_test = ChessDataset('valid/', get_transform(train=False))
# PennFudanPed
# dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
# dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))

# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
indices_test = torch.randperm(len(dataset_test)).tolist()
dataset_ = torch.utils.data.Subset(dataset, indices[:len(indices)//2])
dataset_test_ = torch.utils.data.Subset(dataset_test, indices_test)

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset_, batch_size=2, shuffle=True, num_workers=0,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test_, batch_size=1, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 14

# get the model using our helper function
model = get_instance_segmentation_model(num_classes)
# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

# number of epochs to train for
num_epochs = 20
if os.path.isfile('./model_{}epoch_statedict'.format(num_epochs)):
    model.load_state_dict(torch.load(
        './model_{}epoch_statedict'.format(num_epochs), map_location=device))
else:

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader,
                        device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)


ind = 4
for i in range(len(dataset_test)):
    print('{}: {}'.format(i, len(dataset_test[i][1]['labels'])))
    if len(dataset_test[i][1]['labels']) > len(dataset_test[ind][1]['labels']):
        ind = i
print(ind)

# pick one image from the test set
img, obj = dataset_test[ind]
# put the model in evaluation mode
model.eval()
with torch.no_grad():
    prediction = model([img.to(device)])


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
    imgdraw.text((bbox[0], bbox[1]), get_name(dataset.categories, label))


img_path = dataset_test.get_path(ind)


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

plt.subplot(121), plt.imshow(cv_img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

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
    y_vals = [(line[0] - x_vals[0]*math.cos(line[1]))/math.sin(line[1]),
              (line[0] - x_vals[1]*math.cos(line[1]))/math.sin(line[1])]
    # plt.plot(x_vals, y_vals)

for line in v_lines:
    y_vals = [0, 500]
    x_vals = [(line[0] - y_vals[0]*math.sin(line[1]))/math.cos(line[1]),
              (line[0] - y_vals[1]*math.sin(line[1]))/math.cos(line[1])]
    # plt.plot(x_vals, y_vals)

# plt.show()

# print(h_lines)

# print(len(intersections))


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
image = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
imgdraw = ImageDraw.Draw(image)
for bbox, label, score in zip(prediction[0]['boxes'], prediction[0]['labels'],
                              prediction[0]['scores']):
    # if score < 0.3:
    #   continue
    # shape = [(bbox[0], bbox[1]), (bbox[2], bbox[3])]
    # imgdraw.rectangle(shape)
    # imgdraw.text((bbox[0], bbox[1]), get_name(dataset.categories, label))
    # imgdraw.text((bbox[0] + 20, bbox[1] + 20), str(score))

    bl = [bbox[0], bbox[3]]
    br = [bbox[2], bbox[3]]
    closest_br, closest_bl = -1, -1
    min_br, min_bl = 1e9, 1e9
    for point in clusterpoints:
        br_dist = math.sqrt((br[0] - point[0])**2 + (br[1] - point[1])**2)
        bl_dist = math.sqrt((bl[0] - point[0])**2 + (bl[1] - point[1])**2)
        if br_dist < min_br:
            min_br = br_dist
            closest_br = point
        if bl_dist < min_bl:
            min_bl = bl_dist
            closest_bl = point

    new_piece = Piece(label=label, bbox=bbox, corner_bl=closest_bl, corner_br=closest_br, score=score)
    pieces.append(new_piece)

print('hehehehe')

piece_dict = {}
for i in range(len(pieces)):
    piece = pieces[i]
    if (piece.corner_bl, piece.corner_br) in piece_dict:
        if piece_dict[(piece.corner_bl, piece.corner_br)].score < piece.score:
            piece_dict[(piece.corner_bl, piece.corner_br)] = piece

    else:
        piece_dict[(piece.corner_bl, piece.corner_br)] = piece

print(piece_dict)

pieces = []
for piece in piece_dict:
    pieces.append(piece_dict[piece])
    print(piece)

for piece in pieces:
    imgdraw.ellipse((piece.corner_bl[0], piece.corner_bl[1],
                     piece.corner_bl[0] + 2, piece.corner_bl[1] + 2), fill='red')
    imgdraw.ellipse((piece.corner_br[0], piece.corner_br[1],
                     piece.corner_br[0] + 2, piece.corner_br[1] + 2), fill='blue')
    bbox = piece.bbox
    label = piece.label
    score = piece.score
    shape = [(bbox[0], bbox[1]), (bbox[2], bbox[3])]
    imgdraw.rectangle(shape)
    imgdraw.text((bbox[0], bbox[1]), get_name(dataset.categories, label))
    # imgdraw.text((bbox[0] + 20, bbox[1] + 20), str(round(score.item(), 3)))

image.show()
