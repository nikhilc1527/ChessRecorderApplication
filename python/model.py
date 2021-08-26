from engine import train_one_epoch, evaluate
import transforms as T
import utils
import torchvision
import os
import torch.utils.data
import torch
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


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
        return self.layers(x)


def get_instance_segmentation_model(num_classes):
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

    # print(model)
    # exit(0)
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



def setup_model(ChessDataset):
    # use our dataset and defined transformations
    dataset = ChessDataset('../train/', '_annotations.coco.json', get_transform(train=True))
    dataset_test = ChessDataset('../valid/', '_annotations.coco.json', get_transform(train=False))

    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    indices_test = torch.randperm(len(dataset_test)).tolist()
    dataset_ = torch.utils.data.Subset(dataset, indices[:len(indices) // 2])
    dataset_test_ = torch.utils.data.Subset(dataset_test, indices_test)

    data_loader = torch.utils.data.DataLoader(
        dataset_, batch_size=2, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test_, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    device = torch.device('cpu')

    num_classes = 14

    model = get_instance_segmentation_model(num_classes)
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
    if os.path.isfile('../models/model_{}epoch_statedict'.format(num_epochs)):
        model.load_state_dict(torch.load(
            '../models/model_{}epoch_statedict'.format(num_epochs), map_location=device))

    else:
        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(model, optimizer, data_loader,
                            device, epoch, print_freq=1)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            evaluate(model, data_loader_test, device=device)

    return model, dataset, dataset_test
