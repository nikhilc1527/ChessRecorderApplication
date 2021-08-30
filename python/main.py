#!/bin/env python3

import PIL
from PIL import ImageShow, Image
import random

# my modules
from dataset import ChessDataset

import model
import hough
import apply_model


class MyViewer(PIL.ImageShow.UnixViewer):
    def get_command_ex(self, file, **options):
        command = executable = "sxiv"
        return command, executable


PIL.ImageShow.register(MyViewer, -1)

piece_model, piece_dataset, piece_dataset_test = model.setup_model(ChessDataset, '..', '_annotations.coco.json', "piece_", 14, 20)
char_model, char_dataset, char_dataset_test = model.setup_model(ChessDataset, '..', '_characters_annotations.coco.json', "char_", 18, 5)


def run(img_path, ind=None):
    # img = Image.open(img_path)
    if ind:
        img, _ = piece_dataset_test[ind]
    else:
        img, _ = piece_dataset_test.get_dummy(img_path)

    points, pieces, chars = apply_model.apply_model(piece_model, char_model, img_path, img)

    img2 = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())

    # apply_model.draw(img2, points, pieces, piece_dataset)
    # apply_model.draw_points(img2, chars)

    pos = hough.get_position(chars, pieces, points, img2)

    apply_model.draw_points(img2, points, 'orange')

    img2.show()

    return pos


if __name__ == "__main__":
    i = random.randint(0, 60)
    # i = 4
    print(i)
    run(piece_dataset_test.get_path(i))
