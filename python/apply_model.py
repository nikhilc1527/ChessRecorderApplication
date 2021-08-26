import torch
from PIL import Image, ImageDraw
import math

import hough

def get_name(categories, id):
    for c in categories:
        if c['id'] == id:
            return c['name']
    return 'none'

class Piece:
    def __init__(self, label=0, bbox=((0, 0), (0, 0)), corner_bl=(0, 0),
                 corner_br=(0, 0), square='a1', score=0):
        self.label = label
        self.bbox = bbox
        self.corner_bl = corner_bl
        self.corner_br = corner_br
        self.square = square
        self.score = score


def draw(img, points, pieces, piece_dataset):
    imgdraw = ImageDraw.Draw(img)
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
        imgdraw.text((bbox[0], bbox[1]), get_name(piece_dataset.categories, label), fill=(0, 0, 0))
        # imgdraw.text((bbox[0] + 20, bbox[1] + 20), str(round(score.item(), 3)))

# filter out all pieces, so that only the best piece per each square stays
def filter_pieces(piece_prediction, clusterpoints):
    pieces = []
    for bbox, label, score in zip(piece_prediction[0]['boxes'],
                                  piece_prediction[0]['labels'],
                                  piece_prediction[0]['scores']):
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
    pieces = []
    for piece in piece_dict:
        pieces.append(piece_dict[piece])

    return pieces


def apply_model(piece_model, char_model, img_path, img):
    piece_model.eval()
    char_model.eval()
    with torch.no_grad():
        piece_prediction = piece_model([img.to(torch.device('cpu'))])
        char_prediction = char_model([img.to(torch.device('cpu'))])
    x, y, clusterpoints = hough.grid_points(img_path)
    points = []
    for x_val, y_val in zip(x, y):
        points.append((x_val, y_val))

    pieces = filter_pieces(piece_prediction, clusterpoints)

    return points, pieces
