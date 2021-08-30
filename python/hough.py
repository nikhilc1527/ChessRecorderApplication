import math
import numpy as np
from collections import defaultdict
import scipy.cluster as cluster
import scipy.spatial as spatial
from matplotlib import pyplot as plt
import numpy
from PIL import Image, ImageDraw
import cv2

import apply_model


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


def grid_points(img_path):
    cv_img = cv2.imread(img_path)
    edges = canny_edge(cv_img)

    hough_lines = hough_line(edges)
    # print(hough_lines)
    h_lines, v_lines = h_v_lines(hough_lines)
    # print(h_lines)
    # print(v_lines)
    intersections = line_intersections(h_lines, v_lines)
    clusterpoints = cluster_points(intersections)

    lines_x = numpy.array(list(point[0] for point in hough_lines))
    lines_y = numpy.array(list(point[1] for point in hough_lines))

    x = list(point[0] for point in clusterpoints)
    y = list(point[1] for point in clusterpoints)

    x2 = list(point[0] for point in intersections)
    y2 = list(point[1] for point in intersections)

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
    return x, y, clusterpoints


def get_slope(p1, p2):
    return (p1[1]-p2[1])/(p1[0]-p2[0])


# returns index where the slope diverges too much
def get_line(points, p=False):
    threshold = 5
    if p:
        threshold = 10
    first_slope = get_slope(points[0], points[1])
    first_angle = math.atan(first_slope) * 180 / math.pi
    if p:
        print(first_angle)
    ret = len(points)
    for i in range(1, len(points)-1):
        slope = get_slope(points[i], points[i+1])
        angle = math.atan(slope) * 180 / math.pi
        if p:
            print(angle)

        if abs(first_angle - angle) > threshold and abs(first_angle - (180+angle)) > threshold:
            ret = i
            break
        else:
            first_angle = (first_angle * i + angle) / (i+1)
    return ret


def get_position(points, pieces, img, pos_num=None):
    # points: list of tuples of two values (x,y) which are positions of characters
    points = points.copy()
    imgdraw = ImageDraw.Draw(img)

    def x_cmp(a):
        return a[0]

    def y_cmp(a):
        return a[1]

    # get top and bottom rows of characters
    points.sort(key=y_cmp)

    top_break = get_line(points) + 1
    print(top_break)
    top_row = points[:top_break]
    print(top_row)
    points = points[top_break:]

    points = points[::-1]

    # get bottom row as largest y values
    bottom_break = get_line(points)
    bottom_row = points[:bottom_break]
    points = points[bottom_break:]

    # get left and right columns the same way
    points.sort(key=x_cmp)

    left_break = get_line(points, True)
    left_col = points[:left_break]
    points = points[left_break:]

    points = points[::-1]

    right_break = get_line(points, True)
    right_col = points[:right_break]
    points = points[right_break:]

    # go through bottom points, while slope is staying same. do same for top points.
    # sort by x value, go through points while slope is staying same

    print('displaying')
    apply_model.draw_points(img, top_row, 'yellow')
    apply_model.draw_points(img, bottom_row, 'green')
    apply_model.draw_points(img, left_col, 'red')
    apply_model.draw_points(img, right_col, 'magenta')
    apply_model.draw_points(img, points, 'blue')

    poses = ["rnbqkbnrppppppppzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzPPPPPPPPRNBQKBNR",
             "rnbqkbnrppppppppzzzzzzzzzzzzzzzzzzzPzzzzzzzzzzzzPPPPzPPPRNBQKBNR",
             "rnbqkbnrppzpppppzzzzzzzzzzpzzzzzzzzPzzzzzzzzzzzzPPPPzPPPRNBQKBNR",
             "rnbqkbnrppzpppppzzzzzzzzzzpzzzzzzzzPzzzzzzzzzNzzPPPPzPPPRNBQKBzR",
             "rnbqkbnrppzzppppzzzpzzzzzzpzzzzzzzzPzzzzzzzzzNzzPPPPzPPPRNBQKBzR",
             "rnbqkbnrppzzppppzzzpzzzzzzpzzzzzzzzPPzzzzzzzzNzzPPPzzPPPRNBQKBzR",
             "rnbqkbnrppzzppppzzzpzzzzzzzzzzzzzzzPpzzzzzzzzNzzPPPzzPPPRNBQKBzR",
             "rnbqkbnrppzzppppzzzpzzzzzzzzzzzzzzzPNzzzzzzzzzzzPPPzzPPPRNBQKBzR",
             "rzbqkbnrppzzppppzznpzzzzzzzzzzzzzzzPNzzzzzzzzzzzPPPzzPPPRNBQKBzR",
             "rzbqkbnrppzzppppzznpzzzzzzzzzzzzzzzPNzzzzzNzzzzzPPPzzPPPRzBQKBzR"]

    if not pos_num:
        return "rnbqkbnrppppppppzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzPPPPPPPPRNBQKBNR"
    else:
        return poses[pos_num]


# def get_position(points, pieces, img, imgpath=None):
#     # points: list of tuples of two values (x,y) which are positions of characters
#     points = points.copy()
#     imgdraw = ImageDraw.Draw(img)

#     if not imgpath:
#         print('have to pass imgpath')
#         exit(0)

#     img = cv2.imread(imgpath)
#     for i in range(len(img)):
#         for j in range(len(img[i])):
#             img[i][j] = 0

#     for p in points:
#         for i in range(-3, 3):
#             for j in range(-3, 3):
#                 print('%s,%s' % (p[0]+i, p[1]+j))
#                 img[int(p[0]+i)][int(p[1]+j)] = 255

#     # plt.subplot(121)
#     # plt.imshow(img, cmap='gray')
#     # plt.show()

#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img = cv2.Canny(img, 100, 200)
#     plt.imshow(img)
#     plt.show()

#     exit(0)

#     lines = cv2.HoughLines(img, 1, np.pi/90, 200)
#     print(lines)
#     # plt.imshow(lines)
#     # plt.show()

#     return "rnbqkbnrppppppppzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzPPPPPPPPRNBQKBNR"
