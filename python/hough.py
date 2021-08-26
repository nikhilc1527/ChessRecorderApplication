import math
import numpy as np
from collections import defaultdict
import scipy.cluster as cluster
import scipy.spatial as spatial
from matplotlib import pyplot as plt
import numpy
from PIL import Image, ImageDraw
import cv2


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
    print(hough_lines)
    h_lines, v_lines = h_v_lines(hough_lines)
    print(h_lines)
    print(v_lines)
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
