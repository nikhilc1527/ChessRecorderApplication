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
    # first_slope = get_slope(points[0], points[1])
    # first_angle = math.atan(first_slope) * 180 / math.pi
    total_angle = math.atan2(points[0][1] - points[1][1], points[0][0] - points[1][0]) * 180 / math.pi
    # while total_angle < -45:
    #     total_angle += 90
    avg_angle = total_angle
    total = 1

    print("avg_angle: %s" % (avg_angle))

    ret = len(points)
    for i in range(1, len(points)-1):
        # slope = get_slope(points[i], points[i+1])
        # angle = math.atan(slope) * 180 / math.pi
        angle = math.atan2(points[i][1] - points[i+1][1], points[i][0] - points[i+1][0]) * 180 / math.pi
        # while angle < 135:
        #     angle += 90

        # print(angle)

        if abs(avg_angle - angle) > threshold and abs(avg_angle - (180+angle)) > threshold:
            ret = i + 1
            # first_angle = (first_angle * i + angle) / (i+1)
            # total_angle += angle
            # avg_angle = total_angle / i
            break
        else:
            # first_angle = (first_angle * i + angle) / (i+1)
            if abs(avg_angle - angle) < threshold:
                total_angle += angle
                total += 1
                avg_angle = total_angle / (total)
            # avg_angle = math.atan2(points[i+1][1]-points[0][1], points[i+1][0]-points[0][0])
            # print("angle: " + str(angle))
            # print("avg angle: " + str(avg_angle))
            pass

    # avg_angle = (avg_angle + 180) % 180
    # if avg_angle < 0:
    #     avg_angle *= -1
    #     avg_angle = 180 - avg_angle
    print('overall angle: %s' % (math.atan2(points[ret-1][1]-points[0][1], points[ret-1][0]-points[0][0])))
    # first_angle = first_angle / 180 * math.pi
    return ret, avg_angle


def get_position(chars, pieces, points, img):
    # points: list of tuples of two values (x,y) which are positions of characters
    imgdraw = ImageDraw.Draw(img)

    def x_cmp(a):
        return a[0]

    def y_cmp(a):
        return a[1]

    # set of the 4 (list of points, angle) of top, bottom, left, and right
    lines = []

    # get top and bottom rows of characters
    chars.sort(key=y_cmp)

    top_break, top_angle = get_line(chars)
    print("top_angle: %s" % (top_angle))
    top_row = chars[:top_break]
    top_row.sort(key=x_cmp)
    lines.append((top_row, top_angle))
    chars = chars[top_break:]

    chars = chars[::-1]

    # get bottom row as largest y values
    bottom_break, bottom_angle = get_line(chars)
    print("bottom_angle: %s" % (bottom_angle))
    bottom_row = chars[:bottom_break]
    bottom_row.sort(key=x_cmp)
    lines.append((bottom_row, bottom_angle))
    chars = chars[bottom_break:]

    # get left and right columns the same way
    chars.sort(key=x_cmp)

    left_break, left_angle = get_line(chars# , True
                                      )
    print("left_angle: %s" % (left_angle))
    left_col = chars[:left_break]
    left_col.sort(key=y_cmp)
    lines.append((left_col, left_angle))
    chars = chars[left_break:]

    chars = chars[::-1]

    right_break, right_angle = get_line(chars# , True
                                        )
    print("right_angle: %s" % (right_angle))
    right_col = chars[:right_break]
    right_col.sort(key=y_cmp)
    lines.append((right_col, right_angle))
    chars = chars[right_break:]

    # go through bottom chars, while slope is staying same. do same for top chars.
    # sort by x value, go through chars while slope is staying same

    print('displaying')
    apply_model.draw_points(img, top_row, 'yellow')
    apply_model.draw_points(img, bottom_row, 'green')
    apply_model.draw_points(img, left_col, 'red')
    apply_model.draw_points(img, right_col, 'magenta')
    apply_model.draw_points(img, chars, 'blue')

    # imgdraw.line([int(top_row[0][0]), int(top_row[0][1]), int(top_row[-1][0]), int(top_row[-1][1])])
    # imgdraw.line([int(bottom_row[0][0]), int(bottom_row[0][1]), int(bottom_row[-1][0]), int(bottom_row[-1][1])])
    # imgdraw.line([int(left_col[0][0]), int(left_col[0][1]), int(left_col[-1][0]), int(left_col[-1][1])])
    # imgdraw.line([int(right_col[0][0]), int(right_col[0][1]), int(right_col[-1][0]), int(right_col[-1][1])])

    for line in lines:
        line = line[0]
        for i in range(len(line)):
            line[i] = (int(line[i][0]), int(line[i][1]))

    for line in lines:
        x = line[0][0][0]
        y = line[0][0][1]
        x1 = 0
        y1 = y + x * math.tan(line[1] / 180 * math.pi)
        y2 = line[0][1][1]
        x2 = x + y / math.tan(line[1] / 180 * math.pi)
        # imgdraw.line([line[0][0][0], line[0][0][1], line[0][-1][0], line[0][-1][1]])
        imgdraw.line([x, y, x2, y2])

    i = 0
    while i < len(points):
        if i >= len(points):
            break
        point = points[i]
        hits = 0
        for line in lines:
            first_point = line[0][0]
            angle = line[1]
            if (point[0] - first_point[0]) < \
               ((point[1] - first_point[1]) / math.tan(angle * math.pi / 180)):
                hits += 1
        imgdraw.text(point, str(hits))
        if not(hits == 1):
            del points[i]
            i -= 1

        i += 1

    avg_dist = 0

    for i in range(len(points)):
        closest_j, closest = -1, float('inf')
        for j in range(len(points)):
            if i == j:
                continue
            dist = math.sqrt((points[j][0]-points[i][0])**2+(points[j][1]-points[i][1])**2)
            if dist < closest:
                closest = dist
                closest_j = j
        avg_dist = (avg_dist * i + closest) / (i+1)

    points.sort(key=y_cmp)
    points.reverse()
    points.sort(key=x_cmp)
    # first index in points is bottom left corner of board

    bottom_left_point = points[0]
    position = ''
    for i in range(64):
        position += 'z'
    for piece in pieces:
        dist_x = int(round(abs(piece.corner_bl[0] - bottom_left_point[0]) / avg_dist))
        dist_y = int(round(abs(piece.corner_bl[1] - bottom_left_point[1]) / avg_dist))
        imgdraw.text([piece.bbox[0], piece.bbox[1]], '(%s, %s)' % (dist_x, dist_y))

    return "rnbqkbnrppppppppzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzPPPPPPPPRNBQKBNR"
