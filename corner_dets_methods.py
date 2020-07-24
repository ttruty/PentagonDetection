# Import the required modules
from skimage.transform import pyramid_gaussian
from skimage.io import imread
from skimage.feature import hog
from sklearn.externals import joblib
from sklearn.decomposition import PCA
import cv2, os, time, math, itertools, random
import numpy as np
from skimage import img_as_ubyte
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu
from skimage.filters import gaussian
from itertools import groupby
from operator import itemgetter
import line_equations
import matplotlib.pyplot as plt
import warnings
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from skimage.morphology import skeletonize
from sklearn.preprocessing import binarize
from scipy.cluster.vq import kmeans, vq
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)

cwd = os.getcwd()
path = os.path.join(cwd, 'training_mmse_pentagons')
model_path = os.path.join(path, "models", "svm.model")


class corrections():
    '''
    holder for the x,y hieght width corection from detections
    '''
    x = 0
    y = 0
    w = 0
    h = 0


def black_pixels(img):

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    color_im = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    clone = color_im.copy()
    inv_img = np.invert(img)
    binary = binarize(inv_img, threshold=15.0)
    # binary = np.invert(binary)
    skel = skeletonize(binary)

    cv_skel = img_as_ubyte(skel)
    match_p = np.where(skel == 1)

    im2, contours, hierarchy = cv2.findContours(cv_skel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  #

    all_cnt_p = []
    x = match_p[0]
    y = match_p[1]

    black_coords = list(zip(x, y))

    edit_black = []

    for cnt in contours:
        hull = cv2.convexHull(cnt)
        area = cv2.contourArea(hull)
        if area > 25:
            for i in black_coords:
                inside = cv2.pointPolygonTest(cnt, (i[1], i[0]), False)
                if inside != -1:
                    edit_black.append((i[0], i[1]))
    return edit_black


def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def line_points(start, end):
    "Bresenham's line algorithm"
    x0, y0 = start
    x1, y1 = end
    points_in_line = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points_in_line.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points_in_line.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points_in_line.append((x, y))
    return points_in_line


def cornerMeths(im, detections, corners, quality, distance, line_thresh, line_length, line_gap):
    '''
    Methods for the corner detections including the corrections for the scaling factors
    '''
    x_min = []  # Make container box with all features
    y_min = []
    x_max = []
    y_max = []
    for (x, y, _, w, h) in detections:  # index 2 is detection score
        x_min.append(x)
        y_min.append(y)
        x_max.append(x + w)
        y_max.append(y + h)
    x = min(x_min)
    y = min(y_min)
    w = max(x_max)
    h = max(y_max)
    corrections.x = x
    corrections.y = y
    corrections.w = w
    corrections.h = h

    imCrop = im[y:h, x:w]  # cropped image of the detection
    color_im = cv2.cvtColor(imCrop, cv2.COLOR_GRAY2RGB)

    shi_corners = shiCorners(imCrop, corners, quality, distance)
    lines, distilled_lines = hough_lines(imCrop, line_thresh, line_length, line_gap)
    return shi_corners, lines, distilled_lines


def detection(img):
    '''
    The detections methond using the HOG algorithm
    '''

    clf = joblib.load(model_path)  # prediciton method using SVM
    # Binarize.
    blur = cv2.GaussianBlur(img, (5, 5), 0)  # preprocessing in of image
    ret, thesh_image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thesh_image, cv2.MORPH_OPEN, kernel)
    opening = cv2.resize(opening, (128, 128))

    fd, _ = hog(opening, 9, (8, 8), (3, 3), visualise=True,
                transform_sqrt=True)  # defined parameters for the HOG method
    return clf.decision_function([fd])


def find_contours(path, corners, quality, distance, detection_threshold):
    '''
    cv2 contour finding to get the shapes on page according to the black pixels
    '''
    found_dets = []
    output_image = np.zeros((100, 100, 3), np.uint8) # blank image as placeholder

    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    height, width, channels = img.shape
    print("Original Image Size: Height=" + str(height) + " Width=" + str(width))
    clone = img.copy()

    group_x = []
    group_y = []

    cv2.imshow('before canny image', img)

    blur = cv2.GaussianBlur(im, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    edges = cv2.Canny(img, 0, 255, apertureSize=3)  # Canny image

    cv2.imshow('after canny image', edges)
    cv2.waitKey()

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # create the kernle
    edges = cv2.dilate(edges, kernel, iterations=5)



    im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # find each contour
    cv2.drawContours(clone, contours, -1, (0, 255, 0), 3)  # draw all contours

    found_cnts = []
    rectangles = []
    features = []

    # DRAW Contours
    cv2.drawContours(edges, contours, -1, (0, 255, 255), 3)
    for cnt in contours:
        # get the perimeter circularity of the contours
        hull = cv2.convexHull(cnt)
        area = cv2.contourArea(hull)
        perimeter = cv2.arcLength(hull, True)
        all_contours = []
        #cv2.drawContours(img, cnt, -1, (0, 255, 255), 3)
        if perimeter != 0:
            x, y, w, h = cv2.boundingRect(cnt)
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 167, 120), 2)
            if (h * w) > 3000:  # minimum area of the contour bounding box
                feature = im[y:y + h, x:x + w] # feature is crop of contour
                features.append(feature) # add the feature to list of features

                #SHOW IMAGE
                #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)

                cd = detection(feature) # current detection
                if (cd > detection_threshold):

                    # SHOW IMAGE
                    #cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                    found_dets.append((x, y, cd, w, h))
    det_compares = itertools.combinations(found_dets, 2) # iterate over combinations of each pair of detections
    for i in det_compares:  # Ignore the small detection inside a bigger detection by looking at comparisons
        det1, det2 = i
        x1, y1, _, w1, h1 = det1
        x2, y2, _, w2, h2 = det2
        rect1_p1 = x1, y1
        rect1_p2 = x1 + w1, y1 + h1
        rect2_p1 = x2, y2
        rect2_p2 = x2 + w2, y2 + h2
        if (rect1_p1[0] < rect2_p1[0] < rect2_p2[0] < rect1_p2[0] and rect1_p1[1] < rect2_p1[1] < rect2_p2[1] <
            rect1_p2[1]):
            small = min(i, key=lambda x: x[3] * x[4])

            #cv2.rectangle(img, (small[0], small[1]), (small[0] + small[3], small[1] + small[4]), (167, 167, 0), 2)
            if small in found_dets:
                #cv2.rectangle(img, (small[0], small[1]), (small[0] + small[3], small[1] + small[4]), (167, 167, 0), 2)
                found_dets.remove(small) # remove the smaller detections from the larger
    for i in found_dets:
        x, y, _, w, h = i

        output_image = img[y:y + h, x:x + w] #save the image in output with filename


    return img, found_dets, output_image

def inside_contours(image):
    '''
    cv2 contour finding to get the shapes on page according to the black pixels
    '''
    output_image = np.zeros((100, 100, 3), np.uint8) # blank image as placeholder
    cnt_list = []
    # im = cv2.ctvColor(image, cv2.IMREAD_GRAYSCALE)
    # img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    img = image
    height, width, channels = img.shape
    clone = img.copy()

    edges = cv2.Canny(img, 0, 255, apertureSize=3)  # Canny image
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # create the kernle
    edges = cv2.dilate(edges, kernel, iterations=5)
    im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # find each contour
    cv2.drawContours(clone, contours, -1, (0, 255, 0), 3)  # draw all contours

    # DRAW Contours
    cv2.drawContours(edges, contours, -1, (0, 255, 255), 3)
    for cnt in contours:
        # get the perimeter circularity of the contours
        hull = cv2.convexHull(cnt)
        area = cv2.contourArea(hull)
        perimeter = cv2.arcLength(hull, True)
        all_contours = []
        #cv2.drawContours(img, cnt, -1, (0, 255, 255), 3)
        if perimeter != 0:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 167, 120), 2)
            cnt_list.append([x,y,w,h])

        # output_image = img[y:y + h, x:x + w] #save the image in output with filename
        # cv2.imshow('image', img)
        # cv2.waitKey()
    print(cnt_list)
    return cnt_list



def connectLines(img, corners, line_threshold):
    '''
    connect the found corners in image with lines
    check if the lines are over black pixels
    '''


    color_im = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    edges = cv2.Canny(img, 0, 255, apertureSize=3)  # canny edge transform
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # kernel to dialate
    edges = cv2.dilate(edges, kernel, iterations=5)  # dialate
    org_im = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    color_im = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    copy = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    edges = edges.astype(np.uint8)


    line_count = 0
    found_lines = []
    bad_lines = []
    lines = itertools.combinations(corners, 2)  # create all possible lines
    line_img = np.ones_like(color_im) * 255  # white image to draw line markings on
    for line in lines:  # loop through each line
        bin_line = np.zeros_like(edges)  # create a matrix to draw the line in
        start, end = line  # grab endpoints
        points = line_points(start, end)  # use Bresenham's algorythm
        cv2.line(bin_line, tuple(start), tuple(end), color=255, thickness=1)  # draw line
        conj = (edges / 255 + bin_line / 255)  # create agreement image
        n_agree = np.sum(conj == 2)  # agreement points
        n_wrong = np.sum(conj == 1)  # disagreement poitns

        if n_agree / (len(points)) > line_threshold:  # high agreements vs disagreements
            # cv2.line(org_im, tuple(start), tuple(end), color=[0,200,0], thickness=2)
            line = [start, end]
            found_lines.append(line)
            line_count += 1
        if n_agree / (len(points)) < .95 and n_agree / (len(points)) > .85:  # high agreements vs disagreements
            # cv2.line(org_im, tuple(start), tuple(end), color=[155,0,0], thickness=2)
            line = [start, end]
            bad_lines.append(line)
    # print('number of found lines', len(found_lines))
    return found_lines, bad_lines, line_count


def hough_lines(img, line_thresh, line_length, line_gap):
    # color_im = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    thresh = threshold_otsu(img)  # adaptive thresholding for lines
    binary = img > thresh  # binarize image
    binary = np.invert(binary)  # invert binary image
    skel = skeletonize(binary)  # skeletonize sk image method

    lines = probabilistic_hough_line(skel,  # using the p hough method from sk learn module
                                     threshold=line_thresh,
                                     line_length=line_length,
                                     line_gap=line_gap)

    ####### HOUGH EXAMPLE ################

    ##    from matplotlib import cm
    ##    h, theta, d = hough_line(skel)
    ##    fig, axes = plt.subplots(2, 3, figsize=(20, 8),
    ##                         subplot_kw={'adjustable': 'box-forced'})
    ##    ax = axes.ravel()
    ##
    ##    ax[0].imshow(skel, cmap=cm.gray)
    ##    ax[0].set_title('Input image')
    ##    ax[0].set_axis_off()
    ##
    ##    ax[1].imshow(np.log(1 + h),
    ##                 extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
    ##                 cmap=cm.gray, aspect=1/1.5)
    ##    ax[1].set_title('Hough transform')
    ##    ax[1].set_xlabel('Angles (degrees)')
    ##    ax[1].set_ylabel('Distance (pixels)')
    ##    ax[1].axis('image')
    ##
    ##    ax[2].imshow(skel, cmap=cm.gray)
    ##    line_count = 0
    ##    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    ##        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
    ##        y1 = (dist - skel.shape[1] * np.cos(angle)) / np.sin(angle)
    ##        print(angle, dist)
    ##        line_count += 1
    ##        ax[2].plot((0, skel.shape[1]), (y0, y1), '-r')
    ##    ax[2].set_xlim((0, skel.shape[1]))
    ##    ax[2].set_ylim((skel.shape[0], 0))
    ##    ax[2].set_xlabel('Line count = ' + str(line_count))
    ##    #ax[2].set_axis_off()
    ##    ax[2].set_title('Hough Lines')


    ####### HOUGH EXAMPLE ################


    skel = img_as_ubyte(skel)

    count = 0
    corrected_lines = []
    line_dict = []
    for line in lines:
        meta_lines = {}
        p0, p1 = line
        startX, startY = p0[0], p0[1]
        stopX, stopY = p1[0], p1[1]

        # apply correction to add line on main image from cropped image
        # correction is taken from height/ width of cropped image
        fix_startX = startX + corrections.x
        fix_startY = startY + corrections.y
        fix_stopX = stopX + corrections.x
        fix_stopY = stopY + corrections.y
        start = (fix_startX, fix_startY)
        end = (fix_stopX, fix_stopY)

        # add lines to fixed line dict
        corrected_lines.append([start, end])

        # line equations and add line info to line dictonary
        line_data = line_equations.Line(line)
        slope = line_data.slope()

        # get angle of line if not vert or horiz
        if (start[0] - end[0]) != 0 and (end[0] - start[0]) != 0 and \
                        (start[1] - end[1]) != 0 and (end[1] - start[1]) != 0:
            angle = np.degrees(np.arctan(float((start[1] - end[1])) / float((start[0] - end[0]))))
            angle = [angle if angle > 0 else 360 + angle][0]
        # angle is horiz
        elif (start[1] - end[1]) == 0 or (end[1] - start[1]) == 0:
            angle = 180
        # angle is vert
        elif (start[0] - end[0]) == 0 or (end[0] - start[0]) == 0:
            angle = 90

        # line metrics
        y_int = line_data.yintercept(slope)
        distance = [end[0] - start[0], end[1] - start[1]]
        midpoint = [(start[0] - end[0]) / 2, (start[1] - end[1]) / 2]
        meta_lines["start"] = start
        meta_lines["end"] = end
        meta_lines["slope"] = slope
        meta_lines["angle"] = angle
        meta_lines["yintercept"] = y_int
        meta_lines["points"] = start, end
        meta_lines["midpoint"] = midpoint

        norm = math.sqrt(distance[0] ** 2 + distance[1] ** 2)
        meta_lines["distance"] = norm
        direction = [distance[0] / norm, distance[1] / norm]
        meta_lines["unit-vector"] = direction
        line_dict.append(meta_lines)

        line_data = []

    # sort lines by angle
    line_dict.sort(key=itemgetter("angle"))
    # turn line dict into dataframe
    df = pd.DataFrame(line_dict)
    clone = np.zeros((800, 800, 3), np.uint8)
    clone.fill(255)
    distilled_lines = []
    # bin lines with slope within 20 degress
    bins = list(range(0, 361, 30))  # used for binning slopes withing 20 degrees
    groups = [x for x in range(len(bins) - 1)]
    categories = pd.cut(df['angle'], bins, labels=groups)
    df['categories'] = pd.cut(df['angle'], bins, labels=groups)
    df['scoresBinned'] = pd.cut(df['angle'], bins)
    grouped = df.groupby("categories")
    cluster_lines = [] # Placeholder for cluster groups

    return corrected_lines, cluster_lines

def sample_hough_lines(img):
    sample_hough = []
    img = img[:, :, 0]
    thresh = threshold_otsu(img)  # adaptive thresholding for lines
    binary = img > thresh  # binarize image
    binary = np.invert(binary)  # invert binary image
    skel = skeletonize(binary)  # skeletonize sk image method
    h, theta, d = hough_line(skel)

    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - skel.shape[1] * np.cos(angle)) / np.sin(angle)
        # print(angle, dist)
        # line_count += 1
        sample_hough.append([y0, y1])
    return sample_hough, skel

def shiCorners(img, corners, quality, distance):
    '''
    Corners with Shi Tomais corner detection
    '''
    fixed_corners = []
    thresh = threshold_otsu(img)
    binary = img > thresh
    binary = np.invert(binary)
    skel = skeletonize(binary)
    skel = img_as_ubyte(skel)
    goodcorners = cv2.goodFeaturesToTrack(skel, corners, quality, distance)
    goodcorners = np.int0(goodcorners)  # cast corners to int, needed to plot

    for i in goodcorners:
        X = i[0][0] + corrections.x  # correct on org image instead of crop
        Y = i[0][1] + corrections.y
        fixed_corners.append([X, Y])
    return fixed_corners


def gapCorners(img, corners):
    #corners = np.array([corners])
    black_pixel = black_pixels(img)
    #print(len(black_pixel))
    corner_connections = []
    corner_combo = itertools.combinations(corners, 2)
    #print(corner_combo)
    # for i in corner_combo:
    #     print('gaps', i)
    #     points = [tup for tup in itertools.product(i[0], i[1])]
    #     print(points)
    #     corner_connections.append(points[np.argmin([distance(Pa, Pb) for (Pa, Pb) in points])])

    gaps = []
    for i in corner_combo:
        single_gap = []
        x, y = i
        dist = (distance(x, y))
        if dist < 20.0:
            min_x = min(x[0], y[0])  # left
            min_y = min(x[1], y[1])  # top
            max_x = max(x[0], y[0])
            max_y = max(x[1], y[1])
            width = max(x[0], y[0]) - min_x
            height = max(x[1], y[1]) - min_y

            # if x[0] == y[0] or x[1] == y[1]:
            inside = [point for point in black_pixel if
                      (min_x <= point[0] <= min_x + width and min_y <= point[1] <= min_y + height)
                      or (min_x <= point[1] <= min_x + width and min_y <= point[0] <= min_y + height)]
            # else:
            #     inside = [point for point in edit_black if (min_x < point[0] < min_x + width and
            #           min_y < point[1] < min_y + height)
            #           or (min_x < point[1] < min_x + width and min_y < point[0] < min_y + height)]
            #print(inside)
            if len(inside) <= 3:
                single_gap.append((x[0], x[1]))
                single_gap.append((y[0], y[1]))

            if single_gap != []:
                gaps.append(single_gap)
    #print("GAPS", gaps)
    return gaps
